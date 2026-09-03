#!/usr/bin/env python
"""
动作反归一化统计量对拍 —— **不加载权重**，只读 config，十几秒。

    python scripts/check_norm_stats.py \
        --ours runs/<G0 的 run 目录>/dataset_statistics.json

问一个问题：**我们训练时算出来的 q01/q99，和官方 checkpoint 烘在权重里的
那一份，是不是同一个东西？**

为什么值得单独查：`predict_action` 的后半段是

    action = 0.5 * (z + 1) * (q99 - q01) + q01        （mask 为 True 的维度）

q01/q99 偏一点，**输出的动作幅度就整体偏掉**，而

  · loss 不受影响（它算在 token 上）
  · 动作 token 准确率不受影响（同上）
  · 不会抛任何异常

——正好是"token 准确率 0.705、成功率却只有 23.4%"这个组合的形状。
本项目已有的三道对拍（中心裁、视觉缓存、批量）全都在这一步**之前**。

⚠️ 这不是判据，是**定位**：对得上就排除这条，对不上就找到了。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

KEYS = ("q01", "q99", "mean", "std", "min", "max")


def _pick(ns, want: str, where: str) -> str:
    """
    ⚠️ **两边的键名本来就不同**：官方 finetuned checkpoint 存的是 `libero_10`，
    我们训练时写的是 `libero_10_no_noops`（数据集带 _no_noops 后缀）。
    `run_libero_eval_traj.py` 里那条回退正是为此。所以两边各自解析，
    不能拿一个键去查两处 —— 那只会得到"键不存在"，看不见真正要比的东西。
    """
    for k in (want, f"{want}_no_noops", want.replace("_no_noops", "")):
        if k in ns:
            return k
    raise SystemExit(f"{where} 里找不到 `{want}`（也试过加/去 _no_noops）；"
                     f"有的是：{list(ns)}")


def official(repo: str, key: str) -> tuple:
    """从 config 里取 norm_stats —— 不下载权重。"""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
    ns = getattr(cfg, "norm_stats", None)
    if ns is None:
        raise SystemExit(f"{repo} 的 config 里没有 norm_stats")
    k = _pick(ns, key, repo)
    return ns[k]["action"], k


def ours(path: Path, key: str) -> tuple:
    d = json.loads(path.read_text())
    # 训练脚本写的是 {key: {"action": {...}, "proprio": {...}}}；
    # 也见过直接就是 {"action": {...}} 的，两种都收
    if "action" in d:
        return d["action"], "(顶层)"
    k = _pick(d, key, str(path))
    if "action" not in d[k]:
        raise SystemExit(f"{path} 的 `{k}` 下找不到 action；有的是：{list(d[k])}")
    return d[k]["action"], k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="训练 run 目录下的 dataset_statistics.json")
    ap.add_argument("--repo", default="openvla/openvla-7b-finetuned-libero-10")
    ap.add_argument("--key", default="libero_10_no_noops")
    args = ap.parse_args()

    o, ko = official(args.repo, args.key)
    m, km = ours(Path(args.ours).expanduser(), args.key)

    print(f"官方: {args.repo}  键 `{ko}`")
    print(f"我们: {args.ours}  键 `{km}`\n")
    if ko != km:
        print(f"  ℹ️ 两边键名不同（`{ko}` vs `{km}`）—— 这本身正常，"
              f"数据集带 _no_noops 后缀而官方 checkpoint 不带。要比的是**值**。\n")

    worst = 0.0
    for k in KEYS:
        if k not in o or k not in m:
            print(f"  {k:>5}: 缺席（官方 {'有' if k in o else '无'} / "
                  f"我们 {'有' if k in m else '无'}）")
            continue
        a, b = np.asarray(o[k], float), np.asarray(m[k], float)
        if a.shape != b.shape:
            print(f"  {k:>5}: ❌ 维度不同 {a.shape} vs {b.shape}")
            worst = np.inf
            continue
        d = np.abs(a - b)
        rel = d / np.maximum(np.abs(a), 1e-8)
        worst = max(worst, float(rel.max()))
        flag = "✅" if rel.max() < 0.01 else ("🔶" if rel.max() < 0.05 else "❌")
        print(f"  {k:>5}: {flag} 最大相对差 {rel.max():.2%}（第 {int(rel.argmax())} 维）")
        if rel.max() >= 0.01:
            print(f"         官方 {np.round(a, 4)}")
            print(f"         我们 {np.round(b, 4)}")

    # ⚠️ mask 决定**哪些维度走反归一化**。夹爪那一维通常是 False（直接用
    #    bin center），错了的话夹爪开合会被当成连续量去缩放 —— 静默且致命。
    ma, mb = o.get("mask"), m.get("mask")
    print(f"\n  mask 官方 {ma}\n       我们 {mb}")
    if ma is not None and mb is not None and list(ma) != list(mb):
        print("  ❌ **mask 不同** —— 决定哪几维走反归一化，错了动作整体变形")
        worst = np.inf

    print()
    if worst is np.inf or worst >= 0.05:
        print("❌ 统计量对不上。G0 的动作在反归一化这一步就被缩放错了 —— "
              "而 loss 与 token 准确率完全看不出来。")
        sys.exit(1)
    elif worst >= 0.01:
        print("🔶 有差异但不大（<5%）。记下来，但多半不足以解释 24.8 点。")
    else:
        print("✅ 统计量一致。**这条排除**，24.8 点的缺口在别处。")


if __name__ == "__main__":
    main()
