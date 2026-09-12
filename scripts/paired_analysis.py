#!/usr/bin/env python
"""
2×2 的配对分析 —— McNemar + 主效应分解。

    python scripts/paired_analysis.py

输入是 `results/logs/EVAL-*.episodes.jsonl`（每局一行 `{arm, task_id, episode,
success}`）。四臂跑的是**同一批确定性初始状态**，`(task_id, episode)` 就是配对键。

⚠️ **为什么必须配对，以及它实际帮了多少。** 聚合后的 task 级成功数**恢复不出**
   配对表（b/c 两格永远拿不回来），所以逐局落盘是这一步的前提
   （`docs/05` §13.5 ⑤）。但 **G3/G4 实测的收益远小于事前估计**：
   不一致对 152/450 = **33.8%**，McNemar 下 SE = √(b+c)/n = **2.74 点**，
   而非配对是 2.98 —— 只降了 8%，不是 §8.5c 假设的 35%（3.1 → 2.0）。
   两臂在三分之一的局上给出不同结果，**逐局几乎独立**，配对就没什么可消的。
   所以本脚本**逐对打印配对 SE**：不打印它，就会以为"做了配对所以更灵敏"。

⚠️ **主分析是 9-task（排除 task 1）**，事前登记于 `docs/05` §12.3b：
   固定子集只覆盖了 task 1 的 4%，四臂在它上面全是 0/50，而 G2 有 48% ——
   那一格没有训练数据。10-task 只用于与文献并列。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ARMS = ("G3", "M3", "M2", "G4")
EXCLUDE_TASKS = (1,)                  # 事前登记，见模块 docstring
LOGDIR = Path("results/logs")


def load() -> dict:
    """arm → {(task_id, episode): success}。同一臂多份文件时后写的覆盖先写的。"""
    out = {a: {} for a in ARMS}
    files = sorted(LOGDIR.glob("EVAL-*.episodes.jsonl"),
                   key=lambda f: f.stat().st_mtime)
    for f in files:
        for line in f.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("arm") in out:
                out[r["arm"]][(int(r["task_id"]), int(r["episode"]))] = int(r["success"])
    return out


def mcnemar(base: dict, treat: dict, keys) -> tuple:
    """
    返回 `(n_base, n_treat, z, p)`：

      n_base  = 只有 base 对（treat 错）的局数
      n_treat = 只有 treat 对（base 错）的局数
      z       > 0 表示 **treat 更好**

    ⚠️ **z 的符号必须与显示的 Δ 同向。** 初版按 `n10 - n01` 取符号，
       结果 Δ=+2.67（treat 更好）却打出 z=−1.40 —— 一张自相矛盾的表比没有表更坏。
    """
    n_base = sum(1 for k in keys if base[k] and not treat[k])
    n_treat = sum(1 for k in keys if treat[k] and not base[k])
    m = n_base + n_treat
    if m == 0:
        return n_base, n_treat, 0.0, 1.0
    # 连续性校正的 McNemar；m 小的时候它保守，正是我们要的方向
    chi = (abs(n_treat - n_base) - 1) ** 2 / m if m > 1 else 0.0
    z = math.copysign(math.sqrt(max(chi, 0.0)), n_treat - n_base)
    p = math.erfc(abs(z) / math.sqrt(2))
    return n_base, n_treat, z, p


def main() -> int:
    data = load()
    have = [a for a in ARMS if data[a]]
    missing = [a for a in ARMS if not data[a]]
    if missing:
        print(f"⚠️ 缺逐局数据的臂: {', '.join(missing)}")
        print("   逐局 JSONL 是 2026-09 才加的，更早跑的评测没有它。")
        print("   补跑（每臂约 2.6 h）：")
        for a in missing:
            print(f"     A=$(ls -d runs/*{a}*sub255*/adapter/step30000); "
                  f"python scripts/run_eval_kframe.py --arm {a} --adapter \"$A\" "
                  f"--num_trials_per_task 50 --eval_batch 8 --run_note sub255")
        print("   **不补就只能做非配对比较**，SE 约 3.0 而非 ~2.0，"
              "而 2×2 的差值正是 1–6 点这个量级。\n")
    if len(have) < 2:
        print("至少要两臂才能配对。")
        return 1

    # 配对键 = 所有臂都有的局，且不在排除列表里
    keys = set.intersection(*(set(data[a]) for a in have))
    keys = {k for k in keys if k[0] not in EXCLUDE_TASKS}
    if not keys:
        print("没有共同的局 —— 检查 (task_id, episode) 是否对得上。")
        return 1
    keys = sorted(keys)
    print(f"配对样本：{len(keys)} 局（9-task，已排除 task {EXCLUDE_TASKS}）"
          f"，参与的臂：{', '.join(have)}\n")

    p = {a: sum(data[a][k] for k in keys) / len(keys) for a in have}
    print("成功率")
    for a in have:
        print(f"  {a:<4}{p[a]:.4f}  ({sum(data[a][k] for k in keys)}/{len(keys)})")

    pairs = [("G4", "G3", "L1  主命题：度量 4D vs 图像网格"),
             ("G4", "M2", "L3a 只在池化侧用度量不够"),
             ("G4", "M3", "L3b 只在 PE 侧用度量不够"),
             ("M2", "G3", "池化侧换度量（PE 固定网格）"),
             ("M3", "G3", "PE 侧换度量（池化固定网格）")]
    print(f"\n配对检验（McNemar，连续性校正）")
    print(f"{'对照（Δ = 前者 − 后者）':<34}{'Δ点':>7}{'一致':>7}"
          f"{'仅后':>6}{'仅前':>6}{'SE':>7}{'z':>7}{'p':>9}")
    for a, b, desc in pairs:
        if a not in have or b not in have:
            print(f"{desc:<34}{'—— 缺臂 ——':>36}")
            continue
        n_b, n_a, z, pv = mcnemar(data[b], data[a], keys)    # b 基准，a 处理
        agree = len(keys) - n_b - n_a
        # ⭐ **把配对 SE 打出来。** McnEmar 下 Δ 的标准误是 √(b+c)/n，它只依赖
        #    不一致对的数量 —— 不一致越多，配对越接近非配对。§8.5c 事前假设
        #    配对能把 SE 从 3.1 压到 ~2.0，而 G3/G4 实测不一致率 34%，
        #    SE 只从 2.98 降到 2.74（收益 8%，不是 35%）。**不打印这个数，
        #    就会以为"做了配对所以更灵敏"，而实际上几乎没有。**
        se = math.sqrt(n_b + n_a) / len(keys) * 100 if (n_b + n_a) else float("nan")
        print(f"{desc:<34}{(p[a]-p[b])*100:+7.2f}{agree:>7}{n_b:>6}{n_a:>6}"
              f"{se:>7.2f}{z:>7.2f}{pv:>9.4f}")

    if len(have) == 4:
        pool = (p["M2"] + p["G4"]) / 2 - (p["G3"] + p["M3"]) / 2
        pe = (p["M3"] + p["G4"]) / 2 - (p["G3"] + p["M2"]) / 2
        inter = p["G4"] - p["M2"] - p["M3"] + p["G3"]
        print("\n主效应分解（2×2 边际）")
        print(f"  池化侧用度量坐标   {pool * 100:+6.2f} 点")
        print(f"  PE 侧用度量坐标    {pe * 100:+6.2f} 点")
        print(f"  交互（超加性）     {inter * 100:+6.2f} 点")
        print("  ⚠️ **两次单独胜出不等于超加性** —— 要声称"
              "『超出两个独立收益之和』，看的是交互这一行本身。")

    print("\n⚠️ 判读规则事前登记于 `docs/06` §3.2 / §3.3，**不得按结果调整**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
