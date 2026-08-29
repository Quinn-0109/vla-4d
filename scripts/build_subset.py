#!/usr/bin/env python
"""
固定 episode 子集 + 训练集深度缓存 —— **G4 / M2 的前置**（`docs/06` §4.1 纪律 1b）。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/build_subset.py --suite libero_10 --limit 5      # 先冒烟
    python scripts/build_subset.py --suite libero_10                # 全量

一次遍历做三件事（都要渲染同一批帧，分两次跑是白花一倍的时间）：

  ① **覆盖率**：每条 episode 能不能对齐到某条 demo。实测已知 task 9 的两条都
     对不上，HF 上还缺 task 1 的 demo 文件 —— 覆盖率不是 100%。
  ② **固定子集**：能对上的列成清单。⚠️ **六组全部只用这份清单**：若 G4/M2 只用
     "能对上深度"的子集而 G2/G3 用全部，两者数据量不同，比较作废。
  ③ **深度缓存**：对齐后每个 RLDS 帧的 patch 级深度 (T,16,16)，G4/M2 的输入。

判据**先写在这里**，不看结果再定（`docs/06` §3.0 纪律 2）：

    初始帧误差 ≤ 5.0（平均绝对像素差，0–255）
    逐帧对齐 ≤5.0 的帧占 ≥ 90%
    匹配下标单调性 ≥ 95%          ← 对齐正确必然接近 100%，噪声不会

三条全过才收进子集。脚本会把三个量的**分布**打出来，好让"阈值定得合不合理"
这件事本身可被检查 —— 而不是只报一个通过率。

⚠️ **可续跑**：每条 episode 的结果即时追加进 `<out>/episodes.jsonl`，
   重跑会跳过已完成的。这台机器每 9 小时要关一次，不可续跑等于跑不完。

⚠️ **渲染走 EGL，占的是 GPU**。训练在跑的话会互相抢：要么等，
   要么 `MUJOCO_GL=osmesa` 走 CPU（慢很多但不抢卡）。
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库路径>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero.libero import benchmark  # noqa: E402

from check_replay import (align_by_state, build_env, demo_all_states,  # noqa: E402
                          diff_profile, find_task, match_init, rgb_of)
from data.depth_cache import hash_bytes  # noqa: E402

TOL = 5.0            # 平均绝对像素差
MIN_GOOD = 0.90      # 逐帧对齐里 ≤TOL 的帧占比
MIN_MONO = 0.95      # 匹配下标单调性


def episode_stream(data_root: str, name: str):
    """
    逐条产出 episode（**不能一次全读进内存**：libero_10 约 380 条 × 250 帧 ×
    256² × 3 ≈ 18 GB）。同时把 episode 级的元数据一并带出来 —— 训练时要靠它
    把深度查回去，见下面 `describe_features`。
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # ⚠️ **必须 SkipDecoding 拿到未解码的 JPEG 字节** —— 深度缓存的键就是它的
    #    指纹（`data/depth_cache`）。训练那边在轨迹变换阶段看到的也正是这份字节，
    #    两边指纹才对得上。解码后再哈希会因增广/重采样而对不上，且不报错。
    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    ds = b.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding()}}})
    for i, ep in enumerate(ds):
        raw = [s["observation"]["image"] for s in ep["steps"]]
        yield i, {
            "meta": {k: v.numpy().decode() if v.dtype == tf.string else v.numpy()
                     for k, v in ep["episode_metadata"].items()}
            if "episode_metadata" in ep else {},
            "lang": next(iter(ep["steps"]))["language_instruction"]
            .numpy().decode().strip().lower(),
            "images": np.stack([tf.io.decode_image(r, expand_animations=False)
                                .numpy() for r in raw]),
            "hash": np.array([int(hash_bytes([r])[0].numpy()) for r in raw],
                             dtype=np.uint64),
        }


def describe_features(data_root: str, name: str) -> None:
    """
    ⚠️ **先回答一个会决定整个设计的问题：训练时拿什么把深度查回去？**

    RLDS 在训练时是打乱 + 交错读的，位置下标没有意义；而 `KFrameBatchTransform`
    拿到的图像**已经过增广**，按内容做指纹也不成立。所以键必须是 episode 级
    元数据里某个**唯一**的字段，并在轨迹变换里一路带到 batch transform。
    这里先把有什么打出来，再定 —— 猜错了就是又一份对不上的缓存。
    """
    import tensorflow_datasets as tfds

    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    print("episode 级字段（steps 之外）:")
    for k, v in b.info.features.items():
        if k != "steps":
            print(f"  {k}: {v}")
    ep = next(iter(b.as_dataset(split="train", shuffle_files=False).take(1)))
    for k, v in ep.items():
        if k == "steps":
            continue
        print(f"  第 0 条的值 {k} = {v}")


def judge(d0: float, good: float, mono: float) -> tuple[bool, str]:
    if d0 > TOL:
        return False, f"初始帧对不上（{d0:.2f} > {TOL}）"
    if good < MIN_GOOD:
        return False, f"逐帧对齐只有 {good:.1%} < {MIN_GOOD:.0%}"
    if mono < MIN_MONO:
        return False, f"匹配下标单调性 {mono:.1%} < {MIN_MONO:.0%}"
    return True, "ok"


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 管道下的块缓冲，第三次了
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--data_root", default=str(
        Path("~/autodl-tmp/datasets/modified_libero_rlds").expanduser()))
    ap.add_argument("--out", default="results/subset")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--align_stride", type=int, default=2)
    ap.add_argument("--n_init_try", type=int, default=50)
    ap.add_argument("--demo_frames", type=int, default=40)
    ap.add_argument("--features_only", action="store_true",
                    help="只打印 episode 级字段就退出（决定深度用什么键查）")
    args = ap.parse_args()
    name = f"{args.suite}_no_noops"

    if args.features_only:
        describe_features(args.data_root, name)
        return

    out = Path(args.out) / args.suite
    (out / "depth").mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    done = {json.loads(l)["ep"] for l in log.open()} if log.exists() else set()
    if done:
        print(f"续跑：已完成 {len(done)} 条，跳过")

    bmark = benchmark.get_benchmark_dict()[args.suite]()
    env, env_tid = None, None
    try:
        for i, ep in episode_stream(args.data_root, name):
            if args.limit and i >= args.limit:
                break
            if i in done:
                continue
            tid = find_task(bmark, ep["lang"])
            rec = {"ep": i, "task": tid, "lang": ep["lang"],
                   "n_frames": int(len(ep["images"])), "meta": ep["meta"]}
            if tid is None:
                rec.update(ok=False, why="任务描述对不上任何 task")
                _append(log, rec)
                continue

            # ⚠️ 同一个 task 的 env 复用：建一次要编译 MuJoCo 模型，几秒起步，
            #    每条 episode 重建就是白花一小时。RLDS 按 task 分文件存，
            #    顺序读时切换很少 —— 真切换了也只是慢一点，不影响正确性。
            if env is None or tid != env_tid:
                if env is not None:
                    env.close()
                env = build_env(bmark.get_task(tid))
                env.seed(0)
                env_tid = tid

            m = match_init(env, bmark, tid, ep["images"][0], args.n_init_try,
                           args.demo_frames)
            env.reset()
            obs = env.set_init_state(m["state"])
            prof = diff_profile(rgb_of(obs, True), ep["images"][0])
            rec.update(d0=m["best"], d1=m["second"], src=m["src"],
                       frame_off=m.get("frame_off", 0), p50=prof["p50"],
                       top10=prof["top10_share"])

            key = m["src"].split(":", 1)[1] if m["src"].startswith("demo:") else None
            st_all = demo_all_states(bmark.get_task(tid), key) if key else None
            if m["best"] > TOL or st_all is None:
                rec.update(ok=False, why=("初始帧对不上" if m["best"] > TOL
                                          else "取不到该 demo 的全帧状态"))
                _append(log, rec)
                continue

            errs, mj, idxs, dep = align_by_state(env, st_all, ep["images"],
                                                 True, args.align_stride)
            good = float((errs <= TOL).mean())
            mono = float((np.diff(idxs[mj]) >= 0).mean())
            ok, why = judge(m["best"], good, mono)
            rec.update(good=good, mono=mono, ok=ok, why=why, demo=key)
            if ok:
                # 对齐后每个 RLDS 帧的 patch 级深度 (T,16,16)，按帧指纹存。
                # float16 足够：分辨率 ~1 mm，而体素箱宽约 30 cm、跳变阈值 5 cm。
                np.savez(out / "depth" / f"ep{i:04d}.npz",
                         hash=ep["hash"], depth=dep[mj].astype(np.float16))
            _append(log, rec)
            print(f"  [{i:>4}] task {tid}  d0={m['best']:.2f}  "
                  f"对齐 {good:.0%}  单调 {mono:.0%}  → {'✅' if ok else '❌ ' + why}")
    finally:
        if env is not None:
            env.close()

    summarize(log, out)


def _append(log: Path, rec: dict) -> None:
    with log.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(log: Path, out: Path) -> None:
    recs = [json.loads(l) for l in log.open()]
    ok = [r for r in recs if r.get("ok")]
    print(f"\n=== 覆盖率：{len(ok)}/{len(recs)} = {len(ok) / max(len(recs), 1):.1%} ===")

    by_task: dict = {}
    for r in recs:
        t = by_task.setdefault(r.get("task"), [0, 0])
        t[1] += 1
        t[0] += bool(r.get("ok"))
    print(f"{'task':>6} {'通过':>8} {'占比':>7}")
    for t in sorted(by_task, key=lambda x: (x is None, x)):
        a, n = by_task[t]
        print(f"{str(t):>6} {a:>4}/{n:<3} {a / n:>7.0%}")

    # ⚠️ 报分布，不只报通过率 —— 阈值定得合不合理，只有看见分布才判得了
    for k, lo in (("d0", None), ("good", None), ("mono", None)):
        v = np.array([r[k] for r in recs if k in r], dtype=float)
        if len(v):
            print(f"  {k:>5} 分位 p10={np.percentile(v, 10):.3f} "
                  f"p50={np.median(v):.3f} p90={np.percentile(v, 90):.3f}")
    why: dict = {}
    for r in recs:
        if not r.get("ok"):
            why[r.get("why", "?")] = why.get(r.get("why", "?"), 0) + 1
    if why:
        print("  失败原因：" + "  ".join(f"{k} ×{v}" for k, v in why.items()))

    # 合并成一张表，训练侧只读这一个文件（DepthCache.load）
    parts = sorted((out / "depth").glob("ep*.npz"))
    if parts:
        hs, ds_ = [], []
        for f in parts:
            z = np.load(f)
            hs.append(z["hash"])
            ds_.append(z["depth"])
        h, d = np.concatenate(hs), np.concatenate(ds_)
        u, first = np.unique(h, return_index=True)
        if len(u) != len(h):
            print(f"  ⚠️ {len(h) - len(u)} 帧指纹重复（同一帧被两条 episode 收录？）"
                  "，只保留首次出现的")
            h, d = h[np.sort(first)], d[np.sort(first)]
        np.savez_compressed(out / "depth_cache.npz", hash=h, depth=d)
        print(f"深度缓存 -> {out / 'depth_cache.npz'}  {len(h)} 帧，"
              f"{(out / 'depth_cache.npz').stat().st_size / 2**20:.0f} MB")

    sub = out / "subset.json"
    sub.write_text(json.dumps({
        "criteria": {"tol": TOL, "min_good": MIN_GOOD, "min_mono": MIN_MONO},
        "n_total": len(recs), "n_ok": len(ok),
        "episodes": [{"ep": r["ep"], "task": r["task"], "demo": r.get("demo")}
                     for r in ok],
    }, ensure_ascii=False, indent=1))
    print(f"\n子集清单 -> {sub}")
    print("⚠️ 这份清单要连同结果一起发布 —— 它是复现的一部分（docs/06 §4.1 1b）。")
    print("⚠️ **六组全部只用这份清单**，只给 G4/M2 用会让数据量不同，比较作废。")


if __name__ == "__main__":
    main()
