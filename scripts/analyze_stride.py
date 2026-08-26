"""
定历史帧的采样间隔（stride）—— 阶段 C 开跑前必须先解决的第一件事。

    python scripts/analyze_stride.py --traj_dir results/trajectories --mode pixel
    python scripts/analyze_stride.py --mode feature --n_episodes 40   # 要 GPU

## 为什么必须先做

LIBERO 是 20 Hz。K=8 若取**连续帧**，"历史"只有 0.4 秒——这段时间机械臂几乎
没动，八帧近乎同一张图。用它跑「历史帧有没有用」的对照，得到"没用"是必然的，
而那是彻底的假阴性（见 docs/06 C.0 ①）。

## 判据：历史帧在多远之后就等同于"随便一帧"

不是问"变化有多大"（那随 Δt 单调增，给不出拐点），而是问：

    d(t, t−Δt)  相对  d(t, 同 episode 内随机一帧)

比值接近 1 时，该帧已不携带任何时序连续性——它与"从这条轨迹里随便抽一帧"
无法区分。**stride 应取在比值明显小于 1 的区间内**：太小则帧间几乎没有新信息，
太大则不再是"历史"而是"另一个场景"。

这个判据还顺带预测了 C-5（假历史对照）的可行性：若在选定的 stride 上比值已
接近 1，那么"打乱时序"与"真历史"本就无从区分，C-5 必然打平——
届时打平说明的是 stride 选错了，而不是历史帧没用。

## 两种模式

- `pixel`   逐 patch 平均绝对差。只用 CPU，快，够定量级
- `feature` DINOv2 倒数第二层 patch token 的余弦距离。要 GPU，
            与模型实际"看到"的东西一致，最终以它为准
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_redundancy import (  # noqa: E402
    IMG_SIZE, FeatureExtractor, get_libero_dummy_action, get_libero_env,
    get_libero_image,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def frame_distance(a, b, kind: str) -> float:
    """两帧之间的整体距离。pixel: 归一化平均绝对差；feature: 余弦距离。"""
    if kind == "pixel":
        return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)
    # feature: a,b 为 (N_PATCH, D)，先做 patch 级余弦距离再平均
    an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return float(1.0 - (an * bn).sum(-1).mean())


def collect_frames(traj: dict, task_suite):
    """回放一条轨迹取回全部帧。逻辑与 measure_redundancy.replay_episode 一致。"""
    task_id, ep_idx = traj["task_id"], traj["episode_idx"]
    task = task_suite.get_task(task_id)
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.seed(0)                                    # 与评测一致，勿改
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(task_id)[ep_idx])
    for _ in range(traj.get("num_steps_wait", 10)):
        obs, *_ = env.step(get_libero_dummy_action("openvla"))
    frames = []
    for st in traj["steps"]:
        frames.append(get_libero_image(obs, IMG_SIZE))
        obs, *_ = env.step(st["action_env"])
    env.close()
    return frames


def analyze(frames, deltas, kind: str, extractor=None, rng=None):
    """返回 {Δt: 平均距离} 与随机基线距离。"""
    reps = extractor.encode(frames) if extractor is not None else frames
    n = len(reps)
    out = {}
    for dt in deltas:
        if dt >= n:
            continue
        ds = [frame_distance(reps[t], reps[t + dt], kind) for t in range(n - dt)]
        out[dt] = float(np.mean(ds))

    # 随机基线: 同 episode 内任取两帧。这是"不携带任何时序连续性"的参照，
    # 也是 d(t, t−Δt) 在 Δt→∞ 时的渐近值
    pairs = min(200, n * (n - 1) // 2)
    idx = rng.integers(0, n, size=(pairs, 2))
    idx = idx[idx[:, 0] != idx[:, 1]]
    base = float(np.mean([frame_distance(reps[i], reps[j], kind) for i, j in idx]))
    return out, base, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="results/trajectories")
    ap.add_argument("--suite", default="libero_10",
                    help="默认 libero_10 —— 阶段 C 的主战场，episode 最长")
    ap.add_argument("--mode", default="pixel", choices=["pixel", "feature"])
    ap.add_argument("--n_episodes", type=int, default=40)
    ap.add_argument("--deltas", type=int, nargs="+",
                    default=[1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 40])
    ap.add_argument("--out", default="results/tables/stride_analysis.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from libero.libero import benchmark
    task_suite = benchmark.get_benchmark_dict()[args.suite]()

    files = sorted(Path(args.traj_dir).glob(f"*{args.suite}*/task*_ep*.json"))
    if not files:
        raise SystemExit(f"{args.traj_dir} 下没有 {args.suite} 的轨迹")
    rng = np.random.default_rng(args.seed)
    files = [files[i] for i in rng.permutation(len(files))[:args.n_episodes]]

    extractor = FeatureExtractor("dino") if args.mode == "feature" else None
    print(f"suite={args.suite}  模式={args.mode}  episode={len(files)}  Δt={args.deltas}")

    per_dt, bases, lens = {d: [] for d in args.deltas}, [], []
    for i, f in enumerate(files, 1):
        traj = json.loads(f.read_text())
        if len(traj["steps"]) < 12:
            continue
        frames = collect_frames(traj, task_suite)
        out, base, n = analyze(frames, args.deltas, args.mode, extractor, rng)
        for d, v in out.items():
            per_dt[d].append(v)
        bases.append(base)
        lens.append(n)
        print(f"  [{i}/{len(files)}] {f.parent.name[-12:]}/{f.stem}  {n} 帧", flush=True)

    base = float(np.mean(bases))
    rows = []
    for d in args.deltas:
        if not per_dt[d]:
            continue
        m = float(np.mean(per_dt[d]))
        rows.append({"delta_t": d, "seconds": d / 20.0, "distance": m,
                     "ratio_to_random": m / base, "n_episodes": len(per_dt[d])})

    print(f"\n{'='*70}")
    print(f"{args.suite} · {args.mode} · episode 平均 {np.mean(lens):.0f} 帧 "
          f"({np.mean(lens)/20:.1f} 秒)")
    print(f"随机两帧的基线距离: {base:.4f}")
    print(f"{'='*70}")
    print(f"{'Δt(帧)':>7} {'秒':>6} {'距离':>9} {'/随机基线':>10}  K=8 时的历史跨度")
    print("-" * 70)
    for r in rows:
        span = 7 * r["delta_t"] / 20.0        # K=8 -> 7 个间隔
        print(f"{r['delta_t']:>7} {r['seconds']:>6.2f} {r['distance']:>9.4f} "
              f"{r['ratio_to_random']:>10.3f}  {span:>5.1f} 秒")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"suite": args.suite, "mode": args.mode, "random_baseline": base,
         "mean_episode_frames": float(np.mean(lens)), "rows": rows}, indent=2))
    print(f"\n-> {args.out}")

    # 建议: 取比值落在 0.5~0.8 的区间 —— 帧间已有实质变化，但仍明显区别于随机帧
    #
    # ⚠️ **带内取最小，不取中位。** 初版取中位，是在发现「填充率」问题之前写的。
    # 0.5–0.8 是「信息量够不够」的**下界约束**，落在带内的都合格；
    # 带内选哪个，应由另一个目标决定 —— **最小化补帧**。
    # K 帧历史需要第 (K-1)*stride 帧之后才填得满，stride 越大，
    # episode 里「历史是补出来的」的步数越多。libero_10 实测（K=8）：
    #     stride=16 → 41% 的步骤填不满    stride=25 → 65%
    # 65% 已经不是「开头几步的边界情况」，而是主体状态。
    cand = [r for r in rows if 0.5 <= r["ratio_to_random"] <= 0.8]
    print("\n=== stride 建议 ===")
    if cand:
        pick = cand[0]                      # 带内最小 = 补帧最少
        print(f"  比值落在 0.5–0.8 的 Δt: {[r['delta_t'] for r in cand]}")
        print(f"  **取带内最小: stride = {pick['delta_t']}** "
              f"({pick['seconds']:.2f} 秒/帧，K=8 覆盖 {7*pick['seconds']:.1f} 秒)")
        print(f"  理由: 判据是下界约束，带内都合格；带内取最小以少补帧。")
        need = 7 * pick["delta_t"]
        print(f"  K=8 需第 {need} 帧后才填满历史 —— 对照各 suite 的 episode 长度"
              f"检查补帧占比（docs/06 §4.1 ①）")
    else:
        lo = [r for r in rows if r["ratio_to_random"] < 0.5]
        print(f"  没有 Δt 落在 0.5–0.8。扫描范围可能不够宽。")
        print(f"  比值 <0.5 的最大 Δt: {max([r['delta_t'] for r in lo], default='无')}")
        print(f"  若所有比值都 >0.8，说明连 Δt=1 都已接近随机帧——"
              f"那 C-5 的假历史对照会失效，需重新设计")
    print("\n注: 比值接近 1 表示该帧与'同 episode 内随便一帧'无法区分，")
    print("    已不携带时序连续性；此时 C-5（打乱时序）必然打平，")
    print("    而那说明的是 stride 选错，不是历史帧没用。")


if __name__ == "__main__":
    main()
