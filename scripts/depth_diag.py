#!/usr/bin/env python
"""
深度诊断 —— **主线开跑前唯一能证伪核心创新的一步**（docs/06 §4.5）。

要测的不是"场景有多立体"，而是 §1.2① 那个机制的**发生率**：

    同一 (h,w) 格子，在 K 帧的历史窗口内，深度 max−min 超过阈值的比例

这直接对应"图像网格会把不同物理表面错误合并"这件事发生得有多频繁。
**跳变率过低 → G3 的错误合并根本不发生 → G4 相对 G3 没什么可赢**，
那时要立刻把叙事重心从"z 区分表面"转到"跨帧时间一致性"，并写明适用边界。

⚠️ **方差高 ≠ 机制会触发。** 桌面场景本身就立体（近处物体 / 桌面 / 背景），
patch 深度的分位差大概率不小；但若机械臂从不穿越视线，跳变率仍可以接近 0。
所以主判据是**跳变率**，深度跨度只作辅助（它决定体素分箱分不分得开）。

顺带把 patch 级深度缓存下来（256 值/帧），G4/M2 的 metric_coords 直接用。

    python scripts/depth_diag.py --suite libero_10 --n_episodes 20

⚠️ **这一步不等于 docs/06 §4.2 的"回放保真度验证"。**
本脚本回放的是**评测轨迹**（我们自己录的 action_env，评测确定性保证可复现），
用来测场景统计足够。而 §4.2 要验的是**回放能否复现 RLDS 训练数据的画面**——
那是另一件事，在给训练集缓存深度之前必须单独做。别把这两件混为一谈。
"""

from __future__ import annotations

import argparse
import glob
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

from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
)

IMG_SIZE = 224
PATCH = 14
GRID = IMG_SIZE // PATCH          # 16

# 判据：同一 (h,w) 在窗口内深度跨度超过它就算"跳变"。
# 5 cm 的量级依据：LIBERO 的物体高度多在 5–15 cm，夹爪穿越视线造成的
# 前后景深度差远大于此；而同一表面因视角造成的深度变化远小于此。
JUMP_THRESH_M = 0.05


def build_env_with_depth(task, resolution: int = 256):
    """
    自己造带深度的环境 —— 官方 `get_libero_env` 不开 `camera_depths`。

    其余参数必须与官方一致（相机名、分辨率、控制器），否则回放出的画面
    与评测对不上，测出来的统计就不是评测里那个场景。
    """
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(**{
        "bddl_file_name": bddl,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,          # ← 唯一的增量
    })
    return env


def real_depth(env, d: np.ndarray) -> np.ndarray:
    """
    OpenGL 深度缓冲 → **米**。

    ⚠️ 这一步不能省。robosuite 返回的 `*_depth` 是归一化到 [0,1] 的深度缓冲，
    **不是米**，而且与真实距离是非线性关系。直接拿它当 z 用，
    度量坐标就不是度量的了 —— 而且不会报错，只会让 G4 悄悄退化。
    """
    try:
        from robosuite.utils.camera_utils import get_real_depth_map
        return get_real_depth_map(env.sim, d)
    except Exception:
        # 手算兜底：z = near / (1 − d·(1 − near/far))
        m = env.sim.model
        extent = m.stat.extent
        near = m.vis.map.znear * extent
        far = m.vis.map.zfar * extent
        return near / (1.0 - d * (1.0 - near / far))


def to_patch_depth(depth_hw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    全分辨率深度 → patch 级 (16, 16)。返回 (中位数, 均值)。

    **两个都存**，因为它们服务于不同目的、且各有各的对：
    - **中位数**：诊断用。patch 跨在深度不连续处时，均值落在两个表面之间、
      哪个都不是；中位数取多数那个表面，稳。
    - **均值**：`metric_coords` 用。与特征侧的均值池化保持同一套平均逻辑
      （docs/06 §3.0.5 ①），两侧一致才谈得上"共用坐标系"。

    存两份的成本是每 suite 多 38 MB，远小于日后为这个选择重跑一次的代价。
    """
    if depth_hw.shape[0] != IMG_SIZE:
        from PIL import Image
        depth_hw = np.array(Image.fromarray(depth_hw.astype(np.float32), mode="F")
                            .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))
    blocks = depth_hw.reshape(GRID, PATCH, GRID, PATCH).transpose(0, 2, 1, 3)
    flat = blocks.reshape(GRID, GRID, PATCH * PATCH)
    return np.median(flat, axis=-1), flat.mean(axis=-1)


def replay_depth(traj: dict, task_suite) -> np.ndarray | None:
    """回放一条评测轨迹，返回 (T, 16, 16, 2) 的 patch 深度：[..., 0]=中位数 [..., 1]=均值。"""
    task = task_suite.get_task(traj["task_id"])
    env = build_env_with_depth(task)
    env.seed(0)                                   # 与评测一致，勿改
    env.reset()
    obs = env.set_init_state(
        task_suite.get_task_init_states(traj["task_id"])[traj["episode_idx"]])
    for _ in range(traj.get("num_steps_wait", 10)):
        obs, *_ = env.step(get_libero_dummy_action("openvla"))

    out = []
    for st in traj["steps"]:
        d = obs["agentview_depth"]
        if d.ndim == 3:
            d = d[..., 0]
        # ⚠️ 与 RGB 同样翻转 180°。`get_libero_image` 对 RGB 做了这个翻转
        # （官方训练/测试都翻），深度不跟着翻，深度与 patch 就错位了 ——
        # 这种错位不会报错，只会让 z 对应到别的 patch 上。
        dm = real_depth(env, d)[::-1, ::-1]
        med, avg = to_patch_depth(dm)
        out.append(np.stack([med, avg], axis=-1))
        obs, *_ = env.step(st["action_env"])
    env.close()
    return np.asarray(out, dtype=np.float32) if out else None


def diagnose(dep: np.ndarray, k: int, stride: int, thresh: float) -> dict:
    """
    dep: (T, 16, 16) 单通道 patch 深度。

    对每个时刻 t，取历史窗口 [t−(K−1)·s, …, t]（不足则跳过），
    统计有多少格子在窗口内 max−min > thresh。
    """
    T = dep.shape[0]
    span = (k - 1) * stride
    if T <= span:
        return {}
    jump, cells, spans = 0, 0, []
    for t in range(span, T):
        w = dep[t - span:t + 1:stride]                   # (K, 16, 16)
        rng = w.max(axis=0) - w.min(axis=0)
        jump += int((rng > thresh).sum())
        cells += rng.size
        spans.append(float(np.percentile(dep[t], 95) - np.percentile(dep[t], 5)))
    return {"jump_cells": jump, "total_cells": cells,
            "scene_span_p5_95": float(np.mean(spans)),
            "n_windows": T - span}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="results/trajectories")
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16,   # docs/06 §4.1 feature 定稿
                    help="历史帧间隔。默认 16 = feature 模式定下的值")
    ap.add_argument("--thresh", type=float, default=JUMP_THRESH_M)
    ap.add_argument("--cache_dir", default="results/depth_cache")
    ap.add_argument("--out", default="results/tables/depth_diag.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.traj_dir}/**/{args.suite}*.json", recursive=True))
    if not files:
        files = sorted(glob.glob(f"{args.traj_dir}/**/*.json", recursive=True))
        files = [f for f in files if args.suite in f]
    if not files:
        raise SystemExit(f"在 {args.traj_dir} 下找不到 {args.suite} 的轨迹")

    rng = np.random.default_rng(args.seed)
    if len(files) > args.n_episodes:
        files = [files[i] for i in rng.choice(len(files), args.n_episodes, replace=False)]

    bmark = benchmark.get_benchmark_dict()[args.suite]()
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    tot_jump = tot_cells = tot_win = 0
    spans, per_ep = [], []
    print(f"{args.suite} · K={args.K} · stride={args.stride} · 阈值 {args.thresh*100:.0f} cm\n")

    for i, f in enumerate(files):
        traj = json.loads(Path(f).read_text())
        dep = replay_depth(traj, bmark)
        if dep is None:
            continue
        np.save(Path(args.cache_dir) / f"{args.suite}_t{traj['task_id']}_e{traj['episode_idx']}.npy",
                dep.astype(np.float16))          # fp16：深度精度到毫米足够，省一半盘
        st = diagnose(dep[..., 0], args.K, args.stride, args.thresh)
        if not st:
            print(f"  [{i+1}/{len(files)}] 帧数 {dep.shape[0]} < 窗口 {(args.K-1)*args.stride+1}，跳过")
            continue
        r = st["jump_cells"] / st["total_cells"]
        tot_jump += st["jump_cells"]; tot_cells += st["total_cells"]; tot_win += st["n_windows"]
        spans.append(st["scene_span_p5_95"]); per_ep.append(r)
        print(f"  [{i+1}/{len(files)}] task {traj['task_id']} ep {traj['episode_idx']}  "
              f"帧 {dep.shape[0]:>3}  跳变率 {r:>6.1%}  深度跨度 {st['scene_span_p5_95']:.3f} m")

    if not per_ep:
        raise SystemExit(
            f"没有一条 episode 够长（需 {(args.K-1)*args.stride+1} 帧）。\n"
            f"这与 docs/06 §4.1 ① 的填充率问题同源：K=8、stride=16 要 113 帧，\n"
            f"而 spatial(106)/goal(103) 的成功 episode 整条都不够，只有 10(388)/object(140) 够。\n"
            f"减小 --stride 或 --K，或换 --suite libero_10。")

    rate = tot_jump / tot_cells
    span = float(np.mean(spans))
    res = {"suite": args.suite, "K": args.K, "stride": args.stride,
           "thresh_m": args.thresh, "n_episodes": len(per_ep),
           "n_windows": tot_win, "jump_rate": rate,
           "jump_rate_per_episode": per_ep, "scene_span_p5_95_m": span}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))

    print(f"\n{'='*64}\n跨帧深度跳变率: **{rate:.1%}**  "
          f"（{len(per_ep)} episode，{tot_win} 个窗口）")
    print(f"episode 间范围: {min(per_ep):.1%} – {max(per_ep):.1%}")
    print(f"场景深度跨度 (p5–p95): {span:.3f} m")
    print(f"-> {args.out}\n-> 深度缓存 {args.cache_dir}/\n")

    print("=== 判读（docs/06 §4.5）===")
    if rate < 0.05:
        print(f"  ❌ 跳变率 {rate:.1%} < 5% —— **§1.2① 的错误合并机制基本不触发**。")
        print("     G4 相对 G3 缺少物理基础，G4≈G3 的风险很高。")
        print("     立即行动：把叙事重心从「z 区分表面」转到「跨帧时间一致性(t 轴)」，")
        print("     并在论文里写明适用边界——本方法在深度变化显著的场景生效，")
        print("     纯平面运动时退化为图像网格池化。**这是提前设边界，不是事后找补。**")
    elif rate < 0.15:
        print(f"  🔶 跳变率 {rate:.1%}，偏低但非零。机制会触发，只是不频繁。")
        print("     主线照跑，但**分层分析必须做**：收益应集中在跳变发生的时刻，")
        print("     若均匀分布则更像正则效应而非坐标系的作用（§3.1 预测 3）。")
    else:
        print(f"  ✅ 跳变率 {rate:.1%} —— 机制频繁触发，G4 vs G3 有物理基础。")

    # 体素分箱能不能分开，取决于跨度相对分箱分辨率
    cell = span / 6.0        # G4 空间轴约 5–7 个箱（coord_pool 实测）
    print(f"\n  深度跨度 {span:.3f} m / 约 6 个体素 = 每箱 {cell*100:.1f} cm")
    if cell < 0.03:
        print("  ⚠️ 每箱不到 3 cm，接近深度噪声量级，度量分箱可能分不开 ——"
              "需调 metric_extent 的包围盒或减少 z 轴箱数")


if __name__ == "__main__":
    main()
