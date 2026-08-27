#!/usr/bin/env python
"""
体素分箱能不能分开 —— `depth_diag.py` 的续算，**不重新渲染**，只读深度缓存。

`depth_diag` 回答的是"错误合并机制发不发生"（跳变率）。
它不回答"G4 分箱之后能不能把这些跳变分开"——而这两件事用的是**两套尺度**：

    诊断阈值   5 cm      ← 判定"同一格子里有两个表面"
    分箱宽度   跨度 / 箱数 ← G4 实际的 z 分辨率

冒烟跑出来是 5 cm vs 29.7 cm，差 6 倍。**若跳变幅度普遍小于箱宽，
G4 的 z 轴就分不开它本该分开的东西，G4≈G3，而这不会报任何错。**
所以主线开跑前要把跳变幅度的**分布**看清楚，而不是只看一个发生率。

    python scripts/depth_bins.py

⚠️ 这里用 patch 深度的**均值**通道（`metric_coords` 用的那个），
不是 `depth_diag` 判据用的中位数通道 —— 分箱侧要看分箱侧的量。
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np

THRESHOLDS_CM = [1, 2, 5, 10, 20, 30, 50, 100]


def windows(dep: np.ndarray, k: int, stride: int) -> np.ndarray:
    """(T,256) -> 每个窗口每个格子的深度跨度 (n_win, 256)。不够长返回空。"""
    span = (k - 1) * stride
    if dep.shape[0] <= span:
        return np.empty((0, dep.shape[1]), dtype=np.float32)
    out = []
    for t in range(span, dep.shape[0]):
        w = dep[t - span:t + 1:stride]
        out.append(w.max(axis=0) - w.min(axis=0))
    return np.asarray(out, dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="results/depth_cache")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--z_bins", type=int, default=6,
                    help="G4 的 z 轴箱数（coord_pool 实测空间轴约 5-7 箱）")
    ap.add_argument("--jump_thresh_cm", type=float, default=5.0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.cache_dir}/*.npy"))
    if not files:
        raise SystemExit(f"{args.cache_dir} 下没有深度缓存，先跑 scripts/depth_diag.py")

    dvals, rngs = [], []
    for f in files:
        a = np.load(f).astype(np.float32)          # (T,16,16,2)
        d = a[..., 1].reshape(a.shape[0], -1)      # 均值通道 → (T,256)
        dvals.append(d.ravel())
        r = windows(d, args.K, args.stride)
        if r.size:
            rngs.append(r.ravel())
    if not rngs:
        raise SystemExit(f"没有 episode 够 {(args.K-1)*args.stride+1} 帧")

    dv = np.concatenate(dvals)
    rg = np.concatenate(rngs)
    print(f"{len(files)} 条 episode · {dv.size:,} 个 patch 深度值 · "
          f"{rg.size:,} 个窗口格子\n")

    # ---- 1. 深度分布：跨度到底是谁撑起来的
    qs = [1, 5, 25, 50, 75, 90, 95, 99]
    pv = np.percentile(dv, qs)
    print("深度分布 (m)")
    print("  " + "  ".join(f"p{q}={v:.2f}" for q, v in zip(qs, pv)))
    p5, p95 = pv[1], pv[6]
    print(f"  p5–p95 跨度 = {p95 - p5:.3f} m")
    for cut in (1.0, 1.5, 2.0, 3.0):
        frac = float((dv > cut).mean())
        if frac > 0.001:
            print(f"  > {cut:.1f} m 的格子占 {frac:.1%}"
                  + ("   ← 多半是背景/远平面" if cut >= 1.5 and frac > 0.05 else ""))

    # ---- 2. 跳变率随阈值怎么衰减
    print(f"\n跳变率 vs 阈值 (K={args.K}, stride={args.stride})")
    for c in THRESHOLDS_CM:
        print(f"  > {c:>3} cm : {float((rg > c / 100).mean()):>6.1%}")

    # ---- 3. 关键：跳变幅度 vs 箱宽
    #    这是全脚本唯一真正的判据。前两节只是解释它为什么是这个值。
    for name, span in (("全跨度 p5–p95", p95 - p5),
                       ("剪掉背景 p5–p90", pv[5] - p5)):
        w = span / args.z_bins
        sep = float((rg > w).mean())
        print(f"\n[{name}] 跨度 {span:.3f} m / {args.z_bins} 箱 = 每箱 {w*100:.1f} cm")
        print(f"  跳变幅度 > 箱宽的格子: **{sep:.1%}**  ← G4 的 z 轴真能分开的部分")
        need = int(np.ceil(span / (args.jump_thresh_cm / 100)))
        print(f"  要让箱宽 ≤ {args.jump_thresh_cm:.0f} cm（诊断阈值），z 轴需要 {need} 箱")

    # ---- 4. 判读
    w = (p95 - p5) / args.z_bins
    sep = float((rg > w).mean())
    print(f"\n{'='*64}\n=== 判读 ===")
    if sep >= 0.15:
        print(f"  ✅ {sep:.1%} 的格子跳变幅度超过箱宽，均匀分箱够用，照原计划。")
    elif sep >= 0.05:
        print(f"  🔶 只有 {sep:.1%} 超过箱宽。均匀分箱把大部分跳变吃掉了。")
        print("     两个方向（都不增加 token 预算）：")
        print("     ① 收紧 z 包围盒到工作空间（剪掉背景），箱宽直接变细；")
        print("     ② z 轴改**分位数分箱**——箱边界由训练集深度分布算一次、固定成常量，")
        print("        不是每个样本各算各的（那会毁掉跨臂可比性）。")
    else:
        print(f"  ❌ 只有 {sep:.1%} 超过箱宽。**当前配置下 G4 的 z 轴基本不起作用**，")
        print("     跳变率 33% 这个数会被分箱吃干净，G4≈G3。")
        print("     必须先解决分辨率问题再开跑，否则是拿 12 小时机时买一个假阴性。")


if __name__ == "__main__":
    main()
