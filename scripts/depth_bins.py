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
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
    ap.add_argument("--camera", default="results/tables/camera_libero.json",
                    help="dump_camera.py 的产物；给了就跑世界系那一节")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--win_per_ep", type=int, default=5,
                    help="每条 episode 抽几个窗口进池化算子")
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
        print("     跳变率会被分箱吃干净，G4≈G3。")
        print("     必须先解决分辨率问题再开跑，否则是拿 12 小时机时买一个假阴性。")

    world_section(files, args)


def world_section(files, args) -> None:
    """
    ⭐ **不用代理，直接跑真正的池化算子。**

    上面那一节量的是「深度跨度 vs 一维箱宽」，是个**代理**：真正决定 G4 能不能
    赢的是「同一 (h,w) 的 K 个 patch 会不会落进**不同的输出槽**」。三个空间轴
    一起分箱时，30 cm 的三维位移可能被摊到三个轴上、哪个轴都不够跨箱——
    代理看着够，实际分不开。

    顺带把 `coord_pool` 自检里那两个「合成深度下待定、真值深度必须重测」的数
    （槽位利用率、跨帧占比）在真数据上量出来。
    """
    import torch

    from common.camera import Camera
    from pooling.coord_pool import (GRID, N_T_DEFAULT, coord_bin_pool,
                                    grid_coords, grid_extent, metric_coords,
                                    metric_extent)

    if not os.path.exists(args.camera):
        print(f"\n（{args.camera} 不存在，跳过世界系一节；先跑 scripts/dump_camera.py）")
        return
    cams = {int(k): Camera(fovy=v["fovy"], height=v["height"], width=v["width"],
                           pos=torch.tensor(v["pos"], dtype=torch.float64),
                           rot=torch.tensor(v["rot"], dtype=torch.float64).reshape(3, 3),
                           flipped=v.get("flipped", True))
            for k, v in json.loads(Path(args.camera).read_text()).items()}

    K, s = args.K, args.stride
    span = (K - 1) * s
    rng = np.random.default_rng(0)

    # 先扫一遍定包围盒（p1–p99），再跑池化 —— 两遍都用同一批窗口
    wins = []          # (task_id, (K, 256) 深度)
    for f in files:
        m = re.search(r"_t(\d+)_e\d+\.npy$", Path(f).name)
        if m is None or int(m.group(1)) not in cams:
            continue
        a = np.load(f).astype(np.float32)[..., 1]          # 均值通道
        d = a.reshape(a.shape[0], -1)
        if d.shape[0] <= span:
            continue
        ts = rng.choice(np.arange(span, d.shape[0]),
                        min(args.win_per_ep, d.shape[0] - span), replace=False)
        for t in ts:
            wins.append((int(m.group(1)), d[t - span:t + 1:s]))
    if not wins:
        print("\n（没有窗口能对上相机参数，跳过世界系一节）")
        return

    pts = torch.cat([cams[tid].patch_xyz(torch.from_numpy(w).double()).reshape(-1, 3)
                     for tid, w in wins])
    lo3 = torch.quantile(pts, 0.01, dim=0)
    hi3 = torch.quantile(pts, 0.99, dim=0)
    bbox = torch.stack([lo3, hi3])
    print(f"\n{'='*64}\n=== 世界系：真正的池化算子（{len(wins)} 个窗口）===")
    print("包围盒 p1–p99 (m): " + "  ".join(
        f"{ax}[{float(lo3[i]):+.2f},{float(hi3[i]):+.2f}]" for i, ax in enumerate("xyz")))

    glo, ghi = grid_extent(K)
    mlo, mhi = metric_extent(K, bbox)
    gc = grid_coords(K).unsqueeze(0)

    def n_slots(o):
        """每个 (h,w) 格子的 K 个 patch 占了几个不同的输出槽。"""
        a = o.assign[0].reshape(K, GRID * GRID)
        return torch.tensor([len(set(a[:, c].tolist()) - {-1})
                             for c in range(GRID * GRID)])

    extra, use_g3, use_g4 = [], [], []
    for tid, w in wins[:200]:
        dep = torch.from_numpy(w).double().unsqueeze(0)                   # (1,K,256)
        feat = torch.zeros(1, K * GRID * GRID, 8)                         # 分箱不看特征
        mc = metric_coords(dep, K, cams[tid]).float()
        g3 = coord_bin_pool(feat, gc, args.budget, glo, ghi, n_t=N_T_DEFAULT)
        g4 = coord_bin_pool(feat, mc, args.budget, mlo, mhi, n_t=N_T_DEFAULT)
        # ⚠️ **必须配对比较。** 直接看"落进多于一个槽的比例"量到的是**时间轴**：
        #    n_t=2 时同一 (h,w) 光靠时间就必然进 ≥2 个槽，G3 也报 100%。
        #    G3 的空间箱由 (h,w) 决定、与内容无关，所以 n_slot_G3 就是"只有时间
        #    能分开多少"。两者之差才是**度量坐标额外分开的部分**。
        extra.append(float((n_slots(g4) > n_slots(g3)).float().mean()))
        use_g3.append(int(g3.n_used[0]))
        use_g4.append(int(g4.n_used[0]))
    print(f"\n同一 (h,w) 的 K 帧 patch，**G4 比 G3 分得更开**的格子占比")
    print(f"  {np.mean(extra):>6.1%}   ← 度量坐标相对图像网格的机制余量"
          f"（已扣掉时间轴，G3 作配对参照）")
    print(f"\n槽位利用率（预算 {args.budget}）  G3 {np.mean(use_g3):.0f}   "
          f"G4 {np.mean(use_g4):.0f}")
    d = float(np.mean(extra))
    print(f"\n=== 判读 ===")
    if d >= 0.10:
        print(f"  ✅ G4 比 G3 多分开 {d:.1%} 的格子，机制余量充足。")
    elif d >= 0.03:
        print(f"  🔶 只多分开 {d:.1%}。机制在，但余量薄 —— 分层分析必做，"
              "\n     且要认真考虑收紧 z 包围盒（现在多半被背景撑着）。")
    else:
        print(f"  ❌ 只多分开 {d:.1%}。**当前包围盒下 G4 的度量分箱几乎没有额外作用**，"
              "\n     先收紧包围盒或调轴向箱数，再决定要不要开跑。")
    if np.mean(use_g4) < 0.85 * args.budget:
        print(f"  ⚠️ G4 只用掉 {np.mean(use_g4):.0f}/{args.budget} 个槽 —— "
              "enforce_n 拉平后三组的公共预算都会被它拖低。")


if __name__ == "__main__":
    main()
