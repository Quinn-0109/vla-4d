#!/usr/bin/env python
"""
导出 LIBERO agentview 的相机参数，并**用仿真器自己的真值验证反投影**。

`src/common/camera.py` 的 6 项自检只证明它自洽。坐标系约定与 MuJoCo 是否一致，
必须拿外部锚点验——本项目已经栽过一次"子进程知道答案却把它丢了"。

三件事，缺一不可：

1. **导出参数**（fovy / pos / rot）→ `results/tables/camera_libero.json`。
2. **跨任务一致性**：agentview 在所有任务里必须是同一台相机，否则不能常量化。
   不一致就报错——把一个会变的东西当常量，是那种跑完才发现的错。
3. ⭐ **锚点验证**：拿仿真器里已知世界坐标的 site（夹爪等），用我们的相机模型
   投影到像素并预测深度，再与**渲染出来的深度**比对。约定错在任何一处
   （y 的符号、−z、翻转、内参分辨率），这一步都会以厘米级的误差暴露。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/dump_camera.py --n_tasks 5

顺带从深度缓存统计工作空间包围盒，给 `coord_pool.metric_extent` 用。
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库路径>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero.libero import benchmark  # noqa: E402

from common.camera import IMG, Camera  # noqa: E402

CAM = "agentview"
RES = 256                     # 渲染分辨率；内参按 IMG=224 建（见 camera.py 约定 3）


def build_env(task, resolution: int = RES):
    """与 depth_diag.build_env_with_depth 同一套参数 —— 两边必须是同一个场景。"""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(**{"bddl_file_name": bddl,
                                 "camera_heights": resolution,
                                 "camera_widths": resolution,
                                 "camera_depths": True})


def read_camera(env, height: int, width: int, flipped: bool) -> Camera:
    sim = env.sim
    cid = sim.model.camera_name2id(CAM)
    return Camera(fovy=float(sim.model.cam_fovy[cid]),
                  height=height, width=width,
                  pos=torch.tensor(np.array(sim.data.cam_xpos[cid]), dtype=torch.float64),
                  rot=torch.tensor(np.array(sim.data.cam_xmat[cid]).reshape(3, 3),
                                   dtype=torch.float64),
                  flipped=flipped)


def real_depth(env, d: np.ndarray) -> np.ndarray:
    """与 depth_diag.real_depth 同一份逻辑：OpenGL 缓冲 → 米。"""
    try:
        from robosuite.utils.camera_utils import get_real_depth_map
        return get_real_depth_map(env.sim, d)
    except Exception:
        m = env.sim.model
        extent = m.stat.extent
        near, far = m.vis.map.znear * extent, m.vis.map.zfar * extent
        return near / (1.0 - d * (1.0 - near / far))


def anchor_check(env, verbose: bool = True) -> tuple[int, float]:
    """
    ⭐ 外部锚点 + **自诊断**。

    取已知世界坐标的 site，用我们的相机模型算出它该落在哪个像素、深度该是多少，
    与渲染深度比对。误差大的时候光报一个数没用，所以再做一件事：
    把整张深度图反投影成点云，**找离真值最近的那个像素**。
    预测像素与最近像素的关系直接指出错在哪一处——

        最近点在 (H−1−v, u)      → 深度图的上下翻转判断反了
        最近点在 (v, W−1−u)      → 左右
        像素对得上但深度差恒定    → z 的符号或近远平面换算

    只统计**可见**的点：预测深度比渲染深度大很多是被挡住了（遮挡不是模型错）。
    """
    sim = env.sim
    cam = read_camera(env, RES, RES, flipped=False)
    out = sim.render(camera_name=CAM, width=RES, height=RES, depth=True)
    d = out[1] if isinstance(out, tuple) else out
    if d.ndim == 3:
        d = d[..., 0]
    dm_raw = real_depth(env, d)
    # MuJoCo/OpenGL 的原点在左下，渲染出来的图上下是倒的。robosuite 的
    # observation 里已经翻好，sim.render 拿到的是原始朝向 —— 但这一条正是
    # 现在要验的，所以两种都算，让数据自己说话。
    variants = {"翻转 [::-1]": dm_raw[::-1], "不翻": dm_raw}

    names = [n for n in sim.model.site_names
             if any(t in n for t in ("grip", "eef", "hand"))][:8]
    best_name, best = None, None
    errs = []
    for n in names:
        try:
            w = torch.tensor(np.array(sim.data.get_site_xpos(n)), dtype=torch.float64)
        except Exception:
            continue
        uv, pred = cam.project(w.unsqueeze(0))
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (1 <= u < RES - 1 and 1 <= v < RES - 1):
            continue
        for vname, dm in variants.items():
            rend = float(dm[int(round(v)), int(round(u))])
            err = float(pred[0]) - rend
            if vname == "翻转 [::-1]":
                if err > 0.03:
                    continue                              # 遮挡
                errs.append(abs(err))
            if best is None or abs(err) < best[0]:
                best = (abs(err), n, vname, u, v, float(pred[0]), rend)
        best_name = n

    if verbose and best_name is not None:
        # 全图最近点：不依赖任何翻转假设
        for vname, dm in variants.items():
            grid_v, grid_u = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
            uvs = torch.tensor(np.stack([grid_u.ravel(), grid_v.ravel()], -1),
                               dtype=torch.float64)
            dd = torch.tensor(np.ascontiguousarray(dm).ravel(), dtype=torch.float64)
            pc = cam.backproject(uvs, dd)
            w = torch.tensor(np.array(sim.data.get_site_xpos(best_name)),
                             dtype=torch.float64)
            k = int(torch.argmin(torch.linalg.norm(pc - w, dim=-1)))
            uvp, _ = cam.project(w.unsqueeze(0))
            print(f"    [{vname}] site {best_name}: 预测像素 "
                  f"(u={float(uvp[0,0]):.0f}, v={float(uvp[0,1]):.0f})，"
                  f"点云最近像素 (u={k % RES}, v={k // RES})，"
                  f"距离 {float(torch.linalg.norm(pc[k]-w))*100:.1f} cm")

    if not errs:
        return 0, float("nan")
    return len(errs), max(errs) * 100.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--n_tasks", type=int, default=5)
    ap.add_argument("--cache_dir", default="results/depth_cache")
    ap.add_argument("--out", default="results/tables/camera_libero.json")
    ap.add_argument("--tol_cm", type=float, default=2.0)
    args = ap.parse_args()

    bmark = benchmark.get_benchmark_dict()[args.suite]()
    n = min(args.n_tasks, bmark.n_tasks)

    cams, worst_err, n_pts = {}, 0.0, 0
    for i in range(n):
        env = build_env(bmark.get_task(i))
        env.seed(0)
        env.reset()
        cams[i] = read_camera(env, IMG, IMG, flipped=True)
        k, err = anchor_check(env, verbose=(i == 0))
        n_pts += k
        worst_err = max(worst_err, 0.0 if np.isnan(err) else err)
        print(f"  task {i}: 锚点 {k} 个，最大误差 {err:.2f} cm")
        env.close()

    # ⚠️ 实测：agentview **不是**跨任务常量。libero_10 的十个任务跨厨房/客厅/
    #    书房多个场景，每个 BDDL 自己摆相机，实测 Δpos 达 0.65 m。
    #    所以按 task 存一份，metric_coords 按 task_id 取。
    #    反投影到**世界系**之后坐标仍跨任务可比 —— 这正是要反投影的理由之一。
    ref = cams[0]
    dp = max(float(torch.abs(c.pos - ref.pos).max()) for c in cams.values())
    dr = max(float(torch.abs(c.rot - ref.rot).max()) for c in cams.values())
    print(f"\n跨 {n} 个任务的 agentview 差异：Δpos 最大 {dp:.3f} m，Δrot 最大 {dr:.3f}")
    print("  → 相机**按 task 存**（世界系坐标仍可比）" if dp > 1e-6
          else "  → 相机跨任务一致，可以常量化")

    print(f"   fovy={ref.fovy:.4f}°  focal={ref.focal():.2f} px @ {IMG}²")

    # 3. 锚点判读
    print(f"\n=== 锚点验证（{n_pts} 个 site）===")
    if n_pts == 0:
        print("  ⚠️ 没有可用锚点。**反投影未经外部验证**，不要就这么去跑 G4。")
    elif worst_err <= args.tol_cm:
        print(f"  ✅ 最大误差 {worst_err:.2f} cm ≤ {args.tol_cm} cm，"
              "相机模型与 MuJoCo 约定一致。")
    else:
        print(f"  ❌ 最大误差 {worst_err:.2f} cm > {args.tol_cm} cm。"
              "\n     看上面 task 0 那两行诊断：预测像素与点云最近像素的关系"
              "\n     直接指出错在哪一处（上下翻转 / 左右 / z 的符号）。"
              "\n     **不要带着这个误差往下走**——它会整体污染 G4 的坐标。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {str(i): {"fovy": c.fovy, "height": c.height, "width": c.width,
                  "pos": c.pos.tolist(), "rot": c.rot.reshape(-1).tolist(),
                  "flipped": c.flipped} for i, c in cams.items()}, indent=2))
    print(f"\n-> {out}（按 task_id 索引）")

    # 4. 工作空间包围盒
    if n_pts == 0 or worst_err > args.tol_cm:
        print("\n（锚点未通过，跳过包围盒统计 —— 用一个错的相机算出来的包围盒"
              "\n  会让人以为这一步做完了，比不算更糟）")
        return
    files = sorted(glob.glob(f"{args.cache_dir}/*.npy"))
    if not files:
        print(f"\n（{args.cache_dir} 下无深度缓存，跳过包围盒统计）")
        return

    # ⚠️ 每条 episode 要用**它自己那个 task 的相机**。缓存文件名形如
    #    {suite}_t{task_id}_e{ep}.npy（depth_diag.py 写的）。
    #    全用 task 0 的相机算，就是刚刚才发现的那个错的翻版。
    xyz, skipped = [], 0
    for f in files:
        m = re.search(r"_t(\d+)_e\d+\.npy$", Path(f).name)
        if m is None or int(m.group(1)) not in cams:
            skipped += 1
            continue
        a = np.load(f).astype(np.float32)[..., 1]         # 均值通道
        d = torch.from_numpy(a.reshape(a.shape[0], -1)).double()
        xyz.append(cams[int(m.group(1))].patch_xyz(d).reshape(-1, 3))
    if not xyz:
        print(f"\n（{len(files)} 个缓存文件都没有对应的相机参数 —— "
              f"用 --n_tasks {bmark.n_tasks} 把全部任务都导出来）")
        return
    if skipped:
        print(f"\n（跳过 {skipped} 条：其 task 未在本次 --n_tasks 范围内）")

    p = torch.cat(xyz)
    lo = torch.quantile(p, 0.01, dim=0)
    hi = torch.quantile(p, 0.99, dim=0)
    print("\n=== 工作空间包围盒（p1–p99，米，%d 条 episode）===" % len(xyz))
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: [{float(lo[i]):+.3f}, {float(hi[i]):+.3f}]  "
              f"跨度 {float(hi[i]-lo[i]):.3f}")
    print("  → 填进 coord_pool.metric_extent 的 bbox（跨样本固定的常量）")


if __name__ == "__main__":
    main()
