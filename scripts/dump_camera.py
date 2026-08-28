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


def anchor_check(env) -> tuple[int, float]:
    """
    ⭐ 外部锚点。取若干已知世界坐标的 site，投影到像素并预测深度，
    与渲染深度比对。返回 (参与比对的点数, 最大误差 cm)。

    只统计**可见**的点：预测深度比渲染深度**大**很多说明该点被挡住了，
    那是遮挡不是模型错，剔掉；预测深度**小**很多才是真错（凭空穿到物体前面）。
    """
    sim = env.sim
    # 渲染分辨率下建相机（不翻转：这里直接用原始深度图，像素坐标最直接）
    cam = read_camera(env, RES, RES, flipped=False)
    out = sim.render(camera_name=CAM, width=RES, height=RES, depth=True)
    d = out[1] if isinstance(out, tuple) else out
    if d.ndim == 3:
        d = d[..., 0]
    dm = real_depth(env, d)
    # ⚠️ MuJoCo 渲染出来的图像**上下是倒的**（OpenGL 原点在左下）。
    #    robosuite 的 observation 里已经翻好，sim.render 拿到的是原始朝向。
    dm = dm[::-1]

    names = [n for n in sim.model.site_names
             if any(t in n for t in ("grip", "eef", "hand"))][:8]
    errs = []
    for n in names:
        try:
            w = torch.tensor(np.array(sim.data.get_site_xpos(n)), dtype=torch.float64)
        except Exception:
            continue
        uv, pred = cam.project(w.unsqueeze(0))
        u, v = float(uv[0, 0]), float(uv[0, 1])
        if not (1 <= u < RES - 1 and 1 <= v < RES - 1):
            continue                                     # 不在画面里
        rend = float(dm[int(round(v)), int(round(u))])
        err = float(pred[0]) - rend
        if err > 0.03:                                   # 被遮挡：预测点在渲染表面之后
            continue
        errs.append(abs(err))
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

    ref, worst_err, n_pts = None, 0.0, 0
    for i in range(n):
        env = build_env(bmark.get_task(i))
        env.seed(0)
        env.reset()
        cam = read_camera(env, IMG, IMG, flipped=True)
        if ref is None:
            ref = cam
        else:
            # 2. 跨任务一致性
            dp = float(torch.abs(cam.pos - ref.pos).max())
            dr = float(torch.abs(cam.rot - ref.rot).max())
            df = abs(cam.fovy - ref.fovy)
            if max(dp, dr, df) > 1e-6:
                raise SystemExit(
                    f"❌ task {i} 的 agentview 与 task 0 不同（Δpos={dp:.2e} "
                    f"Δrot={dr:.2e} Δfovy={df:.2e}）。\n"
                    "   相机不是常量，不能把参数写死成一份 —— "
                    "metric_coords 要改成按 task 取参数。")
        k, err = anchor_check(env)
        n_pts += k
        worst_err = max(worst_err, 0.0 if np.isnan(err) else err)
        print(f"  task {i}: 锚点 {k} 个，最大误差 {err:.2f} cm")
        env.close()

    assert ref is not None
    print(f"\n✅ 跨 {n} 个任务 agentview 完全一致（可以常量化）")
    print(f"   fovy={ref.fovy:.4f}°  focal={ref.focal():.2f} px @ {IMG}²")
    print(f"   pos={[round(float(x), 4) for x in ref.pos]}")

    # 3. 锚点判读
    print(f"\n=== 锚点验证（{n_pts} 个 site）===")
    if n_pts == 0:
        print("  ⚠️ 没有可用锚点（site 名没匹配上或全被遮挡）。**反投影未经外部验证**，"
              "\n     不要就这么去跑 G4：约定错了不会崩，只会得到一个镜像/偏移的世界。")
    elif worst_err <= args.tol_cm:
        print(f"  ✅ 最大误差 {worst_err:.2f} cm ≤ {args.tol_cm} cm，"
              "相机模型与 MuJoCo 约定一致。")
    else:
        print(f"  ❌ 最大误差 {worst_err:.2f} cm > {args.tol_cm} cm。"
              "\n     按可能性排查：y 的符号、−z 朝向、翻转、内参用了 256 而非 224。"
              "\n     **不要带着这个误差往下走**——它会整体污染 G4 的坐标。")

    ref.to_json(args.out)
    print(f"\n-> {args.out}")

    # 4. 工作空间包围盒
    files = sorted(glob.glob(f"{args.cache_dir}/*.npy"))
    if not files:
        print(f"\n（{args.cache_dir} 下无深度缓存，跳过包围盒统计）")
        return
    xyz = []
    for f in files[:20]:
        a = np.load(f).astype(np.float32)[..., 1]         # 均值通道
        d = torch.from_numpy(a.reshape(a.shape[0], -1)).double()
        xyz.append(ref.patch_xyz(d).reshape(-1, 3))
    p = torch.cat(xyz)
    lo = torch.quantile(p, 0.01, dim=0)
    hi = torch.quantile(p, 0.99, dim=0)
    print("\n=== 工作空间包围盒（p1–p99，米）===")
    for i, ax in enumerate("xyz"):
        print(f"  {ax}: [{float(lo[i]):+.3f}, {float(hi[i]):+.3f}]  "
              f"跨度 {float(hi[i]-lo[i]):.3f}")
    print("  → 填进 coord_pool.metric_extent 的 bbox（跨样本固定的常量）")


if __name__ == "__main__":
    main()
