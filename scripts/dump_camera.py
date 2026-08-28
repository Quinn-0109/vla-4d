#!/usr/bin/env python
"""
导出 LIBERO 各任务的相机参数，并**用仿真器自己的真值验证反投影**。

`src/common/camera.py` 的 6 项自检只证明它自洽。坐标系约定与 MuJoCo 是否一致，
必须拿外部证据验。

三件事：

1. **按 task 导出参数**（fovy / pos / rot）→ `results/tables/camera_libero.json`。
   ⚠️ agentview **不是**跨任务常量：libero_10 的十个任务跨多个场景，
   每个 BDDL 自己摆相机，实测 Δpos 达 0.65 m。反投影到世界系之后
   坐标仍跨任务可比——这正是要反投影的理由之一。
2. ⭐ **主判据：两台相机的点云必须在世界系里重合**（`consistency_check`）。
   不依赖挑点，翻转判断错了两片云必然对不上。
3. 辅助判据：site 到点云的最近点距离（`site_check`）。

**这个脚本的判据本身错过两次，都记在对应函数的 docstring 里**：
拿"预测像素处的深度差"当误差（自由空间的 site 会报 60+ cm 假误差）、
以及渲染 B 相机的深度却用 A 相机的位姿反投影（两片云差 70 cm）。
**判据错比结果错更难发现，因为它看起来完全正常。**

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/dump_camera.py --n_tasks 5

顺带从深度缓存统计工作空间包围盒（**锚点没过就不算**），
给 `coord_pool.metric_extent` 用。
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


def read_camera(env, height: int, width: int, flipped: bool,
                name: str = CAM) -> Camera:
    """
    ⚠️ `name` 是必须的。初版把 `CAM` 写死在函数体里，于是两相机一致性检验
    **拿 frontview 的深度配 agentview 的位姿反投影**，两片点云差 70 cm——
    看起来像"相机模型整个错了"，实际是测试代码自己的 bug。
    两种翻转都报 70.1 cm、只差 0.2%，正是"与翻转无关的恒定位姿错配"的指纹。
    """
    sim = env.sim
    cid = sim.model.camera_name2id(name)
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


ALT_CAMS = ("frontview", "birdview", "sideview", "robot0_eye_in_hand")


def _cloud(env, cam_name: str, flip: bool, n: int = 4000) -> torch.Tensor | None:
    """把一台相机的深度图反投影成世界系点云，(n, 3)。取不到就返回 None。"""
    sim = env.sim
    try:
        sim.model.camera_name2id(cam_name)
        out = sim.render(camera_name=cam_name, width=RES, height=RES, depth=True)
    except Exception:
        return None
    d = out[1] if isinstance(out, tuple) else out
    if d.ndim == 3:
        d = d[..., 0]
    dm = real_depth(env, d)
    if flip:
        dm = dm[::-1]
    cam = read_camera(env, RES, RES, flipped=False, name=cam_name)  # 用**这台**相机
    vv, uu = np.meshgrid(np.arange(RES), np.arange(RES), indexing="ij")
    uv = torch.tensor(np.stack([uu.ravel(), vv.ravel()], -1), dtype=torch.float64)
    dd = torch.tensor(np.ascontiguousarray(dm).ravel(), dtype=torch.float64)
    keep = dd < torch.quantile(dd, 0.90)              # 剔掉背景/远平面
    uv, dd = uv[keep], dd[keep]
    if uv.shape[0] < n:
        return None
    idx = torch.randperm(uv.shape[0])[:n]
    return cam.backproject(uv[idx], dd[idx])


def _nn_p10(b: torch.Tensor, a: torch.Tensor) -> tuple[float, float]:
    """b 中每点到 a 的最近邻距离的 (p10, 中位数)，单位 cm。分块算，免得占几 GB。"""
    mins = []
    for i in range(0, b.shape[0], 2000):
        mins.append(torch.cdist(b[i:i + 2000], a).min(dim=1).values)
    d = torch.cat(mins)
    return float(torch.quantile(d, 0.10)) * 100.0, float(d.median()) * 100.0


def consistency_check(env, n: int = 4000) -> dict:
    """
    ⭐ **主判据：两台相机的点云必须在世界系里重合。**

    比"拿 site 当锚点"强，因为它不依赖挑点——site 挑错了会把自由空间里的点
    （比如两指之间的抓取点，那里根本没有几何体）当成表面，射线穿过去打到桌面，
    误差凭空多出半米。第一版就栽在这里。

    翻转判断若是错的，两台相机各自被扭曲的方式**取决于各自的位姿**，不是一个
    共同的世界变换，所以两片点云对不上。

    返回 {翻转与否: (共视部分 p10, 中位数)}，单位 cm。判据取 **p10**：
    中位数会被视角重叠不足污染（B 看得到而 A 看不到的表面本来就配不上），
    低分位才是"确实共视的那部分对得有多准"。

    ⚠️ 另带一个**密度检验**：p10 有个采样下限——两次独立采样之间的最近邻间距
    本身就是厘米量级，分不清"准到毫米"和"差一厘米的系统偏移"。
    点数翻四倍后 p10 应按 n^(-1/3) ≈ 0.63× 缩小；**几乎不动就说明是系统偏移**。
    """
    alt = next((c for c in ALT_CAMS if _try_cam(env, c)), None)
    if alt is None:
        return {}
    res = {}
    for flip in (True, False):
        a, b = _cloud(env, CAM, flip, n), _cloud(env, alt, flip, n)
        if a is None or b is None:
            continue
        res[flip] = _nn_p10(b, a)
    if res:
        best = min(res, key=lambda f: res[f][0])
        a4, b4 = _cloud(env, CAM, best, n * 4), _cloud(env, alt, best, n * 4)
        if a4 is not None and b4 is not None:
            res["dense"] = (_nn_p10(b4, a4)[0], res[best][0], n, n * 4)
    return {"alt": alt, **res}


def _try_cam(env, name: str) -> bool:
    try:
        env.sim.model.camera_name2id(name)
        return True
    except Exception:
        return False


def site_check(env, flip: bool) -> list[tuple[str, float]]:
    """
    辅助判据：已知世界坐标的 site 到点云的**最近点距离**。

    ⚠️ 用最近点距离，不用"预测像素处的深度差"。后者在 site 落在自由空间时
    （`gripper0_grip_site` 就在两指之间）会把身后桌面的深度当成它的深度，
    报出 60+ cm 的假误差 —— 第一版的 64.67 cm 就是这么来的，
    **不是相机模型错，是判据错**。
    """
    sim = env.sim
    cloud = _cloud(env, CAM, flip, n=20000)
    if cloud is None:
        return []
    out = []
    for n in [x for x in sim.model.site_names
              if any(t in x for t in ("grip", "eef", "hand"))][:8]:
        try:
            w = torch.tensor(np.array(sim.data.get_site_xpos(n)), dtype=torch.float64)
        except Exception:
            continue
        out.append((n, float(torch.linalg.norm(cloud - w, dim=-1).min()) * 100.0))
    return out


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

    cams, cons, sites = {}, {}, []
    for i in range(n):
        env = build_env(bmark.get_task(i))
        env.seed(0)
        env.reset()
        cams[i] = read_camera(env, IMG, IMG, flipped=True)
        if i == 0:
            cons = consistency_check(env)
            if cons:
                print(f"\n=== 主判据：agentview 与 {cons['alt']} 的点云重合度 ===")
                for flip in (True, False):
                    if flip in cons:
                        p10, med = cons[flip]
                        print(f"  深度图{'翻转 [::-1]' if flip else '不翻     '} → "
                              f"共视部分(p10) {p10:.2f} cm，中位 {med:.2f} cm")
                best_flip = min((f for f in (True, False) if f in cons),
                                key=lambda f: cons[f][0])
                sites = site_check(env, best_flip)
                if sites:
                    print(f"\n  辅助：site 到点云的最近点距离（翻转={best_flip}）")
                    for nm, e in sites:
                        print(f"    {nm:<34} {e:>6.2f} cm")
        env.close()

    # ⚠️ 实测：agentview **不是**跨任务常量。libero_10 的十个任务跨厨房/客厅/
    #    书房多个场景，每个 BDDL 自己摆相机，Δpos 达 0.65 m。
    #    所以按 task 存一份，metric_coords 按 task_id 取。
    #    反投影到**世界系**之后坐标仍跨任务可比 —— 这正是要反投影的理由之一。
    ref = cams[0]
    dp = max(float(torch.abs(c.pos - ref.pos).max()) for c in cams.values())
    dr = max(float(torch.abs(c.rot - ref.rot).max()) for c in cams.values())
    print(f"\n跨 {n} 个任务的 agentview 差异：Δpos 最大 {dp:.3f} m，Δrot 最大 {dr:.3f}")
    print("  → 相机**按 task 存**（世界系坐标仍可比）" if dp > 1e-6
          else "  → 相机跨任务一致，可以常量化")

    print(f"   fovy={ref.fovy:.4f}°  focal={ref.focal():.2f} px @ {IMG}²")

    # 3. 判读
    print("\n=== 判读 ===")
    if not cons or True not in cons or False not in cons:
        print("  ⚠️ 拿不到第二台相机，主判据没跑成。**反投影未经外部验证**，"
              "\n     不要就这么去跑 G4：约定错了不会崩，只会得到一个镜像或偏移的世界。")
        ok = False
    else:
        best_flip = min((True, False), key=lambda f: cons[f][0])
        worst = cons[best_flip][0]                       # 判据取共视部分(p10)
        ratio = cons[not best_flip][0] / max(worst, 1e-6)
        print(f"  最优翻转设置：flipped={best_flip}（{worst:.2f} cm，"
              f"另一种是它的 {ratio:.1f} 倍）")
        ok = worst <= args.tol_cm and best_flip is True
        if worst > args.tol_cm:
            print(f"  ❌ {worst:.2f} cm > {args.tol_cm} cm —— 两台相机的点云对不上，"
                  "\n     不是翻转的问题（两种都试过了）。查 y 的符号、−z 朝向、"
                  "\n     内参是否用了渲染分辨率而非 224。")
        elif best_flip is not True:
            print("  ❌ 最优是**不翻**，而 camera.py 里 flipped=True 是按"
                  "`depth_diag` 的 [::-1] 定的 —— 两处对不上，先统一再往下走。")
        else:
            print(f"  ✅ {worst:.2f} cm ≤ {args.tol_cm} cm 且翻转约定与 "
                  "depth_diag 一致，相机模型可用。")
        if ratio < 3:
            print(f"  ⚠️ 两种翻转只差 {ratio:.1f} 倍，判别力不足 —— "
                  "这个场景可能过于对称，换个 task 复核。")
        if "dense" in cons:
            p_dense, p_sparse, n0, n1 = cons["dense"]
            shrink = p_dense / max(p_sparse, 1e-9)
            print(f"\n  密度检验：点数 {n0} → {n1}，p10 从 {p_sparse:.2f} 降到 "
                  f"{p_dense:.2f} cm（{shrink:.2f}×，理论 0.63×）")
            if shrink < 0.80:
                print("    ✅ 随密度缩小 —— p10 是**采样下限**，不是系统偏移。"
                      "反投影的真实精度好于这个数。")
            else:
                print(f"    ⚠️ 几乎不动 —— 存在约 {p_dense:.1f} cm 的**系统偏移**。"
                      "\n       量级上像 patch 中心取整或内外参分辨率；"
                      "\n       小于体素箱宽(~30 cm)时不致命，但要记进文档。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {str(i): {"fovy": c.fovy, "height": c.height, "width": c.width,
                  "pos": c.pos.tolist(), "rot": c.rot.reshape(-1).tolist(),
                  "flipped": c.flipped} for i, c in cams.items()}, indent=2))
    print(f"\n-> {out}（按 task_id 索引）")

    # 4. 工作空间包围盒
    if not ok:
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
