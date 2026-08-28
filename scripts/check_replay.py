#!/usr/bin/env python
"""
回放保真度验证 —— **G4 / M2 的唯一前置**（`docs/06` §4.3）。

G4 要的是**训练集**每一帧的 patch 级深度。RLDS 里没有深度，只能把动作序列
放回仿真器重放、取仿真器的深度。这一步成立的前提是：

    **同一初始状态 + 同一动作序列 → 仿真器复现出 RLDS 里的那张画面。**

对不上就说明回放发散了，缓存出来的深度属于另一个场景 ——
**而这不会报任何错**：深度是合法的浮点数，反投影出的坐标也合法，
G4 照跑，只是它看的是一个和图像对不上的三维世界。

⚠️ 这与 `depth_diag.py` 是两件事。那里回放的是**我们自己录的评测轨迹**
（`action_env` 是我们写下的，重放当然一致），测场景统计足够；
这里要验的是**别人生成的训练数据**能不能被复现，难得多。

四步：

  1. 直接读原始 tfds（**不走 openvla 的 pipeline**）—— 那条链路会按
     BOUNDS_Q99 归一化动作，喂回仿真器就不是原来的动作了
  2. 用 language_instruction 反查 task_id（LIBERO 里逐任务唯一）
  3. 逐个试该任务的初始状态，找出与 RLDS 第 0 帧最像的那个
     （顺带把"RLDS 图像是否已 180° 翻转"一并定下来，同 depth_diag 的做法）
  4. 重放动作，在若干时刻比对画面，看误差**随时间怎么走**

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/check_replay.py --suite libero_10 --n_episodes 5
"""

from __future__ import annotations

import argparse
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

RES = 256
CAM = "agentview"


def build_env(task, resolution: int = RES):
    """与 depth_diag / dump_camera 同一套参数 —— 三处必须是同一个场景。"""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    return OffScreenRenderEnv(**{"bddl_file_name": bddl,
                                 "camera_heights": resolution,
                                 "camera_widths": resolution,
                                 "camera_depths": True})


def rgb_of(obs, flip: bool) -> np.ndarray:
    img = np.asarray(obs[f"{CAM}_image"])
    return img[::-1, ::-1] if flip else img


def diff(a: np.ndarray, b: np.ndarray) -> float:
    """平均绝对像素差（0–255）。0 = 逐位相同，>10 = 肉眼可见的不同。"""
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def load_episodes(data_root: str, name: str, n: int):
    """
    直接读原始 tfds。⚠️ **不能走 openvla 的 RLDSDataset**：
    那条链路按 BOUNDS_Q99 归一化了动作，喂回仿真器就不是原来的动作，
    回放必然发散 —— 而那是我们的调用方式错，不是数据的问题。
    """
    import tensorflow_datasets as tfds

    b = tfds.builder(name, data_dir=data_root)
    ds = b.as_dataset(split="train", shuffle_files=False)
    out = []
    for ep in ds.take(n):
        steps = list(ep["steps"].as_numpy_iterator())
        lang = steps[0]["language_instruction"].decode().strip().lower()
        imgs = np.stack([s["observation"]["image"] for s in steps])
        acts = np.stack([s["action"] for s in steps]).astype(np.float64)
        out.append({"lang": lang, "images": imgs, "actions": acts})
    return out


def find_task(bmark, lang: str) -> int | None:
    """language_instruction → task_id。LIBERO 里任务描述逐任务唯一。"""
    for i in range(bmark.n_tasks):
        if bmark.get_task(i).language.strip().lower() == lang:
            return i
    return None


def match_init(env, bmark, tid: int, ref: np.ndarray, n_try: int):
    """
    逐个试初始状态，找与 RLDS 第 0 帧最像的那个；同时定下翻转与否。
    返回 (init_idx, flip, 最好误差, 次好误差, 不翻的最好误差)。

    ⚠️ **必须同时报次好**。只报最好分不清两种情况：
      · best 10.2 / second 25   → 确实找到了那一个初始状态，10.2 是渲染差异
      · best 10.2 / second 10.4 → 所有候选都差不多，初始状态根本不是区分因素，
                                  问题在渲染设置（而"最像的那个"只是噪声）
    """
    states = bmark.get_task_init_states(tid)
    scores = []
    for j in range(min(n_try, len(states))):
        env.reset()
        obs = env.set_init_state(states[j])
        scores.append((diff(rgb_of(obs, True), ref),
                       diff(rgb_of(obs, False), ref), j))
    flipped = sorted(s[0] for s in scores)
    best_row = min(scores, key=lambda x: x[0])
    return (best_row[2], True, flipped[0], flipped[1],
            min(s[1] for s in scores))


def diff_profile(a: np.ndarray, b: np.ndarray) -> dict:
    """
    误差长什么样 —— **这决定了病因**。

    · 全图均匀的小差 → 编码/渲染设置不同（JPEG、光照、纹理），初始状态其实对了
    · 集中在少数像素 → 物体位置不同，初始状态没对上
    """
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean(axis=-1).ravel()
    d_sorted = np.sort(d)[::-1]
    top10 = d_sorted[: max(1, len(d) // 10)].sum() / max(d.sum(), 1e-9)
    return {"mean": float(d.mean()), "p50": float(np.median(d)),
            "p95": float(np.percentile(d, 95)), "max": float(d.max()),
            "top10_share": float(top10)}


def _dump(a: np.ndarray, b: np.ndarray, e_i: int, tid: int) -> None:
    """并排存图。数字说不清的时候，看一眼最快。"""
    from PIL import Image
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"replay_t{tid}_e{e_i}.png"
    Image.fromarray(np.concatenate([a, b, d], axis=1)).save(p)
    print(f"       -> {p}（左：回放  中：RLDS  右：差）")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--data_root", default=str(
        Path("~/autodl-tmp/datasets/modified_libero_rlds").expanduser()))
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_init_try", type=int, default=50)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--dump", action="store_true",
                    help="把回放帧与 RLDS 帧并排存成 PNG，肉眼比对")
    ap.add_argument("--tol", type=float, default=5.0,
                    help="平均绝对像素差的容忍上限（0–255）")
    args = ap.parse_args()

    name = f"{args.suite}_no_noops"
    print(f"读 {name} @ {args.data_root}")
    eps = load_episodes(args.data_root, name, args.n_episodes)
    bmark = benchmark.get_benchmark_dict()[args.suite]()

    worst_init, worst_end, n_ok = 0.0, 0.0, 0
    for e_i, ep in enumerate(eps):
        tid = find_task(bmark, ep["lang"])
        if tid is None:
            print(f"  [{e_i}] ❌ 任务描述对不上任何 task：{ep['lang'][:60]!r}")
            continue
        env = build_env(bmark.get_task(tid))
        env.seed(0)
        j, flip, d0, d1, d_noflip = match_init(
            env, bmark, tid, ep["images"][0], args.n_init_try)
        print(f"\n  [{e_i}] task {tid}  RLDS 图像 {ep['images'].shape[1:]}  "
              f"init {j}  第0帧误差 {d0:.2f}")
        print(f"       次好的候选 {d1:.2f}（差 {d1 - d0:+.2f}）   "
              f"不翻转的最好 {d_noflip:.2f}")

        env.reset()
        obs = env.set_init_state(bmark.get_task_init_states(tid)[j])
        prof = diff_profile(rgb_of(obs, flip), ep["images"][0])
        print(f"       误差分布 p50={prof['p50']:.1f} p95={prof['p95']:.1f} "
              f"max={prof['max']:.0f}  最大 10% 的像素占总误差 "
              f"{prof['top10_share']:.0%}")
        if args.dump:
            _dump(rgb_of(obs, flip), ep["images"][0], e_i, tid)

        if d0 > args.tol:
            if d1 - d0 < 1.0:
                print(f"       ❌ 所有初始状态都差不多（次好只差 {d1 - d0:.2f}）——"
                      "\n          初始状态不是区分因素，问题在**渲染设置**。")
            elif prof["top10_share"] > 0.6:
                print("       ❌ 误差集中在少数像素——**物体位置不同**，初始状态没对上。")
            else:
                print("       ❌ 误差全图均匀——**渲染/编码设置不同**"
                      "（JPEG、光照、纹理、fovy）。")
            env.close()
            worst_init = max(worst_init, d0)
            continue

        # 重放：RLDS 的动作直接喂回去
        env.reset()
        obs = env.set_init_state(bmark.get_task_init_states(tid)[j])
        T = min(len(ep["actions"]), ep["images"].shape[0])
        curve = []
        for t in range(T):
            d = diff(rgb_of(obs, flip), ep["images"][t])
            if t in (0, T // 4, T // 2, 3 * T // 4, T - 1):
                curve.append((t, d))
            obs, *_ = env.step(ep["actions"][t].tolist())
        env.close()
        print("       " + "  ".join(f"t={t}:{d:.1f}" for t, d in curve))
        end = curve[-1][1]
        worst_init, worst_end = max(worst_init, d0), max(worst_end, end)
        n_ok += end <= args.tol

    print(f"\n{'=' * 60}\n=== 判读 ===")
    print(f"  {n_ok}/{len(eps)} 条 episode 全程误差 ≤ {args.tol}")
    print(f"  最差的第 0 帧误差 {worst_init:.2f}，最差的末帧误差 {worst_end:.2f}")
    if n_ok == len(eps) and worst_end <= args.tol:
        print("  ✅ 回放能复现训练数据，可以给训练集缓存深度，G4/M2 解锁。")
    elif worst_init > args.tol:
        print("  ❌ **第 0 帧就对不上**，问题在初始状态或渲染设置，不是发散。"
              "\n     先查：RLDS 的图像分辨率、相机名、是否已翻转。")
    else:
        print("  ❌ **随时间发散**：初始状态对得上，重放却越走越远。"
              "\n     多半是控制器设置不同（RLDS 生成时的 controller config）。"
              "\n     此时不能给训练集缓存深度 —— 深度会属于另一个场景，"
              "\n     而 G4 照跑不报错。回退方案见 docs/06 §6（单目深度）。")


if __name__ == "__main__":
    main()
