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


def demo_init_states(task, n: int = 60):
    """
    从 LIBERO 的**演示** HDF5 里取每条 demo 的初始状态。

    ⚠️ 这是与 `get_task_init_states()` **不同的一批**：后者给的是**评测用**的
    50 个初始摆放，而 modified_libero_rlds 是从**演示数据**转换来的。
    两者不是同一批 —— 拿评测初始状态去对演示画面，50 个候选里一个对的都没有，
    "最像的那个"只是噪声，误差就停在"不同摆放"的量级上（实测 10.2）。
    """
    try:
        import h5py

        from libero.libero import get_libero_path
        root = Path(get_libero_path("datasets"))
    except Exception:
        return []
    hits = list(root.rglob(f"{task.name}_demo.hdf5")) or list(
        root.rglob(f"{task.name}*.hdf5"))
    if not hits:
        return []
    out = []
    with h5py.File(hits[0], "r") as f:
        for k in list(f["data"].keys())[:n]:
            st = f["data"][k]["states"]
            out.append(np.asarray(st[0]))
    return out


def match_init(env, bmark, tid: int, ref: np.ndarray, n_try: int):
    """
    逐个试初始状态，找与 RLDS 第 0 帧最像的那个；同时定下翻转与否。
    返回 (init_idx, flip, 最好误差, 次好误差, 不翻的最好误差)。

    ⚠️ **必须同时报次好**。只报最好分不清两种情况：
      · best 10.2 / second 25   → 确实找到了那一个初始状态，10.2 是渲染差异
      · best 10.2 / second 10.4 → 所有候选都差不多，初始状态根本不是区分因素，
                                  问题在渲染设置（而"最像的那个"只是噪声）
    """
    evals = list(bmark.get_task_init_states(tid))[:n_try]
    demos = demo_init_states(bmark.get_task(tid), n_try)
    cands = [("eval", j, st) for j, st in enumerate(evals)] + \
            [("demo", j, st) for j, st in enumerate(demos)]
    scores = []
    for src, j, st in cands:
        env.reset()
        obs = env.set_init_state(st)
        scores.append((diff(rgb_of(obs, True), ref),
                       diff(rgb_of(obs, False), ref), src, j, st))
    flipped = sorted(x[0] for x in scores)
    best = min(scores, key=lambda x: x[0])
    return {"src": best[2], "idx": best[3], "state": best[4],
            "best": flipped[0], "second": flipped[1],
            "noflip": min(x[1] for x in scores),
            "n_eval": len(evals), "n_demo": len(demos)}


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

    worst_init, worst_end, n_ok = 0.0, None, 0
    for e_i, ep in enumerate(eps):
        tid = find_task(bmark, ep["lang"])
        if tid is None:
            print(f"  [{e_i}] ❌ 任务描述对不上任何 task：{ep['lang'][:60]!r}")
            continue
        env = build_env(bmark.get_task(tid))
        env.seed(0)
        m = match_init(env, bmark, tid, ep["images"][0], args.n_init_try)
        j, flip, d0, d1, d_noflip = m["idx"], True, m["best"], m["second"], m["noflip"]
        print(f"\n  [{e_i}] task {tid}  RLDS 图像 {ep['images'].shape[1:]}  "
              f"候选 {m['n_eval']} eval + {m['n_demo']} demo")
        print(f"       最像的是 **{m['src']}** 的第 {j} 个  第0帧误差 {d0:.2f}")
        print(f"       次好的候选 {d1:.2f}（差 {d1 - d0:+.2f}）   "
              f"不翻转的最好 {d_noflip:.2f}")
        if m["n_demo"] == 0:
            print("       ⚠️ 没找到演示 HDF5 —— 候选里只有**评测**初始状态，"
                  "\n          而 RLDS 是从**演示**转换来的，两者不是同一批。"
                  "\n          下载 LIBERO 演示数据后重跑，这一条才有意义。")

        env.reset()
        obs = env.set_init_state(m["state"])
        prof = diff_profile(rgb_of(obs, flip), ep["images"][0])
        print(f"       误差分布 p50={prof['p50']:.1f} p95={prof['p95']:.1f} "
              f"max={prof['max']:.0f}  最大 10% 的像素占总误差 "
              f"{prof['top10_share']:.0%}")
        if args.dump:
            _dump(rgb_of(obs, flip), ep["images"][0], e_i, tid)

        if d0 > args.tol:
            # ⚠️ **先看误差剖面，再看候选间的差距。** 初版反过来：先判「次好只差
            #    <1 → 渲染设置」，根本没看剖面。结果三条 episode 剖面几乎相同
            #    （p50 全是 1.3、集中度 72–79%），却因为次好差了 1.42 还是 0.02
            #    而给出两种诊断 —— 同一个病因两种结论，说明顺序错了。
            #
            #    剖面才是判病因的：整张图偏（p50 大）= 渲染/编码设置；
            #    只有几小块偏（p50 小、集中度高）= 物体位置不同。
            #    候选间的差距只回答另一个问题：**正确的那个在不在候选里**。
            localized = prof["top10_share"] > 0.6 and prof["p50"] < 3.0
            if localized:
                print(f"       ❌ 画面 95% 相同（p50={prof['p50']:.1f}），"
                      f"误差集中在 {1 - prof['top10_share']:.0%} 以外的少数像素 ——"
                      "\n          **是几个物体摆在不同位置**，不是渲染设置。")
                if d1 - d0 < 1.5:
                    print(f"          且所有候选都差不多（次好只差 {d1 - d0:.2f}）——"
                          "\n          **正确的初始摆放不在候选里**。"
                          + ("\n          候选里没有 demo 初始状态：RLDS 由演示数据转换而来，"
                             "\n          而 get_task_init_states() 给的是评测用的另一批。"
                             "\n          → 下载 LIBERO 演示数据后重跑。"
                             if m["n_demo"] == 0 else
                             "\n          demo 候选也试过了仍对不上 —— 需要另找初始状态来源。"))
            elif prof["p50"] >= 3.0:
                print(f"       ❌ 误差全图均匀（p50={prof['p50']:.1f}）——"
                      "**渲染/编码设置不同**（光照、纹理、fovy）。")
            else:
                print("       ❌ 误差既不集中也不均匀，剖面不典型，看 --dump 的图。")
            env.close()
            worst_init = max(worst_init, d0)
            continue

        # 重放：RLDS 的动作直接喂回去
        env.reset()
        obs = env.set_init_state(m["state"])
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
        worst_init = max(worst_init, d0)
        worst_end = end if worst_end is None else max(worst_end, end)
        n_ok += end <= args.tol

    print(f"\n{'=' * 60}\n=== 判读 ===")
    print(f"  {n_ok}/{len(eps)} 条 episode 全程误差 ≤ {args.tol}")
    # ⚠️ 末帧误差要区分「测出来是 0」和「压根没测到」。初版两者都印 0.00，
    #    读起来像"末帧完美"，实际是所有 episode 都卡在第 0 帧 continue 掉了。
    end_txt = "未测（都卡在第 0 帧）" if worst_end is None else f"{worst_end:.2f}"
    print(f"  最差的第 0 帧误差 {worst_init:.2f}，最差的末帧误差 {end_txt}")
    if n_ok == len(eps) and worst_end <= args.tol:
        print("  ✅ 回放能复现训练数据，可以给训练集缓存深度，G4/M2 解锁。")
    elif worst_init > args.tol:
        print("  ❌ **第 0 帧就对不上**，问题在初始状态或渲染设置，不是发散。"
              "\n     按上面每条 episode 的三行诊断判：次好只差 <1 = 所有候选都不对；"
              "\n     误差集中 = 物体位置不同；误差均匀 = 渲染/编码设置不同。"
              "\n     若候选里 demo 数为 0，先把 LIBERO 演示数据下下来再说 ——"
              "\n     RLDS 是从演示转换来的，评测初始状态是另一批。")
    else:
        print("  ❌ **随时间发散**：初始状态对得上，重放却越走越远。"
              "\n     多半是控制器设置不同（RLDS 生成时的 controller config）。"
              "\n     此时不能给训练集缓存深度 —— 深度会属于另一个场景，"
              "\n     而 G4 照跑不报错。回退方案见 docs/06 §6（单目深度）。")


if __name__ == "__main__":
    main()
