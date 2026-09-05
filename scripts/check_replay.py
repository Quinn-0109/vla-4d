#!/usr/bin/env python
"""
回放保真度验证 —— **G4 / M2 的唯一前置**（`docs/06` §4.4）。

G4 要的是**训练集**每一帧的 patch 级深度。RLDS 里没有深度，只能把动作序列
放回仿真器、取仿真器的深度。这一步成立的前提是：

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
  4. ⭐ **逐帧设状态**（不重放动作）对齐并比对画面

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

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from common.tf_cpu import hide_gpu_from_tf
    hide_gpu_from_tf()          # TF 会预留整张卡，而这里一个 GPU 算子都不用

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


# ⚠️ **HDF5 路径要缓存。** 下面三个函数原先各自做 `root.rglob(...)` ——
#    递归遍历整个 LIBERO 数据集目录，而它们**每条 episode 都会被调用**
#    （demo_states_at 一次、demo_states 一次、demo_all_states 一次）。
#    目录里是几十 GB 的 HDF5，三次全盘扫描 × 每条 episode，是 build_subset
#    "每条都要等很久"的一部分。按 task.name 缓存，进程内只扫一次。
_H5_CACHE: dict = {}


def _demo_h5(task):
    """某个 task 的 demo HDF5 路径（进程内缓存）。找不到返回 None。"""
    if task.name in _H5_CACHE:
        return _H5_CACHE[task.name]
    try:
        from libero.libero import get_libero_path
        root = Path(get_libero_path("datasets"))
    except Exception:
        _H5_CACHE[task.name] = None
        return None
    hits = list(root.rglob(f"{task.name}_demo.hdf5")) or list(
        root.rglob(f"{task.name}*.hdf5"))
    _H5_CACHE[task.name] = hits[0] if hits else None
    return _H5_CACHE[task.name]


def demo_all_states(task, demo_key: str):
    """取某条 demo 的**全部帧**的完整仿真状态 (T, state_dim)。"""
    try:
        import h5py
    except Exception:
        return None
    h5 = _demo_h5(task)
    if h5 is None:
        return None
    with h5py.File(h5, "r") as f:
        return np.asarray(f["data"][demo_key]["states"])


def patch_depth(env, d: np.ndarray, flip: bool, grid: int = 16, patch: int = 16):
    """深度缓冲 → 米 → patch 级均值 (16,16)。与 depth_diag 同一套换算与翻转。"""
    from PIL import Image
    try:
        from robosuite.utils.camera_utils import get_real_depth_map
        m = get_real_depth_map(env.sim, d[..., None])[..., 0]
    except Exception:
        mo = env.sim.model
        ext = mo.stat.extent
        near, far = mo.vis.map.znear * ext, mo.vis.map.zfar * ext
        m = near / (1.0 - d * (1.0 - near / far))
    if flip:
        m = m[::-1, ::-1]
    m = np.array(Image.fromarray(m.astype(np.float32), mode="F")
                 .resize((224, 224), Image.BILINEAR))
    return m.reshape(16, 14, 16, 14).transpose(0, 2, 1, 3).reshape(16, 16, -1).mean(-1)


def align_by_state(env, states, rlds, flip: bool, stride: int = 2):
    """
    ⭐ **逐帧设状态，不重放动作。**

    之前是 `env.step(action)` 一路走下去 —— 那必然累积误差：接触动力学
    （抓取、碰撞）会把微小差异放大。实测 task 6 从中段开始漂（t=142 时 4.6，
    末段 12–15），task 9 从头就偏；而 task 4 那条 100% 的帧都 ≤5，
    说明**渲染与坐标那一侧本来就是对的，漂的是重放这个方法**。
    demo HDF5 里存着每一帧的完整仿真状态，逐帧 set 就是零漂移。

    顺带解决 no_noops 的对齐：先把整条 demo 渲一遍，再给每个 RLDS 帧找最匹配的。
    **匹配到的下标必须单调递增** —— 免费的强验证：对齐正确必然单调，噪声不会。

    返回 (每个 RLDS 帧的误差, 匹配到的 demo 下标)。
    """
    def small(a):
        return a[::8, ::8].astype(np.int16)

    # ⚠️ **不要每帧 env.reset()。** `set_init_state` 本来就设置完整状态，
    #    reset 会把整个仿真器重新初始化 —— 280 帧就是 280 次无用的重置，
    #    这是 --align_stride 1 "跑不出结果"的真正原因（另一半是管道块缓冲）。
    env.reset()
    bank, dep, idxs = [], [], list(range(0, len(states), stride))
    for n, t in enumerate(idxs):
        obs = env.set_init_state(states[t])
        bank.append(rgb_of(obs, flip))
        d = np.asarray(obs[f"{CAM}_depth"])
        dep.append(patch_depth(env, d[..., 0] if d.ndim == 3 else d, flip))
        if n % 50 == 0:
            print(f"         …渲染 {n}/{len(idxs)} 帧", flush=True)
    bank_s = np.stack([small(b) for b in bank])
    errs, matches = [], []
    for r in rlds:
        j = int(np.abs(bank_s - small(r)).mean(axis=(1, 2, 3)).argmin())
        errs.append(diff(bank[j], r))
        matches.append(j)
    dep = np.stack(dep)                                  # (n_bank, 16, 16)
    return np.asarray(errs), np.asarray(matches), np.asarray(idxs), dep


def demo_states(task, n_demo: int = 60, n_frame: int = 1):
    """
    从 LIBERO 的**演示** HDF5 里取状态。`n_frame=1` 只取每条 demo 的第 0 帧。

    ⚠️ 这是与 `get_task_init_states()` **不同的一批**：后者给的是**评测用**的
    50 个初始摆放，而 modified_libero_rlds 是从**演示数据**转换来的。
    实测：换成 demo 之后误差从 ~10 降到 2.1–2.7（JPEG 底噪量级）。

    ⚠️ **`n_frame > 1` 是为 `no_noops` 准备的。** 数据集名字里的 no_noops 意思是
    转换时删掉了动作为零的步骤；若某条 demo 开头有 noop 被删，**RLDS 的第 0 帧
    就不是 demo 的初始状态，而是走了几步之后的状态**。这时只比 states[0] 必然
    对不上（实测 task 9：demo 候选全试过仍 6.41，次好只差 0.02）。
    """
    try:
        import h5py
    except Exception:
        return []
    h5 = _demo_h5(task)
    if h5 is None:
        return []
    out = []
    with h5py.File(h5, "r") as f:
        for k in list(f["data"].keys())[:n_demo]:
            st = f["data"][k]["states"]
            for t in range(min(n_frame, st.shape[0])):
                out.append((k, t, np.asarray(st[t])))
    return out


def demo_states_at(task, n_demo: int, frames):
    """取每条 demo 在指定帧号上的状态。`frames` 是帧号序列（越界的跳过）。"""
    try:
        import h5py
    except Exception:
        return []
    h5 = _demo_h5(task)
    if h5 is None:
        return []
    out = []
    with h5py.File(h5, "r") as f:
        for k in list(f["data"].keys())[:n_demo]:
            st = f["data"][k]["states"]
            for t in frames:
                if t < st.shape[0]:
                    out.append((k, t, np.asarray(st[t])))
    return out


def match_init(env, bmark, tid: int, ref: np.ndarray, n_try: int,
               n_frame: int = 1, stage1_frames=(0,), cache=None):
    """
    逐个试初始状态，找与 RLDS 第 0 帧最像的那个；同时定下翻转与否。
    返回 (init_idx, flip, 最好误差, 次好误差, 不翻的最好误差)。

    ⚠️ **必须同时报次好**。只报最好分不清两种情况：
      · best 10.2 / second 25   → 确实找到了那一个初始状态，10.2 是渲染差异
      · best 10.2 / second 10.4 → 所有候选都差不多，初始状态根本不是区分因素，
                                  问题在渲染设置（而"最像的那个"只是噪声）
    """
    evals = list(bmark.get_task_init_states(tid))[:n_try]
    # 两段搜索：先挑出**哪一条 demo**（便宜），再在这一条里扫前 n_frame 帧
    # 找**偏移量**（应对 no_noops 削掉的开头）。直接 50 demo × 80 帧 = 4000 次
    # 渲染太慢，所以分两段。
    #
    # ⚠️ **第一段不能只比每条 demo 的第 0 帧。** 实测 libero_10 有 90 条 episode
    #    栽在初始帧上，且 d0 中位数 10.9、次好只差 0.2 —— 那是"正确的那个根本
    #    不在候选里"的形态。成因：若 no_noops 削掉的开头很长，RLDS 的第 0 帧
    #    实际是 demo 的第 60 帧，拿它去比 50 条 demo 的**第 0 帧**，选中的是
    #    "哪条 demo 的起始画面最像这个中途画面"——基本是随机的。选错了 demo，
    #    第二段的偏移搜索再准也没用（这也解释了为什么 demo_frames 40→80 只救回一条：
    #    瓶颈不在偏移范围，在选 demo 那一步）。
    #    `stage1_frames` 让第一段在每条 demo 的若干个帧号上都比一遍。
    d0 = demo_states_at(bmark.get_task(tid), n_try, stage1_frames)
    cands = [("eval", str(j), st) for j, st in enumerate(evals)] + \
            [(f"demo:{k}", str(t), st) for k, t, st in d0]
    # ⭐ **候选的渲染与 ref 无关** —— `env.reset() + set_init_state(st)` 只取决于
    #    状态本身，只有最后的 `diff(..., ref)` 才用到当前 episode。同一个 task 的
    #    每条 episode 面对的是**同一批候选**，不缓存就是把 400 次渲染 + 400 次
    #    MuJoCo reset 重复几十遍（task 3 的 27 条 = 10800 次，其中 10400 次是白做的）。
    #    传一个 dict 进来即可跨 episode 复用；**切换 task 时必须换一个新 dict**，
    #    否则会拿另一个 task 的画面去比 —— 那不会报错，只会给出错误的匹配。
    scores = []
    for src, j, st in cands:
        key = (src, j)
        if cache is not None and key in cache:
            rgb_t, rgb_f = cache[key]
        else:
            env.reset()
            obs = env.set_init_state(st)
            rgb_t, rgb_f = rgb_of(obs, True), rgb_of(obs, False)
            if cache is not None:
                cache[key] = (rgb_t, rgb_f)
        scores.append((diff(rgb_t, ref), diff(rgb_f, ref), src, j, st))
    flipped = sorted(x[0] for x in scores)
    best = min(scores, key=lambda x: x[0])
    res = {"src": best[2], "idx": best[3], "state": best[4],
           "best": flipped[0], "second": flipped[1],
           "noflip": min(x[1] for x in scores),
           "n_eval": len(evals), "n_demo": len(d0),
           # 第一段若是在非 0 帧上选中的，那个帧号本身就是偏移量的起点
           "frame_off": int(best[3]) if best[2].startswith("demo:") else 0}

    # 第二段：只在选中的那条 demo 里扫帧偏移
    if n_frame > 1 and best[2].startswith("demo:"):
        key = best[2].split(":", 1)[1]
        for k, t, st in demo_states(bmark.get_task(tid), n_try, n_frame):
            if k != key or t == 0:
                continue
            env.reset()
            obs = env.set_init_state(st)
            d = diff(rgb_of(obs, True), ref)
            if d < res["best"]:
                res.update(best=d, state=st, frame_off=t)
    return res


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
    # ⚠️ 管道输出（| grep / | tee）会让 stdout 变成块缓冲，跑十几分钟一个字
    #    都看不到，看起来像卡死。probe_vram.py 记过这个坑，这里又漏了。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--data_root", default=str(
        Path("~/autodl-tmp/datasets/modified_libero_rlds").expanduser()))
    ap.add_argument("--n_episodes", type=int, default=5)
    ap.add_argument("--n_init_try", type=int, default=50)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--align_stride", type=int, default=2,
                    help="逐帧对齐时每隔几帧渲染一次")
    ap.add_argument("--demo_frames", type=int, default=40,
                    help="在选中的那条 demo 里往后扫多少帧找偏移（no_noops 用）")
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
    depth_p95, rgb_good, monos, n_align = [], [], [], 0
    for e_i, ep in enumerate(eps):
        tid = find_task(bmark, ep["lang"])
        if tid is None:
            print(f"  [{e_i}] ❌ 任务描述对不上任何 task：{ep['lang'][:60]!r}")
            continue
        env = build_env(bmark.get_task(tid))
        env.seed(0)
        m = match_init(env, bmark, tid, ep["images"][0], args.n_init_try,
                       args.demo_frames)
        j, flip, d0, d1, d_noflip = m["idx"], True, m["best"], m["second"], m["noflip"]
        print(f"\n  [{e_i}] task {tid}  RLDS 图像 {ep['images'].shape[1:]}  "
              f"候选 {m['n_eval']} eval + {m['n_demo']} demo")
        off = f"，帧偏移 +{m['frame_off']}" if m.get("frame_off") else ""
        print(f"       最像的是 **{m['src']}**{off}  第0帧误差 {d0:.2f}")
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

        # ⭐ 逐帧设状态对齐（不重放动作，见 align_by_state）
        key = m["src"].split(":", 1)[1] if m["src"].startswith("demo:") else None
        st_all = demo_all_states(bmark.get_task(tid), key) if key else None
        if st_all is None:
            print("       ⚠️ 取不到该 demo 的全帧状态，跳过对齐检验。")
            env.close()
            worst_init = max(worst_init, d0)
            continue
        errs, mj, idxs, dep = align_by_state(env, st_all, ep["images"], flip,
                                             args.align_stride)
        env.close()
        matches = idxs[mj]
        good = float((errs <= args.tol).mean())
        mono = float((np.diff(matches) >= 0).mean())
        print(f"       逐帧对齐（demo {len(st_all)} 帧，步长 {args.align_stride}）："
              f"p50={np.median(errs):.1f} p95={np.percentile(errs, 95):.1f} "
              f"max={errs.max():.1f}  ≤{args.tol} 占 **{good:.1%}**")
        print(f"         匹配下标单调性 {mono:.1%}"
              f"（对齐正确必然接近 100%）  覆盖 {matches.min()}–{matches.max()}")
        # ⭐ **RGB 误差只是代理，深度才是我们要的量。**
        #    时间对齐有量化误差（步长 + no_noops 删帧），机械臂在动时差半帧
        #    就有几个像素差 —— 但真正该问的是：差一帧，**深度**差多少？
        #    拿它和 G4 的两个尺度比：跳变阈值 5 cm、体素箱宽 ~30 cm。
        dd = np.abs(np.diff(dep, axis=0)).ravel() * 100          # cm
        print(f"         ⭐ 相邻帧的 patch 深度差（= 对齐量化误差的上界）："
              f"p50={np.median(dd):.2f} p95={np.percentile(dd, 95):.2f} "
              f"max={dd.max():.1f} cm")
        print(f"            对照：跳变阈值 5 cm，体素箱宽 ~30 cm")
        depth_p95.append(float(np.percentile(dd, 95)))
        rgb_good.append(good)
        monos.append(mono)
        n_align += 1
        end_err = float(errs[-1])
        worst_init = max(worst_init, d0)
        worst_end = end_err if worst_end is None else max(worst_end, end_err)
        n_ok += good >= 0.95

    print(f"\n{'=' * 60}\n=== 判读 ===")
    print(f"  能对上 demo 并完成对齐的 episode：{n_align}/{len(eps)}")
    if not depth_p95:
        print("  ❌ 一条都没对齐上，回放取深度这条路不通。"
              "\n     回退方案见 docs/06 §6（单目深度估计）。")
        return

    # ⭐ **主判据是深度，不是 RGB。**
    #    我们要缓存的是深度；RGB 只是找对应帧的手段。RGB 误差里混着机械臂的
    #    亚帧位移 —— 它对像素敏感，对 patch 深度不敏感。真正该问的是：
    #    「选错一帧」的代价（相邻帧深度差）相对于 G4 的两个尺度有多大。
    p95 = float(np.median(depth_p95))
    print(f"  相邻帧 patch 深度差的 p95（中位于各 episode）：**{p95:.2f} cm**")
    print(f"    对照：跳变阈值 5 cm，体素箱宽 ~30 cm")
    print(f"  RGB 误差 ≤{args.tol} 的帧占比：{np.median(rgb_good):.1%}（各 episode 中位）")
    print(f"  匹配下标单调性：{np.median(monos):.1%}（对齐正确必然接近 100%）")

    if p95 < 5.0 and np.median(monos) > 0.95:
        print("\n  ✅ **可用**：对齐量化误差小于跳变阈值，更远小于箱宽；"
              "\n     且匹配单调，说明帧对应关系正确。可以给训练集缓存深度。")
    elif p95 < 30.0:
        print(f"\n  🔶 深度误差 {p95:.1f} cm 介于跳变阈值与箱宽之间："
              "\n     分箱不受影响，但跳变判定会被污染。§9.2 的跳变率要重测。")
    else:
        print(f"\n  ❌ 深度误差 {p95:.1f} cm 与箱宽同量级，分箱会错。回退单目深度。")

    if n_align < len(eps):
        print(f"\n  ⚠️ **{len(eps) - n_align}/{len(eps)} 条对不上，这是个混淆源。**"
              "\n     若 G4/M2 只用能对上的子集训练，而 G2/G3 用全部，"
              "\n     两者的数据量就不同，比较作废。"
              "\n     **所有臂必须用同一个子集** —— 先跑大样本统计覆盖率，"
              "\n     再决定是排除 episode 还是排除整个 task。")
