#!/usr/bin/env python
"""
固定 episode 子集 + 训练集深度缓存 —— **G4 / M2 的前置**（`docs/06` §4.1 纪律 1b）。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/build_subset.py --suite libero_10 --limit 5      # 先冒烟
    python scripts/build_subset.py --suite libero_10                # 全量

一次遍历做三件事（都要渲染同一批帧，分两次跑是白花一倍的时间）：

  ① **覆盖率**：每条 episode 能不能对齐到某条 demo。实测已知 task 9 的两条都
     对不上，HF 上还缺 task 1 的 demo 文件 —— 覆盖率不是 100%。
  ② **固定子集**：能对上的列成清单。⚠️ **六组全部只用这份清单**：若 G4/M2 只用
     "能对上深度"的子集而 G2/G3 用全部，两者数据量不同，比较作废。
  ③ **深度缓存**：对齐后每个 RLDS 帧的 patch 级深度 (T,16,16)，G4/M2 的输入。

判据：

    ① 初始帧误差 ≤ 5.0（平均绝对像素差 0–255）—— 找对了 demo 没有
    ② 匹配下标单调性 ≥ 95%              —— 对齐正确必然接近 100%，噪声不会
    ③ **深度不确定度 p95 < 5 cm**       —— G4 真正消费的量，尺度取自 §9.2

⚠️ **③ 是第二版，第一版用 RGB 逐帧误差当闸，错了。** 首次冒烟 0/3 全栽在
   「≤5.0 的帧占 ≥90%」上（实测 87% 与 64%），而 d0 是 2.1–2.7、单调性 98–99% ——
   **匹配是对的**，差的是"机械臂在两个采样帧之间动了"。`check_replay` 的注释
   早写过这一条：RGB 误差只是代理，深度才是我们要的量；差一帧，**深度**差多少
   才是该问的。我却把已知是代理的量拿来当硬闸。

   改判据是在看到它失败之后做的，这本身就危险。所以：换的是**代理量 → 目标量**，
   而目标量的尺度（跳变阈值 5 cm、体素箱宽 ~30 cm）在 §9.2 就定死了、不是现在
   挑的；且 `--report_only`（缺省）先只报分布不落子集，看过分布再解开。
   RGB 那两个数照样报，只是不再当闸。

⚠️ **可续跑**：每条 episode 的结果即时追加进 `<out>/episodes.jsonl`，
   重跑会跳过已完成的。这台机器每 9 小时要关一次，不可续跑等于跑不完。

⚠️ **渲染走 EGL，占的是 GPU**。训练在跑的话会互相抢：要么等，
   要么 `MUJOCO_GL=osmesa` 走 CPU（慢很多但不抢卡）。
"""

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from libero.libero import benchmark  # noqa: E402

from check_replay import (align_by_state, build_env, demo_all_states,  # noqa: E402
                          diff_profile, find_task, match_init, rgb_of)
from common.tf_cpu import hide_gpu_from_tf  # noqa: E402
from data.depth_cache import hash_bytes  # noqa: E402

TOL = 5.0            # 平均绝对像素差：只用来判"找对 demo 没有"
MIN_MONO = 0.95      # 匹配下标单调性
MAX_DEPTH_P95 = 5.0  # cm。深度不确定度，尺度取自 docs/05 §9.2（箱宽 ~30 cm）


def episode_stream(data_root: str, name: str, want_index=None, want_lang=None):
    """
    逐条产出 episode（**不能一次全读进内存**：libero_10 约 380 条 × 250 帧 ×
    256² × 3 ≈ 18 GB）。同时把 episode 级的元数据一并带出来 —— 训练时要靠它
    把深度查回去，见下面 `describe_features`。
    """
    import tensorflow as tf
    import tensorflow_datasets as tfds

    # ⚠️ TF 默认预留整张卡的显存（本脚本一个 GPU 算子都不用），
    #    同机的训练/评测会直接 OOM。EGL 渲染仍需要 GPU，所以只藏 TF 这一半。
    hide_gpu_from_tf()

    # ⚠️ **必须 SkipDecoding 拿到未解码的 JPEG 字节** —— 深度缓存的键就是它的
    #    指纹（`data/depth_cache`）。训练那边在轨迹变换阶段看到的也正是这份字节，
    #    两边指纹才对得上。解码后再哈希会因增广/重采样而对不上，且不报错。
    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    ds = b.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding()}}})
    for i, ep in enumerate(ds):
        # ⭐ **先判要不要，再解码。** 下面那两行把 episode 的每一帧都
        #    decode_image + 哈希（约 250 帧/条）；续跑或 --only_task 时绝大多数
        #    episode 立刻就被丢掉，那些解码全是白做的。
        #    want_index 只看下标（最便宜），want_lang 看语言指令（仍不碰图像）。
        if want_index is not None and not want_index(i):
            continue
        if want_lang is not None:
            _lang = next(iter(ep["steps"]))["language_instruction"] \
                .numpy().decode().strip().lower()
            if not want_lang(_lang):
                continue
        raw = [s["observation"]["image"] for s in ep["steps"]]
        yield i, {
            "meta": {k: v.numpy().decode() if v.dtype == tf.string else v.numpy()
                     for k, v in ep["episode_metadata"].items()}
            if "episode_metadata" in ep else {},
            "lang": next(iter(ep["steps"]))["language_instruction"]
            .numpy().decode().strip().lower(),
            "images": np.stack([tf.io.decode_image(r, expand_animations=False)
                                .numpy() for r in raw]),
            "hash": np.array([int(hash_bytes([r])[0].numpy()) for r in raw],
                             dtype=np.uint64),
        }


def describe_features(data_root: str, name: str) -> None:
    """
    ⚠️ **先回答一个会决定整个设计的问题：训练时拿什么把深度查回去？**

    RLDS 在训练时是打乱 + 交错读的，位置下标没有意义；而 `KFrameBatchTransform`
    拿到的图像**已经过增广**，按内容做指纹也不成立。所以键必须是 episode 级
    元数据里某个**唯一**的字段，并在轨迹变换里一路带到 batch transform。
    这里先把有什么打出来，再定 —— 猜错了就是又一份对不上的缓存。
    """
    import tensorflow_datasets as tfds

    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    print("episode 级字段（steps 之外）:")
    for k, v in b.info.features.items():
        if k != "steps":
            print(f"  {k}: {v}")
    ep = next(iter(b.as_dataset(split="train", shuffle_files=False).take(1)))
    for k, v in ep.items():
        if k == "steps":
            continue
        print(f"  第 0 条的值 {k} = {v}")


def depth_uncertainty(dep: np.ndarray, mj: np.ndarray) -> tuple[float, float]:
    """
    每个 RLDS 帧的**深度不确定度**：它匹配到的那一格与相邻格之间的 patch 深度差。

    时间对齐必然有量化误差（渲染步长 + no_noops 删帧），差一帧带来的深度偏移
    就是这个量的量级。拿它和 G4 的两个尺度比：跳变阈值 5 cm、体素箱宽 ~30 cm。
    返回 (p50, p95)，单位 cm。
    """
    if len(dep) < 2:
        return float("nan"), float("nan"), float("nan")
    dd = np.abs(np.diff(dep, axis=0)) * 100.0            # (n-1,16,16) cm
    j = np.clip(mj, 0, len(dd) - 1)
    v = dd[j].ravel()
    # ⚠️ p50 通常**恰好是 0**：桌面与背景静止，只有机械臂和物体在动。
    #    所以 p95 量的正是"会动的那几个 patch"，也正是 G4 要用的那些。
    #    非零占比一并报出来，免得把 p50=0 读成"深度没变化"。
    return (float(np.percentile(v, 50)), float(np.percentile(v, 95)),
            float((v > 0).mean()))


def judge(d0: float, mono: float, dp95: float) -> tuple[bool, str]:
    if d0 > TOL:
        return False, f"初始帧对不上（{d0:.2f} > {TOL}）"
    if mono < MIN_MONO:
        return False, f"匹配下标单调性 {mono:.1%} < {MIN_MONO:.0%}"
    if not (dp95 < MAX_DEPTH_P95):
        return False, f"深度不确定度 p95 {dp95:.1f} cm ≥ {MAX_DEPTH_P95} cm"
    return True, "ok"


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # 管道下的块缓冲，第三次了
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_10")
    ap.add_argument("--data_root", default=str(
        Path("~/autodl-tmp/datasets/modified_libero_rlds").expanduser()))
    ap.add_argument("--out", default="results/subset")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（冒烟用）")
    ap.add_argument("--align_stride", type=int, default=1,
                    help="逐帧对齐的渲染步长。**1 是定稿值**：2 时深度不确定度"
                         "dp95 中位 8.1 cm 过不了闸，1 时 3.9 cm（docs/05 §12.1）")
    ap.add_argument("--n_init_try", type=int, default=50)
    ap.add_argument("--stage1_frames", default="0,20,40,60",
                    help="选 demo 时在每条 demo 的哪几个帧号上比对。只比第 0 帧时，"
                         "no_noops 削掉长开头的 episode 会选错 demo —— 全量实测 90 条"
                         "栽在初始帧上，d0 中位 10.9 而次好只差 0.2（docs/05 §12.2）")
    ap.add_argument("--demo_frames", type=int, default=80,
                    help="在选中的那条 demo 里往后扫多少帧找偏移（no_noops 用）")
    ap.add_argument("--redo_failed", action="store_true",
                    help="只重跑之前失败的 episode（改了 --demo_frames 之类时用）")
    ap.add_argument("--depth_only", action="store_true",
                    help="⭐ 只给**已判定通过**的 episode 补落深度缓存，跳过选 demo 与"
                         "偏移搜索（那 560 次候选渲染的答案已经记在 episodes.jsonl 的 "
                         "demo/frame_off 里，不必重做）。用于 --report_only 跑完之后"
                         "才想起要落盘的情形 —— 逐帧对齐的约 250 次渲染躲不掉，"
                         "深度就是从那一趟出来的。已有 depth/epNNNN.npz 的会跳过，可续跑。")
    ap.add_argument("--only_task", type=int, default=-1,
                    help="只处理这一个 task。失败是按 task 聚集的（交叉表见 §12.3），"
                         "针对某个 task 调搜索参数时，不必把别的 task 的失败项"
                         "一起重渲染。⚠️ 它**只缩小重跑范围，不碰任何判据**")
    ap.add_argument("--report_only", action="store_true", default=True,
                    help="只报分布、不落深度缓存与子集（缺省）")
    ap.add_argument("--commit", dest="report_only", action="store_false",
                    help="看过分布、确认判据之后才用它落盘")
    ap.add_argument("--features_only", action="store_true",
                    help="只打印 episode 级字段就退出（决定深度用什么键查）")
    args = ap.parse_args()
    name = f"{args.suite}_no_noops"

    if args.features_only:
        describe_features(args.data_root, name)
        return

    # ⚠️ 续跑日志必须按 align_stride 分开。深度不确定度**直接由步长决定**
    #    （相邻渲染帧隔 stride 个 demo 帧），换了步长重跑却跳过已完成的 episode，
    #    两组数就混进同一份清单 —— 而混了不会报错，只会让子集的判据不一致。
    out = Path(args.out) / f"{args.suite}_s{args.align_stride}"
    (out / "depth").mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    recs0 = [json.loads(l) for l in log.open()] if log.exists() else []
    # ⚠️ 只重跑失败项时，把旧的失败记录**从日志里删掉**再重来 —— 留着的话
    #    汇总会把同一条 episode 数两次，覆盖率就是错的（而且不会报错）。
    if args.redo_failed:
        # --only_task 时只丢该 task 的失败记录，别的 task 的失败保持原样
        keep = [r for r in recs0 if r.get("ok")
                or (args.only_task >= 0 and r.get("task") != args.only_task)]
        n_drop = len(recs0) - len(keep)
        log.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                               for r in keep))
        recs0 = keep
        print(f"重跑失败项：丢掉 {n_drop} 条失败记录，保留 {len(keep)} 条已通过的")
    done = {r["ep"] for r in recs0}
    if done:
        print(f"续跑：已完成 {len(done)} 条，跳过")

    bmark = benchmark.get_benchmark_dict()[args.suite]()
    env, env_tid, cand_cache = None, None, {}
    try:
        if args.depth_only:
            # 只补已通过、且深度还没落盘的那些
            need = {r["ep"] for r in recs0 if r.get("ok")
                    and not (out / "depth" / f"ep{r['ep']:04d}.npz").exists()}
            demo_of = {r["ep"]: r.get("demo") for r in recs0 if r.get("ok")}
            print(f"补深度缓存：{len(need)} 条待落盘"
                  f"（已通过 {sum(1 for r in recs0 if r.get('ok'))} 条）")
            want_i = lambda i: i in need
        else:
            want_i = (lambda i: i not in done) if done else None
        want_l = (None if args.only_task < 0
                  else (lambda lg: find_task(bmark, lg) == args.only_task))
        for i, ep in episode_stream(args.data_root, name, want_i, want_l):
            if args.limit and i >= args.limit:
                break
            tid = find_task(bmark, ep["lang"])
            rec = {"ep": i, "task": tid, "lang": ep["lang"],
                   "n_frames": int(len(ep["images"])), "meta": ep["meta"],
                   "stride": args.align_stride, "demo_frames": args.demo_frames,
                   "stage1": args.stage1_frames}
            if tid is None:
                rec.update(ok=False, why="任务描述对不上任何 task")
                _append(log, rec)
                continue

            # ⚠️ 同一个 task 的 env 复用：建一次要编译 MuJoCo 模型，几秒起步，
            #    每条 episode 重建就是白花一小时。RLDS 按 task 分文件存，
            #    顺序读时切换很少 —— 真切换了也只是慢一点，不影响正确性。
            if env is None or tid != env_tid:
                if env is not None:
                    env.close()
                env = build_env(bmark.get_task(tid))
                env.seed(0)
                env_tid = tid
                # ⚠️ 换 task 必须换缓存：候选画面是按 task 的，串了不会报错，
                #    只会拿另一个 task 的画面去匹配。
                cand_cache = {}

            if args.depth_only:
                # ⭐ 跳过选 demo 与偏移搜索：答案已在日志里。**不重判**，
                #    只把深度补落盘 —— 重判会用当前参数重算，与清单不一致。
                key = demo_of.get(i)
                st_all = (demo_all_states(bmark.get_task(tid), key)
                          if key else None)
                if st_all is None:
                    print(f"  [{i:>4}] ⚠️ 取不到 demo {key!r} 的全帧状态，跳过")
                    continue
                _, mj, _, dep = align_by_state(env, st_all, ep["images"],
                                               True, args.align_stride)
                np.savez(out / "depth" / f"ep{i:04d}.npz",
                         hash=ep["hash"], depth=dep[mj].astype(np.float16))
                print(f"  [{i:>4}] task {tid}  深度已落盘 "
                      f"({len(ep['hash'])} 帧, demo={key})")
                continue

            m = match_init(env, bmark, tid, ep["images"][0], args.n_init_try,
                           args.demo_frames,
                           tuple(int(x) for x in args.stage1_frames.split(",")),
                           cache=cand_cache)
            env.reset()
            obs = env.set_init_state(m["state"])
            prof = diff_profile(rgb_of(obs, True), ep["images"][0])
            rec.update(d0=m["best"], d1=m["second"], src=m["src"],
                       frame_off=m.get("frame_off", 0), p50=prof["p50"],
                       top10=prof["top10_share"])

            key = m["src"].split(":", 1)[1] if m["src"].startswith("demo:") else None
            st_all = demo_all_states(bmark.get_task(tid), key) if key else None
            if m["best"] > TOL or st_all is None:
                rec.update(ok=False, why=("初始帧对不上" if m["best"] > TOL
                                          else "取不到该 demo 的全帧状态"))
                _append(log, rec)
                continue

            errs, mj, idxs, dep = align_by_state(env, st_all, ep["images"],
                                                 True, args.align_stride)
            good = float((errs <= TOL).mean())
            mono = float((np.diff(idxs[mj]) >= 0).mean())
            dp50, dp95, dnz = depth_uncertainty(dep, mj)
            ok, why = judge(m["best"], mono, dp95)
            rec.update(good=good, mono=mono, dp50=dp50, dp95=dp95, dnz=dnz,
                       ok=ok, why=why, demo=key, stride=args.align_stride,
                       demo_frames=args.demo_frames)
            if ok and not args.report_only:
                # 对齐后每个 RLDS 帧的 patch 级深度 (T,16,16)，按帧指纹存。
                # float16 足够：分辨率 ~1 mm，而体素箱宽约 30 cm、跳变阈值 5 cm。
                np.savez(out / "depth" / f"ep{i:04d}.npz",
                         hash=ep["hash"], depth=dep[mj].astype(np.float16))
            _append(log, rec)
            print(f"  [{i:>4}] task {tid}  d0={m['best']:.2f}  RGB≤{TOL:g} {good:.0%}  "
                  f"单调 {mono:.0%}  深度 p95={dp95:.1f} cm（{dnz:.0%} 的 patch 在动）"
                  f"  → {'✅' if ok else '❌ ' + why}")
    finally:
        if env is not None:
            env.close()

    summarize(log, out, args.report_only and not args.depth_only)


def write_bbox(recs, out: Path, parts, cam_json="results/tables/camera_libero.json"):
    """
    ⭐ 工作空间包围盒**从这份训练深度缓存本身算**，不沿用评测轨迹那一版。

    `metric_extent` 要的是一个**跨样本固定**的常量；用哪批数据算它，就该是
    模型真正会看到的那批。`dump_camera.py` 早先那版是从评测轨迹统计的
    （30 条 episode），量级参考可以，但训练用的是另一批帧。
    """
    import torch

    from common.camera import Camera

    cj = Path(cam_json)
    if not cj.exists():
        print(f"（找不到 {cj}，跳过包围盒 —— 先跑 scripts/dump_camera.py）")
        return
    cams = json.loads(cj.read_text())
    tid_of = {r["ep"]: r["task"] for r in recs if r.get("ok")}
    xyz = []
    for f in parts:
        ep = int(f.stem[2:])
        c = cams.get(str(tid_of.get(ep)))
        if c is None:
            continue
        cam = Camera(fovy=float(c["fovy"]), height=int(c["height"]),
                     width=int(c["width"]),
                     pos=torch.tensor(c["pos"], dtype=torch.float64),
                     rot=torch.tensor(c["rot"], dtype=torch.float64).reshape(3, 3),
                     flipped=bool(c.get("flipped", True)))
        d = torch.from_numpy(np.load(f)["depth"].astype(np.float64))
        xyz.append(cam.patch_xyz(d.reshape(len(d), -1)).reshape(-1, 3).numpy())
    if not xyz:
        print("（没有可用的相机，跳过包围盒）")
        return
    v = np.concatenate(xyz)
    lo, hi = np.percentile(v, [1, 99], axis=0)
    (out / "bbox.json").write_text(json.dumps(
        {"lo": lo.tolist(), "hi": hi.tolist(), "n_points": int(len(v)),
         "note": "p1–p99，由本次训练深度缓存反投影得到；metric_extent 的常量"},
        indent=1))
    print("包围盒（p1–p99，米）: " + "  ".join(
        f"{n}[{a:+.3f},{b:+.3f}]" for n, a, b in zip("xyz", lo, hi))
        + f"  -> {out / 'bbox.json'}")


def _append(log: Path, rec: dict) -> None:
    with log.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def summarize(log: Path, out: Path, report_only: bool = True) -> None:
    recs = [json.loads(l) for l in log.open()]
    ok = [r for r in recs if r.get("ok")]
    print(f"\n=== 覆盖率：{len(ok)}/{len(recs)} = {len(ok) / max(len(recs), 1):.1%} ===")

    by_task: dict = {}
    for r in recs:
        t = by_task.setdefault(r.get("task"), [0, 0])
        t[1] += 1
        t[0] += bool(r.get("ok"))
    print(f"{'task':>6} {'通过':>8} {'占比':>7}")
    for t in sorted(by_task, key=lambda x: (x is None, x)):
        a, n = by_task[t]
        print(f"{str(t):>6} {a:>4}/{n:<3} {a / n:>7.0%}")

    # ⚠️ 报分布，不只报通过率 —— 阈值定得合不合理，只有看见分布才判得了
    for k in ("d0", "good", "mono", "dp95", "dnz"):
        v = np.array([r[k] for r in recs if k in r], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            print(f"  {k:>5} 分位 p10={np.percentile(v, 10):.3f} "
                  f"p50={np.median(v):.3f} p90={np.percentile(v, 90):.3f} "
                  f"max={v.max():.3f}")
    # ⚠️ `why` 里带着具体数值（"匹配下标单调性 92.4% < 95%"），直接当 key 计数
    #    等于每条自成一类，打出来是一面看不懂的墙 —— 而**真正要回答的问题是
    #    "某个 task 死在哪道闸上"**：若失败在各 task 间均匀，那是判据整体偏紧；
    #    若集中在一两个 task，那是那几个 task 有具体毛病，两者处置完全不同。
    #    所以：把原因归成三桶，并且**按 task 交叉**。
    def bucket(w: str) -> str:
        if "初始帧" in w:
            return "初始帧对不上"
        if "单调性" in w:
            return "对齐单调性"
        if "深度不确定度" in w:
            return "深度不确定度"
        return "其他"

    buckets = ("初始帧对不上", "对齐单调性", "深度不确定度", "其他")
    cross: dict = {}
    for r in recs:
        if not r.get("ok"):
            c = cross.setdefault(r.get("task"), dict.fromkeys(buckets, 0))
            c[bucket(r.get("why", "?"))] += 1
    if cross:
        tot = dict.fromkeys(buckets, 0)
        print(f"\n  失败原因 × task（首个触发的闸）")
        print(f"  {'task':>6} " + " ".join(f"{b:>12}" for b in buckets) + f" {'小计':>6}")
        for t in sorted(cross, key=lambda x: (x is None, x)):
            c = cross[t]
            for b in buckets:
                tot[b] += c[b]
            print(f"  {str(t):>6} " + " ".join(f"{c[b]:>12}" for b in buckets)
                  + f" {sum(c.values()):>6}")
        print(f"  {'合计':>6} " + " ".join(f"{tot[b]:>12}" for b in buckets)
              + f" {sum(tot.values()):>6}")

    # ⚠️ 每道闸**单独**的淘汰率（不是"首个触发"）。上表是短路求值的结果，
    #    排在前面的闸会掩盖后面的；要判某道闸是不是定得太紧，得看它单独淘汰多少。
    gates = (("初始帧对不上", "d0", lambda v: v > TOL),
             ("对齐单调性", "mono", lambda v: v < MIN_MONO),
             ("深度不确定度", "dp95", lambda v: v >= MAX_DEPTH_P95))
    print("\n  每道闸单独的淘汰率（互不掩盖）")
    surv = np.ones(len(recs), dtype=bool)
    for name, key, bad in gates:
        v = np.array([r.get(key, np.nan) for r in recs], dtype=float)
        f = np.isfinite(v) & bad(v)
        surv &= ~f
        print(f"  {name:>12}: {f.sum():>3}/{len(recs)} = {f.mean():>5.1%} 淘汰")
    print(f"  {'三闸同时通过':>12}: {surv.sum()}/{len(recs)} = {surv.mean():.1%}"
          f"  ← 三个各淘汰约一成的闸叠起来就是这个数")

    if report_only:
        print("\n（--report_only：没有落深度缓存，也没有写子集清单。"
              "看过上面的分布、确认判据之后，加 --commit 重跑。）")
        return

    # 合并成一张表，训练侧只读这一个文件（DepthCache.load）
    parts = sorted((out / "depth").glob("ep*.npz"))
    if parts:
        hs, ds_ = [], []
        for f in parts:
            z = np.load(f)
            hs.append(z["hash"])
            ds_.append(z["depth"])
        h, d = np.concatenate(hs), np.concatenate(ds_)
        u, first = np.unique(h, return_index=True)
        if len(u) != len(h):
            print(f"  ⚠️ {len(h) - len(u)} 帧指纹重复（同一帧被两条 episode 收录？）"
                  "，只保留首次出现的")
            h, d = h[np.sort(first)], d[np.sort(first)]
        np.savez_compressed(out / "depth_cache.npz", hash=h, depth=d)
        print(f"深度缓存 -> {out / 'depth_cache.npz'}  {len(h)} 帧，"
              f"{(out / 'depth_cache.npz').stat().st_size / 2**20:.0f} MB")
        write_bbox(recs, out, parts)

    sub = out / "subset.json"

    sub.write_text(json.dumps({
        # ⚠️ 这里要记**当前真正生效的三道闸**。原先写的 min_good 是第一版
        #    RGB 代理闸的阈值，§12.1 换成深度不确定度之后常量就删了，
        #    这一行没跟着改 —— 好在是 NameError 直接崩，没有静默写下一份错的判据记录。
        "criteria": {"tol": TOL, "min_mono": MIN_MONO,
                     "max_depth_p95": MAX_DEPTH_P95},
        "n_total": len(recs), "n_ok": len(ok),
        "episodes": [{"ep": r["ep"], "task": r["task"], "demo": r.get("demo")}
                     for r in ok],
    }, ensure_ascii=False, indent=1))
    print(f"\n子集清单 -> {sub}")
    print("⚠️ 这份清单要连同结果一起发布 —— 它是复现的一部分（docs/06 §4.1 1b）。")
    print("⚠️ **六组全部只用这份清单**，只给 G4/M2 用会让数据量不同，比较作废。")


if __name__ == "__main__":
    main()
