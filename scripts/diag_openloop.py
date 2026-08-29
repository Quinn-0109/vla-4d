#!/usr/bin/env python
"""
开环动作诊断 —— **成功率为 0 时先跑这个，不要直接重跑 6 小时的评测。**

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/diag_openloop.py --arm G2 --adapter runs/<...>/adapter/step30000 \
        --data_root_dir ~/autodl-tmp/datasets/modified_libero_rlds

**不开仿真器**，直接从 RLDS 训练集取画面喂进模型，把预测动作与数据集里的真动作比。
测五个变体，每一对之间的差就是一个判据：

    tf         teacher forcing 单次前向 —— **与训练时逐位相同的那条路径**。
               它的 token 准确率可以直接和训练日志里的 acc 比。
    gen        `predict_action`，带 KV cache 自回归 —— 评测走的那条路径
    gen_nocache 同上但 `use_cache=False`。与 gen 的差 = **cache 那条路的问题**
    nocrop     gen 但不做中心裁（训练是 image_aug=True，见 common/imgproc）
    shuffled   gen 但画面换成同 episode 的随机时刻。与 gen 打平 = 画面没进决策

这么排的理由：训练 acc 高、评测成功率 0，这两件事只能同时成立于
"训练那条前向是好的、评测这条坏了"。tf 与 gen 的差把它一刀切开，
而 `gen` vs `gen_nocache` 再把"坏在哪"切一刀 —— 这三个数一次跑完。

⚠️ 用的是 RLDS 里的**原始动作**（`check_replay.load_episodes` 那条注释：不能走
   RLDSDataset，那条链路已经按 BOUNDS_Q99 归一化过）。`predict_action` 吐的也是
   反归一化后的原始动作，两边同一坐标系可以直接减；tf 那一路要自己归一化再
   token 化，用的是同一份 q01/q99 与 mask。
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库根目录>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402

try:
    from experiments.robot.libero.libero_utils import resize_image  # noqa: E402
except ImportError:                                    # 官方改过函数名就走这条
    def resize_image(img, size):
        import tensorflow as tf
        x = tf.io.decode_image(tf.image.encode_jpeg(tf.convert_to_tensor(img)),
                               expand_animations=False, dtype=tf.uint8)
        x = tf.image.resize(x, size, method="lanczos3", antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(x), 0, 255), tf.uint8).numpy()

from common.imgproc import center_crop_resize  # noqa: E402
from common.runs import resolve_adapter  # noqa: E402
from data.kframe import strided_chunk_indices  # noqa: E402
from pooling.wire import (WireConfig, assert_rope_active, set_batch,  # noqa: E402
                          wire)

DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "grip"]
POOLED = ("G2", "G3", "G4", "M2")
VARIANTS = ("tf", "gen", "gen_nocache", "nocrop", "shuffled")


def load_episodes(data_root: str, name: str, n: int):
    """与 check_replay.load_episodes 同一份：直接读 tfds，绕开 openvla 的归一化。"""
    import tensorflow_datasets as tfds

    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    ds = b.as_dataset(split="train", shuffle_files=False)
    out = []
    for ep in ds.take(n):
        steps = list(ep["steps"].as_numpy_iterator())
        out.append({
            "lang": steps[0]["language_instruction"].decode().strip().lower(),
            "images": np.stack([s["observation"]["image"] for s in steps]),
            "actions": np.stack([s["action"] for s in steps]).astype(np.float64),
        })
    return out


def norm_action(a, q01, q99, mask):
    """openvla 的 BOUNDS_Q99：mask 为 False 的维（LIBERO 的夹爪）原样不动。"""
    z = np.clip(2 * (a - q01) / (q99 - q01 + 1e-8) - 1, -1, 1)
    return np.where(mask, z, a)


def denorm_action(z, q01, q99, mask):
    return np.where(mask, 0.5 * (z + 1) * (q99 - q01) + q01, z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="G2")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--n_t", type=int, default=2)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--run_root", default="runs", help="--adapter 找不到时列清单用")
    ap.add_argument("--vla_path", default="openvla/openvla-7b")
    ap.add_argument("--data_root_dir", default="datasets/modified_libero_rlds")
    ap.add_argument("--dataset_name", default="libero_10_no_noops")
    ap.add_argument("--unnorm_key", default=None)
    ap.add_argument("--stats_json", default=None)
    ap.add_argument("--n_episodes", type=int, default=3)
    ap.add_argument("--n_steps", type=int, default=8, help="每条 episode 采样几个时刻")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    unnorm = args.unnorm_key or args.dataset_name
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    assert set(variants) <= set(VARIANTS), f"--variants 只能取 {VARIANTS}"
    dev = "cuda"
    # ⚠️ 先把 checkpoint 路径查清楚再去加载 7B —— 否则要等三分钟才看到一句
    #    与路径无关的 HFValidationError（见 common/runs 的说明）。
    adapter = resolve_adapter(args.adapter, args.run_root) if args.adapter else None

    eps = load_episodes(args.data_root_dir, args.dataset_name, args.n_episodes)
    print(f"读到 {len(eps)} 条 episode，长度 {[len(e['actions']) for e in eps]}")

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        print(f"已加载 adapter: {adapter}")

    if unnorm not in model.norm_stats:
        sj = Path(args.stats_json) if args.stats_json else (
            adapter.parents[1] / "dataset_statistics.json" if adapter else None)
        if sj is None or not sj.exists():
            raise SystemExit(f"找不到动作统计量（找过 {sj}），用 --stats_json 指过去")
        d = json.loads(sj.read_text())
        model.norm_stats[unnorm] = d.get(unnorm, d)
        print(f"已注入动作统计量: {sj}")
    st = model.norm_stats[unnorm]["action"]
    q01, q99, mean = (np.asarray(st[k]) for k in ("q01", "q99", "mean"))
    amask = np.asarray(st.get("mask", [True] * 6 + [False]), dtype=bool)
    print("动作统计量 q01/q99: " + "  ".join(
        f"{n}[{a:+.2f},{b:+.2f}]" for n, a, b in zip(DIMS, q01, q99)))
    print(f"归一化 mask（False = 该维不归一化）: {amask.tolist()}")

    atok = ActionTokenizer(processor.tokenizer)
    state = wire(model, WireConfig(arm=args.arm, K=args.K, budget=args.budget,
                                   n_t=args.n_t))
    n_vis = args.budget if args.arm in POOLED else args.K * 256
    model.eval()

    rng = np.random.default_rng(args.seed)
    pred = {v: [] for v in variants}
    tf_hits = tf_tot = 0
    gt_all, checked = [], False

    def pixels(frames, crop=True):
        if crop:
            frames = [center_crop_resize(f) for f in frames]
        return torch.cat([processor.image_processor.apply_transform(
            Image.fromarray(f)) for f in frames],
            dim=0).unsqueeze(0).to(torch.bfloat16).to(dev)

    for ei, ep in enumerate(eps):
        T = len(ep["actions"])
        # 只在有真实历史的时刻采样：窗口跨 (K-1)*stride 帧，开头全是重复第 0 帧，
        # 那些时刻测的是"没有历史时模型怎么办"，不是我们要问的问题。
        lo = min((args.K - 1) * args.stride, T - 1)
        ts = np.linspace(lo, T - 1, args.n_steps).round().astype(int)
        cache = {}

        def frame(j):
            if j not in cache:
                cache[j] = resize_image(ep["images"][j], (224, 224))
            return cache[j]

        prompt = f"In: What action should the robot take to {ep['lang']}?\nOut:"
        ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)

        for t in ts:
            idx = strided_chunk_indices(T, args.K, args.stride)[t]
            pad = torch.from_numpy(idx >= 0).unsqueeze(0).to(dev)
            base = [frame(int(max(j, 0))) for j in idx]
            act = ep["actions"][t]
            gt_all.append(act)

            for v in variants:
                fr = ([frame(int(rng.integers(0, T))) for _ in idx]
                      if v == "shuffled" else base)
                px = pixels(fr, crop=(v != "nocrop"))

                if v == "tf":
                    # ---- 与训练逐位相同的那条路径：teacher forcing 单次前向 ----
                    pb = PurePromptBuilder("openvla")
                    pb.add_turn("human",
                                f"What action should the robot take to {ep['lang']}?")
                    pb.add_turn("gpt", atok(norm_action(act, q01, q99, amask)))
                    tid = processor.tokenizer(pb.get_prompt(),
                                              add_special_tokens=True).input_ids
                    tid = torch.tensor(tid).unsqueeze(0).to(dev)
                    lab = tid.clone()
                    lab[:, : -(len(act) + 1)] = -100
                    set_batch(state, depth=None, frame_pad_mask=pad)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        out = model(input_ids=tid, attention_mask=torch.ones_like(tid),
                                    pixel_values=px, labels=lab)
                    p = out.logits[:, n_vis:-1].argmax(dim=2)[0]
                    g = lab[0, 1:]
                    m = g > atok.action_token_begin_idx
                    tf_hits += int(((p == g) & m).sum())
                    tf_tot += int(m.sum())
                    z = atok.decode_token_ids_to_actions(
                        p[m].cpu().numpy())          # 归一化空间
                    pred[v].append(denorm_action(np.asarray(z).reshape(-1)[:7],
                                                 q01, q99, amask))
                else:
                    set_batch(state, depth=None, frame_pad_mask=pad)
                    with torch.no_grad():
                        a = model.predict_action(
                            input_ids=ids, pixel_values=px, unnorm_key=unnorm,
                            do_sample=False,
                            **({"use_cache": False} if v == "gen_nocache" else {}))
                    pred[v].append(np.asarray(a, dtype=np.float64).reshape(-1)[:7])

                if not checked:
                    assert_rope_active(state)
                    print(f"  ✓ 4D RoPE 已生效（{state.rope_calls} 次）")
                    checked = True
        print(f"  episode {ei} ({ep['lang'][:38]}…) 采样时刻 {ts.tolist()}")

    gt = np.stack(gt_all)
    scale = np.maximum((q99 - q01) / 2.0, 1e-6)
    base_mae = np.abs(gt - mean).mean(axis=0)

    print(f"\n{'':>12}" + "".join(f"{d:>9}" for d in DIMS) + f"{'总体':>10}")
    print(f"{'基线 MAE':>12}" + "".join(f"{v:>9.3f}" for v in base_mae)
          + f"{(base_mae / scale).mean():>10.3f}")
    rel = {}
    for v in variants:
        p = np.stack(pred[v])
        mae = np.abs(p - gt).mean(axis=0)
        rel[v] = float((mae / np.maximum(base_mae, 1e-6)).mean())
        grip = float((np.sign(p[:, 6]) == np.sign(gt[:, 6])).mean())
        print(f"{v:>12}" + "".join(f"{m:>9.3f}" for m in mae)
              + f"{rel[v]:>10.3f}   夹爪同号 {grip:.1%}")
    print("（末列 = MAE / 基线 MAE 的各维平均。<1 才叫学到了东西，"
          "≈1 等于常数预测）")
    if "tf" in variants:
        print(f"\nteacher forcing 的动作 token 准确率: {tf_hits / max(tf_tot, 1):.3f}"
              f"（{tf_hits}/{tf_tot}）—— **直接和训练日志里的 acc 比**")

    print("\n前 3 个时刻，逐维对照：")
    for i in range(min(3, len(gt))):
        print(f"  {'真':<12}" + "".join(f"{x:>8.3f}" for x in gt[i]))
        for v in variants:
            print(f"  {v:<12}" + "".join(f"{x:>8.3f}" for x in pred[v][i]))

    print("\n判读：")
    tf_acc = tf_hits / max(tf_tot, 1)
    has = lambda v: v in rel                                    # noqa: E731
    if "tf" in variants and tf_acc < 0.15:
        print(f"  ❌ 连 teacher forcing 都只有 {tf_acc:.3f} —— **不是生成路径的问题**，"
              "训练时那条前向在这里就复现不出来。查 adapter 是否真的合进去了、"
              "接线参数（arm/K/budget/n_t）与训练是否一致、n_vis 切片。")
    elif "tf" in variants and has("gen") and rel["gen"] > 0.9 > rel["tf"]:
        print(f"  ⭐ teacher forcing 正常（acc {tf_acc:.3f}，MAE 比 {rel['tf']:.2f}）"
              f"而自回归生成失效（{rel['gen']:.2f}）—— **坏在生成这条路径上**。")
        if has("gen_nocache"):
            if rel["gen_nocache"] < rel["gen"] * 0.8:
                print(f"     关掉 KV cache 就好了（{rel['gen_nocache']:.2f}）—— "
                      "定位到 wire._Rope 里按 position_ids 取 cos/sin 那一段。")
            else:
                print(f"     关掉 KV cache 也没好转（{rel['gen_nocache']:.2f}）—— "
                      "不是 cache，查 prompt/29871/生成时的 pixel_values 通路。")
    elif has("gen") and has("shuffled") and abs(rel["gen"] - rel["shuffled"]) < 0.05:
        print("  ❌ 打乱画面几乎不改变预测 —— **画面没有进入决策**。"
              "查视觉挂点与 pixel_values 的通道排布。")
    else:
        print(f"  ✅ 开环预测优于基线（gen {rel.get('gen', float('nan')):.2f}），"
              "策略是学到了的 —— 0% 出在**闭环**那一侧。")
    if has("gen") and has("nocrop") and rel["nocrop"] > rel["gen"] * 1.15:
        print(f"  ⚠️ 不裁剪时误差高 {rel['nocrop'] / rel['gen'] - 1:.0%}，"
              "评测务必带 --center_crop True。")


if __name__ == "__main__":
    main()
