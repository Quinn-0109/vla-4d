#!/usr/bin/env python
"""
开环动作诊断 —— **成功率为 0 时先跑这个，不要直接重跑 6 小时的评测。**

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/diag_openloop.py --arm G2 \
        --adapter runs/G2+.../adapter/step30000 \
        --data_root_dir ~/autodl-tmp/datasets/modified_libero_rlds

做的事：**不开仿真器**，直接从 RLDS 训练集里取画面，按评测侧的预处理喂进模型，
把预测动作与数据集里的真动作比。回答的是一个非此即彼的问题：

    · 开环误差远小于"常数预测"基线 → **策略是学到了的**，
      0% 出在闭环那一侧（动作缩放、夹爪约定、初始状态、步进循环）。
    · 开环误差与基线相当          → **模型这一侧就是坏的**，
      再怎么调 rollout 也没用（预处理、接线、反归一化统计量）。

三个变体一起测，差值本身就是判据：

    crop      中心裁 0.9，与训练的 image_aug 对齐（评测应当用的那个）
    nocrop    整张 224 —— 与之前那次 0/20 的评测同款
    shuffled  画面换成同 episode 里的随机时刻。⚠️ 这一条是**接线的哑弹检测**：
              它若与 crop 打平，说明画面根本没进到决策里（池化挂点错、
              视觉 token 被覆盖……），而这类失效不会报任何错。

⚠️ 用的是 RLDS 里的**原始动作**（`load_episodes` 那条注释：不能走 RLDSDataset，
   那条链路已经按 BOUNDS_Q99 归一化过）。`predict_action` 吐的也是反归一化后的
   原始动作，两边同一坐标系，可以直接减。
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    unnorm = args.unnorm_key or args.dataset_name
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
    q01, q99 = np.asarray(st["q01"]), np.asarray(st["q99"])
    mean = np.asarray(st["mean"])
    print("动作统计量 q01/q99: " + "  ".join(
        f"{n}[{a:+.2f},{b:+.2f}]" for n, a, b in zip(DIMS, q01, q99)))

    state = wire(model, WireConfig(arm=args.arm, K=args.K, budget=args.budget,
                                   n_t=args.n_t))
    model.eval()

    rng = np.random.default_rng(args.seed)
    variants = ("crop", "nocrop", "shuffled")
    pred = {v: [] for v in variants}
    gt_all, checked = [], False

    for ei, ep in enumerate(eps):
        T = len(ep["actions"])
        # 只在有真实历史的时刻采样：窗口跨 (K-1)*stride 帧，开头全是重复第 0 帧，
        # 那些时刻测的是"没有历史时模型怎么办"，不是我们要问的问题。
        lo = min((args.K - 1) * args.stride, T - 1)
        ts = np.linspace(lo, T - 1, args.n_steps).round().astype(int)
        img224 = {}                                     # 惰性 resize，省时间

        def frame(j):
            if j not in img224:
                img224[j] = resize_image(ep["images"][j], (224, 224))
            return img224[j]

        prompt = f"In: What action should the robot take to {ep['lang']}?\nOut:"
        ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)

        for t in ts:
            idx = strided_chunk_indices(T, args.K, args.stride)[t]
            pad = idx >= 0
            base = [frame(int(max(j, 0))) for j in idx]
            shuf = [frame(int(rng.integers(0, T))) for _ in idx]
            gt_all.append(ep["actions"][t])

            for v in variants:
                fr = shuf if v == "shuffled" else base
                if v != "nocrop":
                    fr = [center_crop_resize(f) for f in fr]
                px = torch.cat([processor.image_processor.apply_transform(
                    Image.fromarray(f)) for f in fr],
                    dim=0).unsqueeze(0).to(torch.bfloat16).to(dev)
                set_batch(state, depth=None,
                          frame_pad_mask=torch.from_numpy(pad).unsqueeze(0).to(dev))
                with torch.no_grad():
                    a = model.predict_action(input_ids=ids, pixel_values=px,
                                             unnorm_key=unnorm, do_sample=False)
                pred[v].append(np.asarray(a, dtype=np.float64))
                if not checked:
                    assert_rope_active(state)
                    print(f"  ✓ 4D RoPE 已生效（{state.rope_calls} 次）")
                    checked = True
        print(f"  episode {ei} ({ep['lang'][:38]}…) 采样时刻 {ts.tolist()}")

    gt = np.stack(gt_all)
    scale = np.maximum((q99 - q01) / 2.0, 1e-6)         # 每维的量纲
    base_mae = np.abs(gt - mean).mean(axis=0)           # 常数预测（数据集均值）基线

    print(f"\n{'':>10}" + "".join(f"{d:>9}" for d in DIMS) + f"{'总体':>10}")
    print(f"{'基线 MAE':>10}" + "".join(f"{v:>9.3f}" for v in base_mae)
          + f"{(base_mae / scale).mean():>10.3f}")
    rel = {}
    for v in variants:
        p = np.stack(pred[v])
        mae = np.abs(p - gt).mean(axis=0)
        rel[v] = float((mae / np.maximum(base_mae, 1e-6)).mean())
        grip = float((np.sign(p[:, 6]) == np.sign(gt[:, 6])).mean())
        print(f"{v:>10}" + "".join(f"{m:>9.3f}" for m in mae)
              + f"{rel[v]:>10.3f}   夹爪同号 {grip:.1%}")
    print("（末列 = MAE / 基线 MAE 的各维平均。<1 才叫学到了东西，"
          "≈1 等于常数预测，>1 比不预测还差）")

    print("\n前 3 个时刻，逐维对照（crop 变体）：")
    for i in range(min(3, len(gt))):
        print(f"  真  " + "".join(f"{x:>8.3f}" for x in gt[i]))
        print(f"  预测" + "".join(f"{x:>8.3f}" for x in pred['crop'][i]))

    print("\n判读：")
    if rel["crop"] > 0.9 and rel["shuffled"] > 0.9:
        print("  ❌ 连训练分布上的开环动作都预测不出来 —— **问题在模型这一侧**。"
              "查接线（arm/K/budget 与训练是否一致）、adapter 路径、"
              "反归一化统计量。闭环那边先别动。")
    elif abs(rel["crop"] - rel["shuffled"]) < 0.05:
        print("  ❌ 打乱画面几乎不改变预测 —— **画面没有进入决策**。"
              "查视觉挂点（pooling.wire._patch_vision/_patch_projector）"
              "与 pixel_values 的通道排布。")
    else:
        print(f"  ✅ 开环预测显著优于基线（crop {rel['crop']:.2f}），"
              "策略是学到了的 —— 0% 出在**闭环**：动作缩放/夹爪约定/"
              "初始状态/步进循环。")
        if rel["nocrop"] > rel["crop"] * 1.15:
            print(f"  ⚠️ 不裁剪时误差高 {rel['nocrop'] / rel['crop'] - 1:.0%} —— "
                  "之前那次 0/20 正是没开中心裁，先带 --center_crop True 重跑冒烟。")


if __name__ == "__main__":
    main()
