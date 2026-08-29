#!/usr/bin/env python
"""
数据侧形状核对 —— **不加载模型**，30 秒出结果。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/check_kframe_shapes.py --K 8 --stride 16 \
        --data_root_dir ~/autodl-tmp/datasets/modified_libero_rlds

问一个问题：**RLDS 链路真的给出 K 帧了吗？**

openvla 的 `RLDSDataset.__init__` 把 `traj_transform_kwargs=dict(window_size=1, …)`
写死在构造函数里，外面传不进去。我们只替换了分窗函数（`data.kframe`），
若不把 K 顶进去，`tf.range(0, 1, stride)` 只剩当前帧 —— **K 静默退化成 1**：

    pixel_values (B, 6, H, W) 而不是 (B, 48, H, W)
    序列长不变（256 个 patch 池成 256 个槽），loss 照降，acc 照升

除了拿去做 K 帧评测，没有任何地方会露馅。所以它值一个独立的、廉价的检查。

⚠️ 这个脚本**故意不打补丁地跑一遍**作对照：一边是官方原样，一边是打了补丁，
   两边形状必须不同 —— 否则说明补丁没起作用，而"看起来对"和"真的对"是两回事。
"""

import argparse
import os
import sys
from pathlib import Path

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库根目录>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def build(K, stride, root, name, patched: bool):
    """在**子进程**里建一次数据集，取一个 batch，回报形状。"""
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSDataset

    from data.kframe import (KFrameBatchTransform, PaddedCollatorKFrame,
                             patch_strided_chunking)

    if patched:
        patch_strided_chunking(stride, K)
    proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    ds = RLDSDataset(
        Path(root).expanduser(), name,
        KFrameBatchTransform(ActionTokenizer(proc.tokenizer), proc.tokenizer,
                             image_transform=proc.image_processor.apply_transform,
                             prompt_builder_fn=PurePromptBuilder),
        resize_resolution=(224, 224), shuffle_buffer_size=1000, image_aug=False)
    loader = DataLoader(ds, batch_size=2, collate_fn=PaddedCollatorKFrame(
        proc.tokenizer.model_max_length, proc.tokenizer.pad_token_id))
    b = next(iter(loader))
    return tuple(b["pixel_values"].shape), tuple(b["frame_pad_mask"].shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--data_root_dir", default="datasets/modified_libero_rlds")
    ap.add_argument("--dataset_name", default="libero_10_no_noops")
    ap.add_argument("--no_patch", action="store_true",
                    help="不打补丁跑（对照：应当得到 K=1）")
    args = ap.parse_args()

    pv, fm = build(args.K, args.stride, args.data_root_dir, args.dataset_name,
                   patched=not args.no_patch)
    got = pv[1] // 6
    tag = "未打补丁（对照）" if args.no_patch else "打了补丁"
    print(f"\n=== {tag} ===")
    print(f"  pixel_values   {pv}   → K = {got}")
    print(f"  frame_pad_mask {fm}")
    if args.no_patch:
        print(f"  {'✅' if got == 1 else '⚠️'} 官方原样应当是 K=1"
              f"（window_size 写死在 RLDSDataset 里）")
    elif got == args.K and fm[1] == args.K:
        print(f"  ✅ 拿到 K={args.K} 帧，与训练脚本的约定一致")
    else:
        raise SystemExit(
            f"  ❌ 只拿到 K={got}，要的是 {args.K} —— 分窗补丁没顶进去。\n"
            f"     用这份形状训练不会报任何错，只是训的是单帧。")


if __name__ == "__main__":
    main()
