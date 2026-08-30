#!/usr/bin/env python
"""
深度缓存的**键**对拍 —— G4/M2 的硬闸，**不用 GPU**（不加载模型、不渲染）。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/check_hash_key.py --data_root ~/autodl-tmp/datasets/modified_libero_rlds

问一个问题：**离线算的帧指纹，和训练时分窗算出来的，是不是同一个数？**

深度缓存按帧的 JPEG 字节指纹索引（`src/data/depth_cache.py`）。两边都调同一个
哈希函数，但**输入是不是同一份字节，只有真跑一遍才知道**：

  离线（`build_subset`）：`tfds` + `SkipDecoding` 直接拿存储字节
  训练（`data/kframe`）：  octo 的轨迹变换阶段的 `observation/image_primary`

中间隔着 octo 的 `make_dataset_from_rlds`（改键名、standardize_fn、
可能的重编码）。任何一处动过字节，指纹就对不上 —— 而**对不上不会报错**，
只会让 G4 查不到深度（`DepthCache.lookup` 会抛，但那要等到训练开跑）。
这个脚本把它提前到一分钟。

三项检查：
  1. 训练流里每个 `img_hash` 都能在离线指纹全集里找到  ← **命中率必须 100%**
  2. 同一个窗口的 K 帧指纹**互不相同**（否则是按 episode 而非按帧编的键）
  3. 补帧位置的指纹与最早那一帧相同（clamp 到第 0 帧，与 pad_mask 一致）
"""

import argparse
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
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # 明确不碰 GPU


def offline_hashes(data_root: str, name: str, limit: int = 0) -> set:
    """离线那一路：SkipDecoding 拿存储字节，逐帧指纹。不解码、不渲染，很快。"""
    import tensorflow_datasets as tfds

    from data.depth_cache import hash_bytes

    b = tfds.builder(name, data_dir=str(Path(data_root).expanduser()))
    ds = b.as_dataset(split="train", shuffle_files=False, decoders={
        "steps": {"observation": {"image": tfds.decode.SkipDecoding()}}})
    out, n_ep = set(), 0
    for ep in ds:
        raw = [s["observation"]["image"] for s in ep["steps"]]
        out.update(int(h) for h in hash_bytes(raw).numpy())
        n_ep += 1
        if n_ep % 50 == 0:
            print(f"    …{n_ep} 条 episode，{len(out)} 个指纹", flush=True)
        if limit and n_ep >= limit:
            break
    print(f"  离线指纹全集：{n_ep} 条 episode，{len(out)} 个不重复指纹")
    return out


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=str(
        Path("~/autodl-tmp/datasets/modified_libero_rlds").expanduser()))
    ap.add_argument("--dataset_name", default="libero_10_no_noops")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--n_batches", type=int, default=20)
    ap.add_argument("--limit_episodes", type=int, default=0,
                    help="离线那一路只扫前 N 条。⚠️ 仅用于调脚本本身："
                         "训练侧 shuffle files，扫一部分必然低命中，带了它不出判据")
    args = ap.parse_args()
    name = args.dataset_name

    print("① 离线指纹全集")
    ref = offline_hashes(args.data_root, name, args.limit_episodes)

    print("\n② 训练流里的 img_hash")
    from torch.utils.data import DataLoader
    from transformers import AutoProcessor

    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.datasets import RLDSDataset

    from data.kframe import (KFrameBatchTransform, PaddedCollatorKFrame,
                             patch_strided_chunking)

    patch_strided_chunking(args.stride, args.K)
    proc = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    ds = RLDSDataset(
        Path(args.data_root).expanduser(), name,
        KFrameBatchTransform(ActionTokenizer(proc.tokenizer), proc.tokenizer,
                             image_transform=proc.image_processor.apply_transform,
                             prompt_builder_fn=PurePromptBuilder),
        resize_resolution=(224, 224), shuffle_buffer_size=1000,
        image_aug=True)          # ⚠️ 开增广：指纹必须在增广之前算，这里正是要验它
    loader = DataLoader(ds, batch_size=4, collate_fn=PaddedCollatorKFrame(
        proc.tokenizer.model_max_length, proc.tokenizer.pad_token_id))

    hit = tot = 0
    uniq_ok = pad_same = pad_n = 0
    for i, b in enumerate(loader):
        if i >= args.n_batches:
            break
        if "img_hash" not in b:
            raise SystemExit(
                "batch 里没有 img_hash —— 分窗时没打上指纹。"
                "查 data/kframe.strided_chunk_act_obs 里的 ⓪ 那一段，"
                "以及 image_primary 在那时是不是还未解码。")
        h = b["img_hash"].numpy().astype(np.uint64)      # (B, K)
        m = b["frame_pad_mask"].numpy()                  # (B, K)
        for row, mrow in zip(h, m):
            tot += len(row)
            hit += sum(int(x) in ref for x in row)
            real = row[mrow]
            uniq_ok += len(np.unique(real)) == len(real)
            if (~mrow).sum() > 1:
                # 补帧位全部 clamp 到**轨迹第 0 帧**（不是窗口里最早的真实帧），
                # 所以它们彼此相同 —— 但与 real[0] 无关，除非 t 本身就在开头。
                pad_n += 1
                pad_same += bool((row[~mrow] == row[~mrow][0]).all())
        if (i + 1) % 5 == 0:
            print(f"    …{i + 1}/{args.n_batches} 批  命中 {hit}/{tot}", flush=True)

    n_win = tot // args.K
    print(f"\n=== 结果（{n_win} 个窗口，{tot} 帧）===")
    print(f"  1. 命中率            {hit / max(tot, 1):>7.2%}  ({hit}/{tot})")
    print(f"  2. K 帧指纹互不相同  {uniq_ok / max(n_win, 1):>7.2%}")
    print(f"  3. 补帧位彼此相同    {pad_same / max(pad_n, 1):>7.2%}  "
          f"（{pad_n} 个含 ≥2 个补帧的窗口）")

    if args.limit_episodes:
        raise SystemExit(
            f"\n⚠️ 带了 --limit_episodes {args.limit_episodes}，离线只扫了部分 episode，"
            "而训练侧 shuffle files —— **命中率这个数没有意义，不构成判据**。\n"
            "   去掉它重跑（只读字节不解码，全集约两分钟）。")

    if hit != tot:
        raise SystemExit(
            f"\n❌ 有 {tot - hit} 帧的指纹不在离线全集里 —— **键对不上**。\n"
            "   离线拿的是 tfds 存储字节，训练拿的是轨迹变换阶段的 image_primary；\n"
            "   中间 octo 的 make_dataset_from_rlds 若动过字节（重编码、改分辨率），\n"
            "   指纹就不同。这不会报错，只会让 G4 查不到深度。\n"
            "   → 在轨迹变换阶段把 image_primary 的字节 dump 出来，和 tfds 的比对。"
            + ("\n   （--limit_episodes 只扫了部分 episode，先去掉它再看。）"
               if args.limit_episodes else ""))
    if uniq_ok != n_win:
        raise SystemExit("\n❌ 有窗口的真实帧指纹重复 —— 键是按 episode 而非按帧编的，"
                         "深度会取错帧。")
    if pad_n and pad_same != pad_n:
        raise SystemExit("\n❌ 同一窗口的补帧位指纹不一致 —— 它们都该 clamp 到"
                         "轨迹第 0 帧。指纹没跟着 gather 走。")
    print("\n✅ 三项全过：键可用，深度缓存可以按它建。")


if __name__ == "__main__":
    main()
