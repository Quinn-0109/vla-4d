"""
K 帧微调 —— 六组对照共用这一个脚本，差别只在 `--arm`。

    python scripts/finetune_kframe.py --arm G2 --bench_only True   # 先测吞吐
    python scripts/finetune_kframe.py --arm G2 --max_steps 30000

与 `finetune_single.py` 的关系：那一份是 K=1 的基线（G0），保持不动作为对照锚点；
这一份加了 K 帧数据流、坐标池化与 4D RoPE。两者的 LoRA 配置、优化器、
断点续训、checkpoint 分步保存**完全一致**——不一致就没法比。

**有效批固定为 16**（`docs/06` §4.3）。微批按显存实测取：
K=1 用 16×1，K=8 各臂用 2×8。微批不同、有效批相同，数值上等价；
**但有效批一旦跟着显存漂移，G0 vs G1 的比较就作废了。**

⚠️ **G4 / M2 现在还跑不了**：它们要按 (task, episode, timestep) 取训练集的
patch 级深度，而那份缓存的前提是"回放能复现 RLDS 的画面"（`docs/06` §4.3
的回放保真度验证）还没做。脚本会直接拒绝，不会拿假深度凑合。
G2 / G3 不需要深度，现在就能跑，而 **G2 vs G0 正是前提检验**，本来就排在最前。
"""

import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import torch
import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader

_OPENVLA_ROOT = os.environ.get("OPENVLA_ROOT")
if _OPENVLA_ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库根目录>")
sys.path.insert(0, str(Path(_OPENVLA_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets import RLDSDataset  # noqa: E402
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # noqa: E402

from common.lora import lora_targets  # noqa: E402
from data.kframe import (KFrameBatchTransform, PaddedCollatorKFrame,  # noqa: E402
                         patch_strided_chunking)
from pooling.wire import WireConfig, assert_rope_active, set_batch, wire  # noqa: E402

# 微批 × 累积，有效批恒为 16（docs/06 §4.3 的显存实测）
#
# K=8 的微批实测（G2，libero_10）：
#
#     micro=2  2.7 s/步  16.3 GB  30k 步 22.3 h
#     micro=4  2.4 s/步  16.9 GB  30k 步 19.7 h   ← 取这个
#     micro=8  2.4 s/步  17.9 GB  30k 步 19.9 h   （多花 1 GB，一点没快）
#
# 2→4 省 11%，4→8 归零 —— **瓶颈在视觉主干不在 LLM**：每个梯度步要跑
# 16 样本 × 8 帧 = 128 次 DINOv2+SigLIP 前向，这个数与微批怎么分组无关，
# 微批加大只省一点调度开销，到 4 就饱和。
#
# 这和显存那轮是同一件事的两面：**压缩既不省显存也不省时间**，省的是 LLM
# 那一段，而成本大头在视觉主干。池化必须靠效果说话。
#
# ⚠️ **有效批必须恒为 16**，所以只让改微批，累积由它自动配平。
EFF_BATCH = 16
MICRO = {1: (16, 1), 8: (4, 4)}      # K=8 的 4×4 是实测出来的，见下


@dataclass
class Config:
    # fmt: off
    arm: str = "G2"
    K: int = 8
    stride: int = 16                           # docs/05 §8.7 定稿
    budget: int = 256                          # docs/06 §3.0：N=256，所有组同预算
    n_t: int = 2

    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/modified_libero_rlds")
    dataset_name: str = "libero_10_no_noops"   # 主战场是 Long（docs/06 §4.2）
    run_root_dir: Path = Path("runs")

    max_steps: int = 30_000                    # docs/05 §8.2 实测定
    learning_rate: float = 5e-4
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_vision: bool = False
    grad_checkpoint: bool = True

    save_steps: int = 2500
    log_steps: int = 10
    resume_from: Optional[str] = None
    bench_only: bool = False
    micro: int = 0                             # 0 = 用 MICRO 表；>0 覆盖，累积自动配平
    run_id_note: Optional[str] = None
    # fmt: on


@draccus.wrap()
def main(cfg: Config) -> None:
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"

    if cfg.arm in ("G4", "M2"):
        raise SystemExit(
            f"{cfg.arm} 需要**训练集**的 patch 级深度，而那份缓存还没有。\n"
            "前置：先做回放保真度验证（回放 RGB vs RLDS 训练数据 RGB，docs/06 §4.3），"
            "对不上就说明回放发散，缓存出来的深度是另一个场景的。\n"
            "现在能跑的是 G2 / G3（用图像网格坐标，不需要深度），"
            "而 G2 vs G0 正是前提检验，本来就排在最前。")

    micro, accum = MICRO[cfg.K]
    if cfg.micro:
        # ⚠️ **有效批必须恒为 16**（docs/06 §4.3）。只让改微批，累积自动配平；
        #    除不尽就直接拒绝 —— 有效批一旦跟着显存漂移，跨臂比较就作废了，
        #    而那种漂移不会有任何提示。
        if EFF_BATCH % cfg.micro:
            raise SystemExit(
                f"--micro {cfg.micro} 除不尽有效批 {EFF_BATCH}。"
                f"可选：{[m for m in (1, 2, 4, 8, 16) if EFF_BATCH % m == 0]}")
        micro, accum = cfg.micro, EFF_BATCH // cfg.micro
    exp_id = (f"{cfg.arm}+{cfg.dataset_name}+K{cfg.K}s{cfg.stride}"
              f"+N{cfg.budget}+nt{cfg.n_t}+b{micro}x{accum}"
              f"+lr{cfg.learning_rate}+lora-r{cfg.lora_rank}"
              f"{'' if cfg.lora_vision else '+frozen-vision'}"
              f"{'+aug' if cfg.image_aug else ''}")
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    run_dir = Path(cfg.run_root_dir) / exp_id
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"运行目录: {run_dir}\n有效批 = {micro} × {accum} = {micro * accum}")

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)
    image_sizes = tuple(vla.config.image_sizes)

    targets = lora_targets(vla, cfg.lora_vision)
    print(f"LoRA 目标层: {len(targets)} 个"
          f"（视觉主干{'含' if cfg.lora_vision else '**不含**'}）")
    vla = get_peft_model(vla, LoraConfig(
        r=cfg.lora_rank, lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout, target_modules=targets,
        init_lora_weights="gaussian"))
    vla.print_trainable_parameters()
    if cfg.grad_checkpoint:
        vla.gradient_checkpointing_enable()
        vla.enable_input_require_grads()

    # ⚠️ 接线必须在 peft 包装**之后**：get_peft_model 会代理属性，
    #    包装前挂上去的 forward 会被代理层绕过（挂了等于没挂，且不报错）。
    wcfg = WireConfig(arm=cfg.arm, K=cfg.K, budget=cfg.budget, n_t=cfg.n_t)
    state = wire(vla.base_model.model, wcfg)
    print(f"已接线: arm={cfg.arm}  K={cfg.K}  N={cfg.budget}  n_t={cfg.n_t}")

    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad],
                      lr=cfg.learning_rate)
    start_step = 0
    if cfg.resume_from:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        src = Path(cfg.resume_from)
        set_peft_model_state_dict(vla, load_file(src / "adapter_model.safetensors"))
        ck = torch.load(src / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"从 {src} 续训，已完成 {start_step} 步")

    # ⚠️ 数据集路径先自己查一遍。tfds 找不到时会吐三百行的全球数据集清单
    #    （abstract_reasoning、ag_news…），真正有用的那两行被埋在最后，
    #    而问题通常只是路径写错或那个 suite 没下。
    root = Path(cfg.data_root_dir).expanduser()
    ds_dir = root / cfg.dataset_name
    if not ds_dir.is_dir() or not any(ds_dir.glob("*/*.tfrecord*")):
        have = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
        raise SystemExit(
            f"找不到数据集 {cfg.dataset_name}\n"
            f"  --data_root_dir 解析为: {root}"
            f"{'（这个目录不存在）' if not root.is_dir() else ''}\n"
            f"  该目录下已有: {have or '（空）'}\n\n"
            f"用绝对路径指过去，例如：\n"
            f"  --data_root_dir ~/autodl-tmp/datasets/modified_libero_rlds\n"
            f"没下过的话：bash scripts/prepare_finetune.sh data")

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ⚠️ 必须在构造 RLDSDataset **之前**打补丁（见 data.kframe.patch_strided_chunking）
    patch_strided_chunking(cfg.stride)
    dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name,
        KFrameBatchTransform(action_tokenizer, processor.tokenizer,
                             image_transform=processor.image_processor.apply_transform,
                             prompt_builder_fn=PurePromptBuilder),
        resize_resolution=image_sizes,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    save_dataset_statistics(dataset.dataset_statistics, run_dir)
    loader = DataLoader(
        dataset, batch_size=micro, sampler=None,
        collate_fn=PaddedCollatorKFrame(processor.tokenizer.model_max_length,
                                        processor.tokenizer.pad_token_id),
        num_workers=0)                       # RLDS 自带并行，官方要求为 0

    metrics_path = run_dir / "metrics.jsonl"
    losses, accs = deque(maxlen=accum), deque(maxlen=accum)
    t0 = time.time()
    # 视觉块的长度：池化臂是 budget，G1 是 K*256。动作准确率要从这之后切，
    # 切错了 acc 全是噪声 —— 而 loss 一切正常，是个纯静默的指标错。
    n_vis = cfg.budget if cfg.arm in ("G2", "G3", "G4", "M2") else cfg.K * 256

    def save(step: int) -> None:
        d = adapter_dir / f"step{step}"
        vla.save_pretrained(d)
        torch.save({"optimizer": optimizer.state_dict(), "step": step},
                   d / "trainer_state.pt")
        processor.save_pretrained(run_dir)
        print(f"\n[step {step}] 已存 -> {d}")

    vla.train()
    optimizer.zero_grad()
    checked_rope = False
    with tqdm.tqdm(total=cfg.max_steps, initial=start_step) as bar:
        for micro_idx, batch in enumerate(loader):
            set_batch(state, depth=None,
                      frame_pad_mask=batch["frame_pad_mask"].to(dev))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=batch["input_ids"].to(dev),
                          attention_mask=batch["attention_mask"].to(dev),
                          pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                          labels=batch["labels"])
            (out.loss / accum).backward()

            if not checked_rope:
                assert_rope_active(state)     # 挂点错了不会崩，只会全程用 1D RoPE
                print(f"  ✓ 4D RoPE 已生效（{state.rope_calls} 次调用）"
                      f"  序列长 {out.logits.shape[1]}")
                checked_rope = True

            preds = out.logits[:, n_vis:-1].argmax(dim=2)
            gt = batch["labels"][:, 1:].to(preds.device)
            mask = gt > action_tokenizer.action_token_begin_idx
            losses.append(out.loss.item())
            accs.append(((preds == gt) & mask).sum().float().item()
                        / max(mask.sum().item(), 1))

            if (micro_idx + 1) % accum != 0:
                continue
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = start_step + (micro_idx + 1) // accum
            bar.update()

            if step % cfg.log_steps == 0:
                el = time.time() - t0
                sps = (step - start_step) / el
                rec = {"step": step, "arm": cfg.arm,
                       "loss": sum(losses) / len(losses),
                       "action_accuracy": sum(accs) / len(accs),
                       "steps_per_sec": sps, "elapsed_h": el / 3600,
                       "peak_gb": torch.cuda.max_memory_allocated() / 1e9}
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                bar.set_postfix(loss=f"{rec['loss']:.3f}",
                                acc=f"{rec['action_accuracy']:.3f}",
                                eta_h=f"{(cfg.max_steps - step) / sps / 3600:.1f}")

            if cfg.bench_only and step >= cfg.log_steps:
                el = time.time() - t0
                print(f"\n=== 吞吐实测（{cfg.arm}）===")
                print(f"  {step} 个梯度步用时 {el:.0f}s -> {el / step:.1f} s/步")
                print(f"  峰值显存 {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
                for tgt in (5_000, 30_000):
                    h = tgt * el / step / 3600
                    print(f"  跑 {tgt:>6} 步需要 {h:>5.1f} 小时（约 ¥{h * 2:.0f}）")
                print("\n五组主线加起来是这个数的五倍，先看清楚再开长跑。")
                return

            if step > start_step and step % cfg.save_steps == 0:
                save(step)
            if step >= cfg.max_steps:
                break

    save(min(step, cfg.max_steps))
    print(f"完成 -> {adapter_dir}")


if __name__ == "__main__":
    main()
