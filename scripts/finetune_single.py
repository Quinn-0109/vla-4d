"""
单卡 LoRA 微调 —— 官方 vla-scripts/finetune.py 的单卡改写。

    python scripts/finetune_single.py \
        --data_root_dir ~/autodl-tmp/datasets/modified_libero_rlds \
        --dataset_name libero_spatial_no_noops \
        --run_root_dir ./runs --max_steps 200 --bench_only True

阶段 B 第 4 步用它验证硬门槛: K=1 微调能否复现到 ~84%。
跑不到就说明训练配置有问题，后面所有多帧对照都没有地基。

与官方的四处差异，都是有依据的:

1. **去掉 DDP**。官方用 torchrun + DDP，单卡不需要。所有 `vla.module.x`
   改成 `vla.x`，`dist.barrier()` 去掉。
2. **视觉主干不挂 LoRA**。官方用 target_modules="all-linear"，会把 LoRA 挂到
   DINOv2+SigLIP 上。显存实测(docs/06)显示多帧时这笔开销乘 K，而我们改的是
   token 表示与位置编码，不是特征提取器。
   ⚠️ 这是与官方的**实质性偏离**，可能影响能否复现到 84%。因此本脚本保留
   --lora_vision 开关: 若冻结版复现不到，切回 True 再试，以区分"配置问题"
   和"冻结视觉主干的代价"。
3. **梯度累积默认 16、batch 默认 1**。官方是 batch 16 + 累积 1(需 72GB)。
   有效批相同。用 --probe 选出的最优组合覆盖。
4. **默认不合并权重**。官方每次存 checkpoint 都要合并出 15GB 完整模型，
   很慢也很占盘。这里只存 adapter(几百 MB)，合并交给 merge_lora.py 按需做。

⚠️ peft 0.11.1 没有 exclude_modules 参数，所以排除视觉主干只能显式枚举
   目标模块的**全名**。不能用后缀: 投影器的 fc1/fc2 与 ViT block 里的
   fc1/fc2 重名，按后缀匹配会把视觉主干一并挂上，正好与意图相反。
"""

# ⚠️ 不要加 `from __future__ import annotations`:
#    它会把类型注解变成字符串，draccus.wrap() 拿到的就是 "Config" 而非类本身，
#    dataclasses.fields() 随即抛 "must be called with a dataclass type or instance"。
#    本文件用到的 list[str] 等写法在 Python 3.10 上原生支持，不需要该导入。

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
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from prismatic.models.backbones.llm.prompting import PurePromptBuilder  # noqa: E402
from prismatic.util.data_utils import PaddedCollatorForActionPrediction  # noqa: E402
from prismatic.vla.action_tokenizer import ActionTokenizer  # noqa: E402
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset  # noqa: E402
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics  # noqa: E402


@dataclass
class Config:
    # fmt: off
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("runs")

    batch_size: int = 1
    grad_accumulation_steps: int = 16          # 有效批 = batch_size * 此值，官方为 16
    max_steps: int = 100_000                   # 梯度步数(非微批数)
    learning_rate: float = 5e-4                # 官方: 恒定学习率，不做衰减
    image_aug: bool = True
    shuffle_buffer_size: int = 100_000

    lora_rank: int = 32
    lora_dropout: float = 0.0
    lora_vision: bool = False                  # 视觉主干是否挂 LoRA(官方等效于 True)
    grad_checkpoint: bool = True

    save_steps: int = 2500
    log_steps: int = 10
    resume_from: Optional[str] = None          # 指向 adapter/stepN 目录
    bench_only: bool = False                   # 只测吞吐并给出 ETA，不真的训完
    run_id_note: Optional[str] = None
    # fmt: on


def lora_targets(model, include_vision: bool) -> list[str]:
    """
    收集要挂 LoRA 的线性层**全名**。

    peft 0.11.1 的匹配规则是 `key in target_modules or key.endswith("."+t)`，
    所以传全名是精确的；传后缀则会误伤——投影器的 fc1/fc2 和 ViT block 的
    fc1/fc2 同名。lm_head 按 peft 的 all-linear 惯例排除(它是输出嵌入层)。
    """
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        if not include_vision and "vision_backbone" in name:
            continue
        names.append(name)
    return names


@draccus.wrap()
def main(cfg: Config) -> None:
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"
    torch.cuda.empty_cache()

    exp_id = (f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
              f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
              f"+lr{cfg.learning_rate}+lora-r{cfg.lora_rank}"
              f"{'' if cfg.lora_vision else '+frozen-vision'}"
              f"{'+aug' if cfg.image_aug else ''}")
    if cfg.run_id_note:
        exp_id += f"--{cfg.run_id_note}"
    run_dir = Path(cfg.run_root_dir) / exp_id
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"运行目录: {run_dir}")

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(dev)
    image_sizes = tuple(vla.config.image_sizes)     # 包 peft 之前取，之后 config 会被代理
    n_patch = vla.vision_backbone.featurizer.patch_embed.num_patches

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

    # 视觉主干不挂 LoRA 时其中已无可训练参数，autograd 会在它的输出处截断计算图，
    # 激活不再保留——这正是 docs/06 显存实测 ③ 指出的省法。这里确认一遍。
    vis_trainable = sum(p.numel() for n, p in vla.named_parameters()
                        if p.requires_grad and "vision_backbone" in n)
    print(f"视觉主干可训练参数: {vis_trainable:,}"
          + ("（已冻结，其激活不占反传显存）" if vis_trainable == 0 else ""))

    optimizer = AdamW([p for p in vla.parameters() if p.requires_grad], lr=cfg.learning_rate)

    # 断点续训: 6.8 小时的跑在第 5 小时挂掉而没法接着跑，代价太大。
    # RLDS 是无限打乱流，数据位置本就无从精确复原，所以只恢复权重、优化器状态和步数。
    start_step = 0
    if cfg.resume_from:
        src = Path(cfg.resume_from)
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        sd = load_file(src / "adapter_model.safetensors")
        set_peft_model_state_dict(vla, sd)
        ck = torch.load(src / "trainer_state.pt", map_location="cpu")
        optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"]
        print(f"从 {src} 续训，已完成 {start_step} 步")

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name,
        RLDSBatchTransform(action_tokenizer, processor.tokenizer,
                           image_transform=processor.image_processor.apply_transform,
                           prompt_builder_fn=PurePromptBuilder),
        resize_resolution=image_sizes,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    save_dataset_statistics(dataset.dataset_statistics, run_dir)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, sampler=None,
        collate_fn=PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id, padding_side="right"),
        num_workers=0,      # RLDS 自带并行，官方注释明确要求为 0
    )

    metrics_path = run_dir / "metrics.jsonl"
    losses, accs = deque(maxlen=cfg.grad_accumulation_steps), deque(maxlen=cfg.grad_accumulation_steps)
    t0 = time.time()
    micro_per_step = cfg.grad_accumulation_steps

    def save(step: int) -> None:
        """
        每个 step 存到独立目录。官方是反复覆盖同一份，但我们要**评测多个步数的
        checkpoint** 来定位成功率何时到顶——覆盖掉就只剩最后一份，没法回看。
        adapter 只有几百 MB，存几份不心疼。
        """
        d = adapter_dir / f"step{step}"
        vla.save_pretrained(d)
        torch.save({"optimizer": optimizer.state_dict(), "step": step},
                   d / "trainer_state.pt")
        processor.save_pretrained(run_dir)
        print(f"\n[step {step}] 已存 -> {d}")

    vla.train()
    optimizer.zero_grad()
    with tqdm.tqdm(total=cfg.max_steps, initial=start_step) as bar:
        for micro_idx, batch in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=batch["input_ids"].to(dev),
                          attention_mask=batch["attention_mask"].to(dev),
                          pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                          labels=batch["labels"])
            (out.loss / micro_per_step).backward()

            preds = out.logits[:, n_patch:-1].argmax(dim=2)
            gt = batch["labels"][:, 1:].to(preds.device)
            mask = gt > action_tokenizer.action_token_begin_idx
            losses.append(out.loss.item())
            accs.append(((preds == gt) & mask).sum().float().item() / max(mask.sum().item(), 1))

            if (micro_idx + 1) % micro_per_step != 0:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step = start_step + (micro_idx + 1) // micro_per_step
            bar.update()

            if step % cfg.log_steps == 0:
                el = time.time() - t0
                sps = (step - start_step) / el
                rec = {"step": step, "loss": sum(losses) / len(losses),
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
                print(f"\n=== 吞吐实测 ===")
                print(f"  {step} 个梯度步用时 {el:.0f}s -> {el/step:.1f} s/步")
                print(f"  峰值显存 {torch.cuda.max_memory_allocated()/1e9:.1f} GB")
                for tgt in (5_000, 10_000, 20_000):
                    print(f"  跑 {tgt:>6} 步需要 {tgt*el/step/3600:>5.1f} 小时"
                          f"（4090 按 ¥2/h 约 ¥{tgt*el/step/3600*2:.0f}）")
                print("\n先看这个数字再决定 max_steps，别直接开长跑。")
                return

            if step > start_step and step % cfg.save_steps == 0:
                save(step)

            if step >= cfg.max_steps:
                break

    save(min(step, cfg.max_steps))
    print(f"完成 -> {adapter_dir}")


if __name__ == "__main__":
    main()
