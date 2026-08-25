"""
LoRA 目标层的选取 —— 训练与显存实测**必须共用**这一份。

阶段 C 的审查发现: `probe_vram.py` 当时用官方的 `target_modules="all-linear"`
（视觉主干挂 LoRA），而 `finetune_single.py` 冻结视觉主干。两者实测差 1.4 GB
（K=1 时 18.2 vs 16.8），于是那张显存决策表描述的不是真正要跑的配置。
把逻辑收到这里，就是为了不再出现"测的配置和训的配置不一致"。
"""

import torch


def lora_targets(model, include_vision: bool) -> list[str]:
    """
    收集要挂 LoRA 的线性层**全名**。

    peft 0.11.1 的匹配规则是 `key in target_modules or key.endswith("."+t)`，
    所以传全名是精确的；传后缀则会误伤——投影器的 fc1/fc2 与 ViT block 的
    fc1/fc2 同名，按后缀匹配会把视觉主干一并挂上，与意图正好相反。

    lm_head 按 peft 的 all-linear 惯例排除（它是输出嵌入层）。
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
