#!/usr/bin/env python
"""
K 帧评测 —— 六组对照共用，差别只在 `--arm`。

    python scripts/run_eval_kframe.py --arm G2 \
        --adapter runs/G2+.../adapter/step30000 --num_trials_per_task 50

与 `run_libero_eval_traj.py`（单帧，阶段 0 用的那份）的关系：那份保持不动，
G0 仍然用它评；这份只加 K 帧历史窗口 + 接线。**评测协议其余部分逐字照抄**
（初始状态、num_steps_wait、max_steps、成功判定、n=500），
否则跨阶段的数字没法比。

⚠️ **历史窗口的构造必须与训练完全一致**（`docs/06` §4.1）：
  · 取 `t, t−s, …, t−(K−1)s`，s = stride
  · episode 开头不足时**重复最早可得的那一帧**，并由 `frame_pad_mask` 标出
  · 训练侧是 `data/kframe.py` 的 `strided_chunk_act_obs`，两边差一点，
    测的就是"训练与评测不一致"而不是方法本身
"""

# ⚠️ **不要加 `from __future__ import annotations`。**
#    它把类型注解变成字符串，`draccus.wrap()` 拿到的就是 "Config" 而非类本身，
#    `dataclasses.fields()` 随即抛 "must be called with a dataclass type or instance"。
#    `finetune_single.py` 的文件头记过这个坑，我写这份时照样踩了 —— 所以再记一次。

import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm
from PIL import Image

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库根目录>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action, get_libero_env, get_libero_image)
from experiments.robot.robot_utils import (invert_gripper_action,  # noqa: E402
                                           normalize_gripper_action)

from pooling.wire import (WireConfig, assert_rope_active, set_batch,  # noqa: E402
                          wire)

MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
             "libero_goal": 300, "libero_10": 520, "libero_90": 400}


@dataclass
class Config:
    # fmt: off
    arm: str = "G2"
    K: int = 8
    stride: int = 16
    budget: int = 256
    n_t: int = 2

    adapter: Optional[str] = None              # LoRA adapter 目录；None = 用底座
    vla_path: str = "openvla/openvla-7b"
    task_suite_name: str = "libero_10"
    num_trials_per_task: int = 50              # ⚠️ 主结论一律满量（docs/06 §3.0）
    num_steps_wait: int = 10
    unnorm_key: Optional[str] = None           # 默认取 task_suite_name + "_no_noops"
    stats_json: Optional[str] = None           # 默认找 <adapter>/../../dataset_statistics.json
    seed: int = 7                              # 沿用阶段 0
    local_log_dir: str = "results/logs"
    run_note: str = ""
    # fmt: on


def build_window(hist: deque, k: int, stride: int):
    """
    从帧历史里取 K 帧：`t, t−s, …, t−(K−1)s`，不足则重复最早那一帧。
    返回 (帧列表[旧→新], pad_mask[K] bool，True = 真实帧)。
    """
    out, mask = [], []
    for i in range(k - 1, -1, -1):          # 旧 → 新
        idx = i * stride
        if idx < len(hist):
            out.append(hist[idx])           # hist[0] 是最新帧
            mask.append(True)
        else:
            out.append(hist[-1])            # 最早可得的那一帧
            mask.append(False)
    return out, np.asarray(mask, dtype=bool)


@draccus.wrap()
def main(cfg: Config) -> None:
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"
    unnorm = cfg.unnorm_key or f"{cfg.task_suite_name}_no_noops"

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)
    if cfg.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter).merge_and_unload()
        print(f"已加载 adapter: {cfg.adapter}")

    # ⚠️ **动作反归一化的统计量必须注入。**
    #    底座 openvla-7b 的 norm_stats 里只有 OXE 预训练那些数据集，没有 LIBERO；
    #    LIBERO 的 q01/q99 是微调时从数据集算出来的，训练脚本存在
    #    run_dir/dataset_statistics.json。官方流程用 finetuned checkpoint 绕过了
    #    这一步，我们用 LoRA adapter 就得自己接回去 —— 不接会直接抛
    #    "unnorm_key not in available dataset statistics"（好在会报错，不会静默）。
    if unnorm not in model.norm_stats:
        sj = Path(cfg.stats_json) if cfg.stats_json else (
            Path(cfg.adapter).parents[1] / "dataset_statistics.json"
            if cfg.adapter else None)
        if sj is None or not sj.exists():
            raise SystemExit(
                f"底座里没有 `{unnorm}` 的动作统计量，也找不到 dataset_statistics.json。\n"
                f"  找过: {sj}\n"
                f"  它由训练脚本写在 run 目录下（save_dataset_statistics）。\n"
                f"  用 --stats_json 指过去，或确认 --adapter 指的是 "
                f"<run_dir>/adapter/stepN。")
        d = json.loads(sj.read_text())
        model.norm_stats[unnorm] = d.get(unnorm, d)
        print(f"已注入动作统计量: {sj}")

    state = wire(model, WireConfig(arm=cfg.arm, K=cfg.K, budget=cfg.budget,
                                   n_t=cfg.n_t))
    model.eval()

    run_id = (f"EVAL-{cfg.task_suite_name}-{cfg.arm}-K{cfg.K}s{cfg.stride}"
              f"-seed{cfg.seed}" + (f"--{cfg.run_note}" if cfg.run_note else ""))
    Path(cfg.local_log_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    print(f"日志: {log.name}")

    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    max_steps = MAX_STEPS[cfg.task_suite_name]
    total_ep = total_ok = 0
    checked = False

    for task_id in tqdm.tqdm(range(suite.n_tasks), desc="tasks"):
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, "openvla", resolution=256)
        t_ep = t_ok = 0

        for ep in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"task{task_id}",
                            leave=False):
            env.reset()
            obs = env.set_init_state(inits[ep])
            hist: deque = deque(maxlen=(cfg.K - 1) * cfg.stride + 1)
            t, done = 0, False

            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
                    t += 1
                    continue
                img = get_libero_image(obs, 224)
                hist.appendleft(img)
                frames, pad = build_window(hist, cfg.K, cfg.stride)

                # (K*6, H, W)：每帧 6 通道，沿通道拼 —— 与训练侧同一约定
                px = torch.cat([processor.image_processor.apply_transform(
                    Image.fromarray(f)) for f in frames],
                    dim=0).unsqueeze(0).to(torch.bfloat16).to(dev)
                prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
                ids = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)

                set_batch(state, depth=None,
                          frame_pad_mask=torch.from_numpy(pad).unsqueeze(0).to(dev))
                with torch.no_grad():
                    act = model.predict_action(input_ids=ids, pixel_values=px,
                                               unnorm_key=unnorm, do_sample=False)
                if not checked:
                    assert_rope_active(state)
                    print(f"  ✓ 4D RoPE 已生效（{state.rope_calls} 次）")
                    checked = True

                a = normalize_gripper_action(np.asarray(act).copy(), binarize=True)
                a = invert_gripper_action(a)
                obs, _, done, _ = env.step(a.tolist())
                if done:
                    t_ok += 1
                    total_ok += 1
                    break
                t += 1

            t_ep += 1
            total_ep += 1

        env.close()
        line = (f"task {task_id} {desc[:40]!r}: {t_ok}/{t_ep} = "
                f"{t_ok / max(t_ep, 1):.3f}   累计 {total_ok}/{total_ep} = "
                f"{total_ok / max(total_ep, 1):.4f}")
        print(line)
        log.write(line + "\n")
        log.flush()

    final = f"FINAL success_rate={total_ok / max(total_ep, 1):.4f} ({total_ok}/{total_ep})"
    print(final)
    log.write(final + "\n")
    log.close()
    print(json.dumps({"arm": cfg.arm, "suite": cfg.task_suite_name,
                      "n": total_ep, "success": total_ok}))


if __name__ == "__main__":
    main()
