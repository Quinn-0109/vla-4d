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

import gc
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

from common.imgproc import center_crop_resize  # noqa: E402
from common.runs import resolve_adapter  # noqa: E402
from pooling.wire import (WireConfig, assert_arm_wiring, frame_feats,  # noqa: E402
                          set_batch, set_vision_feats, wire)

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
    run_root: str = "runs"                     # --adapter 找不到时列清单用
    center_crop: bool = True                   # ⚠️ 训练开了 image_aug 就必须开，见下
    # ⭐ 每帧的视觉特征只算一次、跨窗口复用（视觉主干冻结，同一帧特征逐位相同）。
    #    实测每步 8 次主干前向是评测的成本大头（144 s/局，docs/05 §11.1）。
    vision_cache: bool = True
    verify_vision_cache: int = 0               # 头 N 步同时算两条路，torch.equal 对拍
    # ⭐ 并行跑几局。成本大头是 LLM 的自回归解码（视觉只占约 1/4，见 docs/05 §11.1），
    #    batch=1 时 LLM 完全没吃满。同一 task 的 prompt 相同，批起来无需 padding。
    eval_batch: int = 8
    # ⚠️ 只在 --eval_batch 1 时可用（B>1 的逐位对拍已知过不了，见下方硬拦与 §11.2）
    verify_batch: int = 0                      # 头 N 步与逐条 predict_action 对拍
    # ⭐ 只跑 [start_task, end_task) 这一段。评测**逐 task 独立**（每个 task 自己
    #    的初始状态表、贪心解码、无跨 task 状态），所以分段跑再把成功数相加，
    #    与一次跑完逐位等价 —— 这是"接着跑"而不是"重跑"的依据。
    #    ⚠️ 分段的数**必须凑满 10 个 task 才是判据数**；缺 task 的合计不作判定。
    start_task: int = 0
    end_task: int = -1                         # -1 = 到最后一个
    seed: int = 7                              # 沿用阶段 0
    local_log_dir: str = "results/logs"
    run_note: str = ""
    # fmt: on


def _allow_batched_generation(model) -> None:
    """
    去掉 openvla 对批量生成的那条断言。

    官方 `prepare_inputs_for_generation` 开头是：

        raise ValueError("Generation with batch size > 1 is not currently supported!")

    但函数体其余部分（`past_key_values` 时截最后一个 token、把 `pixel_values`
    原样带下去）**全是逐样本无关的操作**，docstring 自己写的也是
    "simplified for batch size = 1" —— 是**没做**，不是**不能做**。
    而 `forward` 本身批量安全：训练就是 batch 8 走的同一个 forward。

    ⚠️ 尽管如此，这仍是绕开上游的一道显式检查。**唯一能让它作数的是对拍**：
    `--verify_batch N` 用官方逐条 `predict_action` 跑同样的输入，比较**动作
    token**（不是 logits —— 批量 matmul 的归约顺序本来就不同）。不一致就退出。

    ⚠️ 变长 prompt 需要 padding 时这条路**不成立**（官方那条断言多半正是为它设的）。
    本脚本同一 task 内 prompt 完全等长、无 padding，所以只在这个前提下用。

    生成路径上一共**两处** batch==1 的假设，两处都在这里解开：

      ① `prepare_inputs_for_generation` 开头的 `raise`
      ② `forward` 里"带 cache 解码"那一支开头的 `assert`

    ②的分支体是：`self.language_model(input_ids, attention_mask=None,
    position_ids=None, past_key_values=..., ...)` —— `attention_mask` 与
    `position_ids` 都交给 Llama 从 cache 长度自行推导，全批等长时对每一行都相同，
    没有任何逐样本的硬编码。预填（多模态）那一支**本来就没有** batch 断言，
    正是训练走的那条，已用 batch 8 验过。
    """
    import sys
    import types

    def prep(self, input_ids=None, past_key_values=None, inputs_embeds=None,
             pixel_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update({"attention_mask": attention_mask,
                             "pixel_values": pixel_values,
                             "past_key_values": past_key_values,
                             "use_cache": kwargs.get("use_cache")})
        return model_inputs

    model.prepare_inputs_for_generation = types.MethodType(prep, model)

    Out = sys.modules[type(model).__module__].PrismaticCausalLMOutputWithPast
    orig_forward = model.forward

    def fwd(self, input_ids=None, attention_mask=None, pixel_values=None,
            labels=None, inputs_embeds=None, past_key_values=None,
            use_cache=None, output_attentions=None, output_hidden_states=None,
            output_projector_features=None, return_dict=None, **kw):
        # 只接管"带 cache 解码"这一支，逐行照抄官方分支体、去掉 batch==1 的断言
        if input_ids is not None and input_ids.shape[1] == 1:
            assert past_key_values is not None, "缓存解码必须带 past_key_values"
            assert labels is None, "缓存解码不该带 labels"
            rd = return_dict if return_dict is not None else self.config.use_return_dict
            o = self.language_model(
                input_ids=input_ids, attention_mask=None, position_ids=None,
                past_key_values=past_key_values, inputs_embeds=None, labels=None,
                use_cache=use_cache and not self.training,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, return_dict=rd)
            if not rd:
                return o
            return Out(loss=o.loss, logits=o.logits,
                       past_key_values=o.past_key_values,
                       hidden_states=o.hidden_states, attentions=o.attentions,
                       projector_features=None)
        # 其余一律交回官方实现（预填走多模态支，那支本就支持批量）
        return orig_forward(
            input_ids=input_ids, attention_mask=attention_mask,
            pixel_values=pixel_values, labels=labels, inputs_embeds=inputs_embeds,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_projector_features=output_projector_features,
            return_dict=return_dict, **kw)

    model.forward = types.MethodType(fwd, model)


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
    unnorm_key = cfg.unnorm_key or f"{cfg.task_suite_name}_no_noops"
    # ⚠️ 先把 checkpoint 路径查清楚再去加载 7B（见 common/runs）。
    #    **空串要单独拦。** shell 变量没展开时 `--adapter ""` 会落到这里，
    #    而空串是 falsy —— 于是静默地变成"评测底座模型"，照样跑完 500 局、
    #    照样给出一个成功率。这类"参数没传到但一切正常"是本项目最常见的错。
    if cfg.adapter is not None and not str(cfg.adapter).strip():
        raise SystemExit(
            "--adapter 是空串。多半是 shell 变量没展开（比如在新的 tmux 里 "
            "$G0 未定义）。空串会被当成'不加 adapter'，静默评测底座模型 —— "
            "所以这里硬拦。要评底座请**完全不传** --adapter。")
    adapter = resolve_adapter(cfg.adapter, cfg.run_root) if cfg.adapter else None
    if adapter is None:
        # ⚠️ 不传 adapter 有两种情形，别混为一谈：
        #   ① --vla_path 还是底座 openvla-7b → 真的是未微调模型，这个数没有意义
        #   ② --vla_path 指向官方已微调 checkpoint（openvla-7b-finetuned-libero-*）
        #      → 这是**路径交叉验证**的正确用法：官方权重 + 我们的评测链路，
        #        与阶段 0 用 run_libero_eval_traj.py 测出的数直接对比（docs/05 §3）。
        # 早先这里无条件喊"未微调的底座模型"。**假警报和漏报一样有害** ——
        # 它教人忽略警告，下次真的漏传 adapter 时那行字就不起作用了。
        if "finetuned" in str(cfg.vla_path):
            print(f"ℹ️ 未传 --adapter，直接评测 {cfg.vla_path} —— "
                  "官方已微调权重，用于评测链路的交叉验证。")
        else:
            print("⚠️ 未指定 adapter，评测的是**未微调的底座模型** —— "
                  "这个数不能与各臂比较。")

    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        print(f"已加载 adapter: {adapter}")

    # ⚠️ **动作反归一化的统计量必须注入。**
    #    底座 openvla-7b 的 norm_stats 里只有 OXE 预训练那些数据集，没有 LIBERO；
    #    LIBERO 的 q01/q99 是微调时从数据集算出来的，训练脚本存在
    #    run_dir/dataset_statistics.json。官方流程用 finetuned checkpoint 绕过了
    #    这一步，我们用 LoRA adapter 就得自己接回去 —— 不接会直接抛
    #    "unnorm_key not in available dataset statistics"（好在会报错，不会静默）。
    # 官方已微调 checkpoint 自带 norm_stats，键名可能是 `libero_10_no_noops`
    # 也可能是 `libero_10`（run_libero_eval_traj.py 里就有这条回退）。
    # ⚠️ 只在**用户没显式指定** --unnorm_key 时回退，且只认模型自带的键；
    #    评各臂 adapter 时两个键都不在底座里，行为与从前逐字相同（走注入）。
    if cfg.unnorm_key is None and unnorm_key not in model.norm_stats \
            and cfg.task_suite_name in model.norm_stats:
        unnorm_key = cfg.task_suite_name
        print(f"已回退到 checkpoint 自带的动作统计量键: {unnorm_key}")

    if unnorm_key not in model.norm_stats:
        sj = Path(cfg.stats_json) if cfg.stats_json else (
            adapter.parents[1] / "dataset_statistics.json" if adapter else None)
        if sj is None or not sj.exists():
            raise SystemExit(
                f"底座里没有 `{unnorm_key}` 的动作统计量，也找不到 dataset_statistics.json。\n"
                f"  找过: {sj}\n"
                f"  它由训练脚本写在 run 目录下（save_dataset_statistics）。\n"
                f"  用 --stats_json 指过去，或确认 --adapter 指的是 "
                f"<run_dir>/adapter/stepN。")
        d = json.loads(sj.read_text())
        model.norm_stats[unnorm_key] = d.get(unnorm_key, d)
        print(f"已注入动作统计量: {sj}")

    # ⚠️ `--verify_batch` 配 `--eval_batch > 1` 是一道**已知过不了**的闸，硬拦。
    #    docs/05 §11.2：与官方逐条 predict_action 240 次比较差 9 次（3.75%），
    #    成因已定位为**批量 matmul 的归约顺序**（分箱分辨率、我重写的
    #    gen_actions/unnorm 都已逐一排除）。动作分布双峰，logits 末位一点差
    #    就跳模式，所以幅度不是相邻一格。当时的处置是**弃用逐位对拍**、
    #    改用满量测量判断批量能否用（G2：39.2% vs 42.2%，Δ=3.0 < 事前容差 4.4）。
    #    留着这个组合只会让人以为它是道活闸，然后在跑到第 3 步时被打掉整个 run。
    if cfg.verify_batch and cfg.eval_batch > 1:
        raise SystemExit(
            "--verify_batch 不能与 --eval_batch > 1 同用：这道逐位对拍**已知过不了**，"
            "成因是批量 matmul 的归约顺序（docs/05 §11.2 已定位并弃用该判据）。\n"
            "  · 要判批量能否用 → 同一 checkpoint 各跑一次满量 b1 与 b8，比成功率\n"
            "  · 要查 gen_actions/unnorm 有没有写错 → "
            "`--eval_batch 1 --verify_batch N`（B=1 时两条路除了我的代码没有区别）")

    if cfg.eval_batch > 1:
        _allow_batched_generation(model)

    state = wire(model, WireConfig(arm=cfg.arm, K=cfg.K, budget=cfg.budget,
                                   n_t=cfg.n_t))
    model.eval()

    print(f"中心裁: {'开' if cfg.center_crop else '**关**'}"
          f"（训练用 image_aug=True，关掉就是训练/评测不一致）")
    # ⚠️ 分段跑必须落到**不同的**日志文件：log 是 "w" 模式打开的，
    #    段号不进 run_id 的话，接着跑 task 4-9 会把 task 0-3 的记录直接覆盖掉。
    _seg = (f"-t{cfg.start_task}_{cfg.end_task}"
            if (cfg.start_task or cfg.end_task >= 0) else "")
    run_id = (f"EVAL-{cfg.task_suite_name}-{cfg.arm}-K{cfg.K}s{cfg.stride}"
              f"-seed{cfg.seed}{'' if cfg.center_crop else '-nocrop'}{_seg}"
              + (f"--{cfg.run_note}" if cfg.run_note else ""))
    Path(cfg.local_log_dir).mkdir(parents=True, exist_ok=True)
    log = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    print(f"日志: {log.name}")

    def say(msg: str) -> None:
        """
        ⚠️ **配置与验证结果都要写进日志**。它们是这次测量的凭据：中心裁开没开、
        视觉缓存与批量各自对拍过没有 —— 事后只看得到 FINAL 那一行的话，
        无法判断这个数是在什么条件下得到的。早先它们只 print 到 stdout，
        终端一关就没了。
        """
        print(msg)
        log.write(msg + "\n")
        log.flush()

    say(f"# arm={cfg.arm} K={cfg.K} stride={cfg.stride} N={cfg.budget} "
        f"n_t={cfg.n_t} adapter={adapter} n/task={cfg.num_trials_per_task} "
        f"seed={cfg.seed}")

    say(f"# 中心裁={'开' if cfg.center_crop else '关'} "
        f"视觉特征缓存={'开' if cfg.vision_cache else '关'}"
        + (f"（头 {cfg.verify_vision_cache} 步逐位对拍）"
           if cfg.vision_cache and cfg.verify_vision_cache else ""))
    n_checked = 0
    suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    max_steps = MAX_STEPS[cfg.task_suite_name]
    total_ep = total_ok = 0
    checked = False
    B = max(1, cfg.eval_batch)
    say(f"# 批量={B} 局并行"
        + ("（已覆盖 openvla 的 batch>1 断言，见 _allow_batched_generation）"
           if B > 1 else "") + (f"（头 {cfg.verify_batch} 步与逐条 predict_action 对拍）"
                                if cfg.verify_batch else ""))
    n_bchk = n_bdiff = 0

    def to_px(fr):
        """
        帧列表 → (1, len*6, H, W)。⚠️ **中心裁必须与训练侧的
        `random_resized_crop(scale=0.9)` 对齐**：训练是 image_aug=True，模型只
        见过裁过的画面；评测喂整张图不会报错，只会让它在没见过的分布上决策。
        """
        if cfg.center_crop:
            fr = [center_crop_resize(f) for f in fr]
        return torch.cat([processor.image_processor.apply_transform(
            Image.fromarray(f)) for f in fr],
            dim=0).unsqueeze(0).to(torch.bfloat16).to(dev)

    def unnorm(tok: torch.Tensor) -> np.ndarray:
        """
        (B, 7) 动作 token → (B, 7) 反归一化动作。

        ⚠️ 逐行照抄 `OpenVLAForActionPrediction.predict_action` 的后半段 ——
        官方那份把 `generated_ids[0]` 写死了，只能取批里的第一条，所以批量时
        必须自己算。**算错不会报错**，只会让动作整体偏掉，所以有
        `--verify_batch`：头 N 步同时用官方逐条路径跑一遍，逐位比对。
        """
        d = model.vocab_size - tok.cpu().numpy()
        d = np.clip(d - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
        z = model.bin_centers[d]
        st = model.get_action_stats(unnorm_key)
        hi, lo = np.array(st["q99"]), np.array(st["q01"])
        m = np.array(st.get("mask", np.ones_like(st["q01"], dtype=bool)))
        return np.where(m, 0.5 * (z + 1) * (hi - lo) + lo, z)

    def gen_actions(ids: torch.Tensor, px: torch.Tensor) -> np.ndarray:
        """批量生成 7 个动作 token。ids (B,L)，px (B,·,H,W)。"""
        n = model.get_action_dim(unnorm_key)
        # 官方 predict_action 的那句：在 "Out:" 之后补一个空 token
        if not torch.all(ids[:, -1] == 29871):
            pad_tok = torch.full((ids.shape[0], 1), 29871, dtype=ids.dtype,
                                 device=ids.device)
            ids = torch.cat([ids, pad_tok], dim=1)
        # ⚠️ 不传 attention_mask —— 官方 predict_action 也不传，全批等长时 HF
        #    自己补全 1。多传一个参数就多一处与官方路径的差别，而对拍要的是"同"。
        with torch.no_grad():
            out = model.generate(ids, pixel_values=px, max_new_tokens=n,
                                 do_sample=False)
        return unnorm(out[:, -n:])

    t_end = suite.n_tasks if cfg.end_task < 0 else min(cfg.end_task, suite.n_tasks)
    if cfg.start_task or t_end != suite.n_tasks:
        say(f"# ⚠️ 只跑 task [{cfg.start_task}, {t_end})，共 {suite.n_tasks} 个 —— "
            f"**这不是判据数**，要凑满 10 个 task 才能对判据")
    for task_id in tqdm.tqdm(range(cfg.start_task, t_end), desc="tasks"):
        task = suite.get_task(task_id)
        inits = suite.get_task_init_states(task_id)
        # ⚠️ B 个 env 并行。它们**逐步同步推进**，所以任一时刻 pad_mask 全批相同；
        #    先结束的从活动集里摘掉，不再进 batch。
        envs, desc = [], None
        for _ in range(B):
            e, desc = get_libero_env(task, "openvla", resolution=256)
            envs.append(e)
        prompt = f"In: What action should the robot take to {desc.lower()}?\nOut:"
        ids1 = processor.tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
        t_ep = t_ok = 0

        for lo_ep in tqdm.tqdm(range(0, cfg.num_trials_per_task, B),
                               desc=f"task{task_id}", leave=False):
            eps = list(range(lo_ep, min(lo_ep + B, cfg.num_trials_per_task)))
            b = len(eps)
            obs = []
            for i, ep in enumerate(eps):
                envs[i].reset()
                obs.append(envs[i].set_init_state(inits[ep]))
            mx = (cfg.K - 1) * cfg.stride + 1
            hists = [deque(maxlen=mx) for _ in range(b)]
            feats = [deque(maxlen=mx) for _ in range(b)]
            live = list(range(b))
            t = 0

            while t < max_steps + cfg.num_steps_wait and live:
                if t < cfg.num_steps_wait:
                    for i in live:
                        obs[i], _, _, _ = envs[i].step(
                            get_libero_dummy_action("openvla"))
                    t += 1
                    continue

                px_rows, fe_rows, pads = [], [], []
                for i in live:
                    img = get_libero_image(obs[i], 224)
                    hists[i].appendleft(img)
                    frames, pad = build_window(hists[i], cfg.K, cfg.stride)
                    pads.append(pad)
                    if cfg.vision_cache:
                        # 每帧只算一次：视觉主干冻结，同一帧特征逐位相同，
                        # 而历史窗口里的每一帧都曾是某个时刻的"当前帧"。
                        feats[i].appendleft(frame_feats(state, model, to_px([img])))
                        jj = [k * cfg.stride for k in range(cfg.K - 1, -1, -1)]
                        fe = torch.cat([feats[i][min(j, len(feats[i]) - 1)]
                                        for j in jj], dim=1)
                        if n_checked < cfg.verify_vision_cache:
                            ref = torch.cat([state.orig["vision"](c) for c in
                                             torch.split(to_px(frames), 6, dim=1)],
                                            dim=1)
                            if not torch.equal(fe, ref):
                                d = (fe.float() - ref.float()).abs().max().item()
                                raise SystemExit(
                                    f"❌ 视觉特征缓存与逐帧重算**不逐位相同**"
                                    f"（最大差 {d:.3e}）。缓存的前提是视觉主干冻结、"
                                    "同一帧特征恒定；前提不成立就不能用它测量。"
                                    "--vision_cache False 关掉。")
                            n_checked += 1
                            if n_checked == cfg.verify_vision_cache:
                                say(f"# ✓ 视觉特征缓存逐位对拍通过（{n_checked} 步）")
                        fe_rows.append(fe)
                        px_rows.append(to_px(frames[-1:]))   # 占位，主干已短路
                    else:
                        px_rows.append(to_px(frames))

                px = torch.cat(px_rows, dim=0)
                pm = torch.from_numpy(np.stack(pads)).to(dev)
                ids = ids1.repeat(len(live), 1)      # expand 出来的是视图，generate 要连续
                if cfg.vision_cache:
                    set_vision_feats(state, torch.cat(fe_rows, dim=0))
                set_batch(state, depth=None, frame_pad_mask=pm)
                acts = gen_actions(ids, px)

                if not checked:
                    assert_arm_wiring(state, cfg.arm)
                    say(f"# ✓ 接线正常（arm={cfg.arm}，rotary_emb "
                        f"{state.rope_calls} 次）")
                    checked = True

                # ⭐ 批量 vs 逐条：**比动作 token，不比 logits**。批量 matmul 的
                #    归约顺序与 batch=1 不同，末位可能有差；真正要保证的是
                #    argmax 出来的 token 一致，那才是喂进仿真器的东西。
                # ⚠️ 不要求 len(live) > 1：`--eval_batch 1 --verify_batch N` 正是用来
                #    把"我重写的 gen_actions/unnorm 有没有错"从"批量的浮点差异"里
                #    隔离出来 —— B=1 时两条路除了我的代码没有任何区别，
                #    若仍不一致，那就是我的 bug 而非数值噪声。
                if n_bchk < cfg.verify_batch:
                    for r in range(len(live)):
                        if cfg.vision_cache:
                            set_vision_feats(state, fe_rows[r])
                        set_batch(state, depth=None, frame_pad_mask=pm[r:r + 1])
                        with torch.no_grad():
                            ref = model.predict_action(
                                input_ids=ids1, pixel_values=px_rows[r],
                                unnorm_key=unnorm_key, do_sample=False)
                        if not np.array_equal(np.asarray(ref), acts[r]):
                            n_bdiff += 1
                            print(f"  ⚠️ 批量与逐条不一致 第{n_bchk}步 行{r}:"
                                  f"\n     批量 {np.round(acts[r], 4)}"
                                  f"\n     逐条 {np.round(np.asarray(ref), 4)}")
                    n_bchk += 1
                    if n_bchk == cfg.verify_batch:
                        if n_bdiff:
                            raise SystemExit(
                                f"❌ 批量与逐条 predict_action 在 {n_bdiff} 处不一致。"
                                "批量改变了测量结果，不能用。--eval_batch 1 关掉，"
                                "或先查 unnorm/gen_actions 是否与官方逐位一致。")
                        say(f"# ✓ 批量与逐条 predict_action 逐位一致"
                            f"（{n_bchk} 步 × {len(live)} 行）")

                nxt = []
                for r, i in enumerate(live):
                    a = normalize_gripper_action(acts[r].copy(), binarize=True)
                    a = invert_gripper_action(a)
                    obs[i], _, done, _ = envs[i].step(a.tolist())
                    if done:
                        t_ok += 1
                        total_ok += 1
                    else:
                        nxt.append(i)
                live = nxt
                t += 1

            t_ep += b
            total_ep += b
            # ⚠️ **碎片，不是泄漏。** `live` 随着 episode 陆续成功而缩小，
            #    batch 形状一路 8→7→…→1，每种形状都让缓存分配器切出不同大小的块；
            #    跑满 200 局后 `1.07 GiB reserved but unallocated` 却申请不到
            #    连续的 20 MiB —— 这就是第一次满量在 task 4 崩掉的原因。
            #    每批（8 局）回收一次，相对于一批几分钟的耗时可以忽略。
            torch.cuda.empty_cache()

        # ⚠️ close 本身会抛 EGL 错（退出时那一串 `Exception ignored in __del__`），
        #    抛了就等于没释放，所以逐个 close 且不让异常打断。
        #    ⚠️ **但 OOM 不是它造成的**：报错里 PyTorch 自己占 20.66 GiB /
        #    进程总共 22.68 GiB，non-PyTorch（EGL 渲染上下文）只有约 1 GB，
        #    没在涨。真正的成因是**碎片**，见下面 episode 批结束处的 empty_cache。
        #    我最初把它归到 env 泄漏上，是先有猜想再看数——记在这里。
        for e in envs:
            try:
                e.close()
            except Exception as err:            # noqa: BLE001
                print(f"  (env.close 抛了，已忽略: {type(err).__name__})")
        envs = []
        gc.collect()
        torch.cuda.empty_cache()

        line = (f"task {task_id} {desc[:40]!r}: {t_ok}/{t_ep} = "
                f"{t_ok / max(t_ep, 1):.3f}   累计 {total_ok}/{total_ep} = "
                f"{total_ok / max(total_ep, 1):.4f}   "
                f"显存 {torch.cuda.memory_reserved() / 2**30:.1f} GB")
        print(line)
        log.write(line + "\n")
        log.flush()

    if cfg.start_task == 0 and t_end == suite.n_tasks:
        final = f"FINAL success_rate={total_ok / max(total_ep, 1):.4f} ({total_ok}/{total_ep})"
    else:
        # ⚠️ **残缺的合计不叫 FINAL。** 判据认的是满 10 task 的数；
        #    打上 FINAL 就等于给了一个能被当成判据用的数字，而它不是。
        final = (f"PARTIAL task[{cfg.start_task},{t_end}) "
                 f"{total_ok}/{total_ep} = {total_ok / max(total_ep, 1):.4f}"
                 f"   —— 不是判据数；把各段的成功数相加、总局数相加才是")
    print(final)
    log.write(final + "\n")
    log.close()
    print(json.dumps({"arm": cfg.arm, "suite": cfg.task_suite_name,
                      "n": total_ep, "success": total_ok}))


if __name__ == "__main__":
    main()
