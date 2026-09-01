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
import numpy as np
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
from data.depth_cache import DepthCache  # noqa: E402
from data.kframe import (KFrameBatchTransform, PaddedCollatorKFrame,  # noqa: E402
                         patch_strided_chunking)
from pooling.wire import (WireConfig, assert_arm_wiring, set_batch,  # noqa: E402
                          wire)

# 微批 × 累积，有效批恒为 16（docs/06 §4.3 的显存实测）
#
# ⚠️⚠️ **K=8 的微批表已作废，下面这组数实测于 K=1**（docs/05 §8.10）：
#
#     micro=2  2.7 s/步  16.3 GB      ← 这三行是数据侧 K 退化成 1 时测的，
#     micro=4  2.4 s/步  16.9 GB      ← 当时的结论"4×4 最优""30k 步 19.7 h"
#     micro=8  2.4 s/步  17.9 GB      ← "瓶颈在视觉主干"全部随之作废
#
# 对账的线索一直在磁盘上：`probe_vram` 冻结视觉主干、真 K=8 的实测是
# micro=2 → 19.3 GB、micro=4 → 22.9 GB，比上表高 3–6 GB；而且上表里
# micro 从 2 加到 4 显存几乎不涨 —— K=8 时每多一个样本要多 8 帧激活，
# 不可能这么平。**两份实测互相矛盾，摆了三周没对账。**
#
# 真 K=8 重测（2026-08，micro=2）：**4.4 s/步，峰值 16.6 GB，30k 步 37 h**。
# 与 K=1 那轮（2.35 s/步、16.33 GB）对照，读出来的是：
#
#   **池化省的是显存和 LLM 那一段的时间，成本大头（视觉主干）一点不省。**
#   显存几乎不变（序列长都是 294，视觉主干冻结、分 K 次串行不留激活），
#   时间翻 1.87 倍（8 倍视觉前向）。压缩必须靠效果说话。
#
# 微批扫描（真 K=8，2026-08）：
#
#     micro=2  4.4 s/步  16.6 GB  30k 步 37.0 h
#     micro=4  3.9 s/步  17.4 GB  30k 步 32.1 h
#     micro=8  3.7 s/步  19.0 GB  30k 步 30.8 h   ← 取这个
#
# 2→8 省 16%，4→8 只再省 1.3 h 却多 1.6 GB —— 边际很小，但 24 GB 卡装得下。
# 与 K=1 那轮"4→8 归零"的形状不同：那时每步只有 16 次视觉前向，现在是 128 次。
#
# ⚠️ **上表的 s/步偏高**：`--bench_only` 只跑 10 步，含数据管道预热与内核首次
#    编译。G2 实跑的稳态是 **2.79 s/步**，30k 步 **23.3 h**（micro=8）。
#    做预算用稳态值；上表只用来比较微批之间的相对快慢（那个比较不受影响）。
#
# ⚠️ **有效批必须恒为 16**，所以只让改微批，累积由它自动配平。
EFF_BATCH = 16
MICRO = {1: (16, 1), 8: (8, 2)}      # K=8 取 micro=8，见上表


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
    # ⚠️ 只有**最新**那个 checkpoint 需要优化器状态（续训用）。旧的只拿来评测，
    #    光要 adapter 权重就够。AdamW 状态占每个 checkpoint 约 2/3（~305 MB），
    #    30k 步存 12 个就是 3.7 GB —— 这台机器盘只剩 6 GB，写满了会把长跑断掉。
    #    置 True 保留全部（想从中间某一步分叉续训时才需要）。
    keep_all_optim: bool = False
    # G4/M2 用：build_subset.py 的产物目录（depth_cache.npz + bbox.json）
    subset_dir: str = "results/subset/libero_10"
    camera_json: str = "results/tables/camera_libero.json"
    log_steps: int = 10
    resume_from: Optional[str] = None
    bench_only: bool = False
    # ⚠️ 只前向、不更新，用来**复核某个 checkpoint 的 acc**。与 --resume_from 配合：
    #    走的是这个文件里训练用的同一条链路（同一个 RLDSDataset、同一个
    #    KFrameBatchTransform、同一套 acc 公式），所以数字与训练日志逐位可比 ——
    #    另写一个评测脚本去"重现"训练输入，重现错了会长得像"模型没学会"。
    eval_only: bool = False
    eval_steps: int = 100                      # 只前向时跑几个微批
    micro: int = 0                             # 0 = 用 MICRO 表；>0 覆盖，累积自动配平
    run_id_note: Optional[str] = None
    # fmt: on


@draccus.wrap()
def main(cfg: Config) -> None:
    assert torch.cuda.is_available(), "需要 GPU"
    dev = "cuda"

    # G4/M2 的三份前置产物，缺一不可。**在加载 7B 之前查**。
    metric = cfg.arm in ("G4", "M2")
    depth_cache = cameras = bbox = None
    if metric:
        sd = Path(cfg.subset_dir)
        need = [sd / "depth_cache.npz", sd / "bbox.json", Path(cfg.camera_json)]
        miss = [str(x) for x in need if not x.exists()]
        if miss:
            raise SystemExit(
                f"{cfg.arm} 用度量坐标，需要这几份产物，缺了：\n  "
                + "\n  ".join(miss)
                + "\n\n先跑：\n"
                "  python scripts/dump_camera.py                    # 相机参数\n"
                "  python scripts/build_subset.py --suite libero_10 # 先看分布\n"
                "  python scripts/build_subset.py --suite libero_10 --commit\n"
                "⚠️ 拿不到真深度就只能得到 (t,h,w,z)，那样的\"G4\"是带深度通道的 G3。")
        depth_cache = DepthCache.load(sd / "depth_cache.npz")
        bb = json.loads((sd / "bbox.json").read_text())
        bbox = torch.tensor([bb["lo"], bb["hi"]], dtype=torch.float32)
        cams_raw = json.loads(Path(cfg.camera_json).read_text())
        from common.camera import Camera
        cameras = {int(k): Camera(
            fovy=float(v["fovy"]), height=int(v["height"]), width=int(v["width"]),
            pos=torch.tensor(v["pos"], dtype=torch.float64),
            rot=torch.tensor(v["rot"], dtype=torch.float64).reshape(3, 3),
            flipped=bool(v.get("flipped", True))) for k, v in cams_raw.items()}
        print(f"深度缓存 {len(depth_cache)} 帧；相机 {len(cameras)} 台；"
              f"包围盒 x[{bb['lo'][0]:+.2f},{bb['hi'][0]:+.2f}] "
              f"y[{bb['lo'][1]:+.2f},{bb['hi'][1]:+.2f}] "
              f"z[{bb['lo'][2]:+.2f},{bb['hi'][2]:+.2f}]")

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
    wcfg = WireConfig(arm=cfg.arm, K=cfg.K, budget=cfg.budget, n_t=cfg.n_t,
                      bbox=bbox)
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

    lang_to_task = None
    if metric:
        from libero.libero import benchmark as _bm
        suite = _bm.get_benchmark_dict()[cfg.dataset_name.replace("_no_noops", "")]()
        lang_to_task = {suite.get_task(i).language.strip().lower(): i
                        for i in range(suite.n_tasks)}

    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # ⚠️ 必须在构造 RLDSDataset **之前**打补丁（见 data.kframe.patch_strided_chunking）
    patch_strided_chunking(cfg.stride, cfg.K)
    dataset = RLDSDataset(
        cfg.data_root_dir, cfg.dataset_name,
        KFrameBatchTransform(action_tokenizer, processor.tokenizer,
                             image_transform=processor.image_processor.apply_transform,
                             prompt_builder_fn=PurePromptBuilder,
                             lang_to_task=lang_to_task),
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
        if not cfg.keep_all_optim:
            # 新的存好了再删旧的 —— 顺序反过来的话，存盘中途断电就两头落空
            freed = 0
            for old in adapter_dir.glob("step*/trainer_state.pt"):
                if old.parent != d:
                    freed += old.stat().st_size
                    old.unlink()
            if freed:
                print(f"  （清掉旧 checkpoint 的优化器状态，省 {freed / 2**30:.1f} GB；"
                      f"续训只能从最新这个 step{step} 起）")
        print(f"\n[step {step}] 已存 -> {d}")
        # ⚠️ 续训**必须用同一个 --micro**：exp_id 里带 b{micro}x{accum}，
        #    换了微批就写进另一个目录，等于从头再来而且不报错。
        print(f"  续训: python scripts/finetune_kframe.py --arm {cfg.arm} "
              f"--micro {micro} --data_root_dir {cfg.data_root_dir} "
              + (f"--run_id_note {cfg.run_id_note} " if cfg.run_id_note else "")
              + f"--resume_from {d}", flush=True)

    # ⚠️⚠️ **第一批就把形状钉死。** openvla 的 RLDSDataset 把 window_size=1 写死在
    #    构造函数里，K 一旦没顶进去就会静默退化成单帧：pixel_values 变成
    #    (B, 6, H, W)、序列长不变、loss 照降、acc 照升，只有拿去做 K 帧评测才露馅。
    def batch_depth(batch):
        """
        (B,K) 指纹 → (B,K,256) 深度 + 每个样本的相机。**查不到硬抛**
        （`DepthCache.lookup`）：退回零深度不会报错，只会让 G4 看一个
        与图像无关的三维世界。
        """
        if not metric:
            return None, ()
        h = batch["img_hash"].numpy()
        d = np.stack([depth_cache.lookup(row) for row in h])      # (B,K,16,16)
        cams = [cameras[int(t)] for t in batch["task_id"].tolist()]
        return torch.from_numpy(d.reshape(d.shape[0], d.shape[1], -1)), cams

    def check_shapes(batch) -> None:
        pv, fm = batch["pixel_values"], batch["frame_pad_mask"]
        if pv.shape[1] != cfg.K * 6 or fm.shape[1] != cfg.K:
            raise SystemExit(
                f"数据侧给的不是 K={cfg.K} 帧：pixel_values {tuple(pv.shape)}"
                f"（应为 (B, {cfg.K * 6}, H, W)），frame_pad_mask {tuple(fm.shape)}"
                f"（应为 (B, {cfg.K})）。\n"
                f"实际 K = {pv.shape[1] // 6} —— 分窗补丁没生效或被官方的 "
                f"window_size 顶回去了，先查 data.kframe.patch_strided_chunking。")
        print(f"  ✓ 形状: pixel_values {tuple(pv.shape)}  "
              f"frame_pad_mask {tuple(fm.shape)}  → K={pv.shape[1] // 6}")

    if cfg.eval_only:
        if not cfg.resume_from:
            raise SystemExit("--eval_only 要配 --resume_from <adapter 目录>，"
                             "否则复核的是没微调过的底座")
        vla.eval()
        hit = tot = 0
        loss_sum = 0.0
        checked = False
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= cfg.eval_steps:
                    break
                if i == 0:
                    check_shapes(batch)
                dep, cams = batch_depth(batch)
                set_batch(state, depth=None if dep is None else dep.to(dev),
                          frame_pad_mask=batch["frame_pad_mask"].to(dev),
                          cameras=cams)
                # ⚠️ **必须显式 use_cache=False。** openvla 的 forward 里是
                #    `use_cache = use_cache and not self.training` —— 训练时
                #    self.training=True 把它关掉了，而只前向复核时是 eval() 模式，
                #    于是白建一份 16×278 的 KV cache（约 2.3 GB），加上 logits
                #    升 fp32 的两份拷贝，直接 OOM。我们不生成，用不着 cache。
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = vla(input_ids=batch["input_ids"].to(dev),
                              attention_mask=batch["attention_mask"].to(dev),
                              pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                              labels=batch["labels"], use_cache=False)
                if not checked:
                    assert_arm_wiring(state, cfg.arm)
                    print(f"  ✓ 接线正常（arm={cfg.arm}，rotary_emb {state.rope_calls} 次）"
                          f"  序列长 {out.logits.shape[1]}")
                    checked = True
                preds = out.logits[:, n_vis:-1].argmax(dim=2)
                gt = batch["labels"][:, 1:].to(preds.device)
                m = gt > action_tokenizer.action_token_begin_idx
                hit += int(((preds == gt) & m).sum())
                tot += int(m.sum())
                loss_sum += float(out.loss)
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1:>4}/{cfg.eval_steps} 微批  "
                          f"acc {hit / max(tot, 1):.3f}  loss {loss_sum / (i + 1):.3f}",
                          flush=True)
        print(f"\n=== 只前向复核（{cfg.arm}，{cfg.resume_from}）===")
        print(f"  动作 token 准确率 {hit / max(tot, 1):.4f}（{hit}/{tot}）")
        print(f"  loss {loss_sum / max(min(cfg.eval_steps, i + 1), 1):.4f}")
        print("  ⬆ 与训练日志末尾的 acc/loss 直接比。差不多 = 训练侧没问题，"
              "问题在评测的输入构造；差很多 = checkpoint 或接线本身有问题。")
        return

    vla.train()
    optimizer.zero_grad()
    checked_rope = False
    with tqdm.tqdm(total=cfg.max_steps, initial=start_step) as bar:
        for micro_idx, batch in enumerate(loader):
            if micro_idx == 0:
                check_shapes(batch)
            dep, cams = batch_depth(batch)
            set_batch(state, depth=None if dep is None else dep.to(dev),
                      frame_pad_mask=batch["frame_pad_mask"].to(dev),
                      cameras=cams)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = vla(input_ids=batch["input_ids"].to(dev),
                          attention_mask=batch["attention_mask"].to(dev),
                          pixel_values=batch["pixel_values"].to(torch.bfloat16).to(dev),
                          labels=batch["labels"])
            (out.loss / accum).backward()

            if not checked_rope:
                assert_arm_wiring(state, cfg.arm)   # 挂点错了不会崩，只会全程用 1D RoPE
                print(f"  ✓ 接线正常（arm={cfg.arm}，rotary_emb {state.rope_calls} 次）"
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
                       # ⚠️ 这是**本次进程**的墙钟，续训时 t0 重置。
                       #    总时长要把各段相加，别把它当总数读。
                       "steps_per_sec": sps, "session_h": el / 3600,
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
