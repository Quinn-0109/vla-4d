"""
多帧输入的显存实测 —— 阶段 B 开工前必须先拿到的数字。

    python scripts/probe_vram.py --sweep          # 跑完整决策表
    python scripts/probe_vram.py --K 8 --keep 96 --mode train   # 单点

回答三个问题:
  1. K 帧不压缩，4090 的 24 GB 装不装得下？（K=8 是 2048 个视觉 token）
  2. 压到 keep/帧 之后能装下多少 K？
  3. 梯度检查点能换来多少？代价是慢多少？

⚠️ 不需要数据集。用随机张量测显存是准的——显存占用只取决于形状与 dtype，
   与数值内容无关。所以这一步可以在下载几十 GB 的 RLDS 之前就做完。

多帧输入的构造: pixel_values 堆成 (B, K*6, 224, 224)，视觉主干按 6 通道
一组切开分别过 featurizer，再沿 token 维拼接 -> (B, K*256, D)。
这样批维度不变，modeling_prismatic.py:362 那句
`input_ids.shape[0] == pixel_values.shape[0]` 的断言不用动。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CKPT = os.environ.get("PROBE_CKPT", "openvla/openvla-7b-finetuned-libero-spatial")
N_PATCH, LANG_LEN = 256, 40      # 视觉 token 数；提示词+动作 token 的粗略长度


def patch_multiframe(model, K: int, keep: int):
    """把视觉主干换成多帧版本：(B, K*6, H, W) -> (B, K*keep, D)。"""
    orig = model.vision_backbone.forward
    fn = None
    if keep and keep < N_PATCH:
        from pooling.token_select import tome_merge
        fn = tome_merge

    def wrapped(pixel_values, *a, **kw):
        chunks = torch.split(pixel_values, 6, dim=1)      # K 个 (B, 6, H, W)
        outs = []
        for c in chunks:
            f = orig(c, *a, **kw)                          # (B, 256, D)
            if fn is not None:
                f = fn(f, keep).to(f.dtype)
            outs.append(f)
        return torch.cat(outs, dim=1)                     # (B, K*keep, D)

    model.vision_backbone.forward = wrapped


def measure(K: int, keep: int, batch: int, mode: str, ckpt_grad: bool) -> dict:
    from transformers import AutoModelForVision2Seq

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2",
    ).to("cuda")
    patch_multiframe(model, K, keep)

    n_vis = (keep if keep and keep < N_PATCH else N_PATCH) * K
    weights_gb = torch.cuda.max_memory_allocated() / 1e9

    if mode == "train":
        from peft import LoraConfig, get_peft_model
        # 与 vla-scripts/finetune.py:174 一致
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.0,
            target_modules="all-linear", init_lora_weights="gaussian"))
        if ckpt_grad:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
        model.train()
    else:
        model.eval()

    px = torch.randn(batch, K * 6, 224, 224, dtype=torch.bfloat16, device="cuda")
    ids = torch.randint(100, 30000, (batch, LANG_LEN), device="cuda")
    att = torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats()

    try:
        if mode == "train":
            labels = ids.clone()
            for _ in range(2):                # 第二步才把优化器状态算进去
                out = model(input_ids=ids, attention_mask=att, pixel_values=px, labels=labels)
                out.loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(input_ids=ids, attention_mask=att, pixel_values=px)
        peak = torch.cuda.max_memory_allocated() / 1e9
        ok = True
    except torch.cuda.OutOfMemoryError:
        peak, ok = float("nan"), False

    # 确认梯度检查点真的挂上了——静默失效会让我们把"没省下来"读成"省不了"
    gc_live = any(getattr(m, "gradient_checkpointing", False) for m in model.modules())

    return {"K": K, "keep": keep or N_PATCH, "batch": batch, "mode": mode,
            "grad_ckpt": ckpt_grad, "grad_ckpt_active": gc_live,
            "n_visual_tokens": n_vis,
            "seq_len": 1 + n_vis + LANG_LEN,
            "weights_gb": round(weights_gb, 2),
            "peak_gb": None if not ok else round(peak, 2), "ok": ok}


SWEEP = [
    # (K, keep, batch, mode, grad_ckpt)
    *[(k, 0, 1, "infer", False) for k in (1, 2, 4, 8, 16)],
    *[(k, 96, 1, "infer", False) for k in (1, 2, 4, 8, 16)],
    *[(k, 0, 1, "train", False) for k in (1, 2, 4, 8)],
    *[(k, 0, 1, "train", True) for k in (4, 8, 16)],
    *[(k, 96, 1, "train", False) for k in (1, 2, 4, 8)],
    *[(k, 96, 1, "train", True) for k in (8, 16)],
    (1, 0, 4, "train", True), (4, 96, 4, "train", True),
]


def main():
    # ⚠️ 管道输出(| tee)会让 stdout 变成块缓冲，整个 sweep 的输出填不满一个
    #    缓冲区，于是跑 25 分钟一个字都看不到，看起来像卡死。强制行缓冲。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="跑完整决策表（每个配置一个子进程）")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--keep", type=int, default=0, help="0 = 不压缩")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mode", default="infer", choices=["infer", "train"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--out", default="results/tables/vram_probe.json")
    ap.add_argument("--timeout", type=int, default=900,
                    help="单个配置的超时(秒)；卡住就当失败继续，不拖垮整轮")
    args = ap.parse_args()

    if not args.sweep:
        print(json.dumps(measure(args.K, args.keep, args.batch, args.mode, args.grad_ckpt)))
        return

    # 每个配置起独立子进程: OOM 之后显存碎片会污染同进程内的后续测量
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"显卡: {torch.cuda.get_device_name(0)}  显存 {total:.1f} GB")
    print(f"共 {len(SWEEP)} 个配置，每个都要重新加载 7B 模型（约 1 分钟），"
          f"预计 {len(SWEEP)}~{len(SWEEP)*2} 分钟\n")
    results = []
    for i, (K, keep, batch, mode, gc) in enumerate(SWEEP, 1):
        cmd = [sys.executable, "-u", __file__, "--K", str(K), "--keep", str(keep),
               "--batch", str(batch), "--mode", mode] + (["--grad_ckpt"] if gc else [])
        # 测之前就报，否则加载模型那一分钟里用户只能盯着空屏幕猜是不是卡了
        print(f"[{i:>2}/{len(SWEEP)}] K={K:<2} keep={keep or N_PATCH:<3} b={batch} "
              f"{mode:<5} {'ckpt' if gc else '    '} ... ", end="", flush=True)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        except subprocess.TimeoutExpired:
            print(f"超时(>{args.timeout}s)")
            results.append({"K": K, "keep": keep or N_PATCH, "batch": batch,
                            "mode": mode, "grad_ckpt": gc, "ok": False,
                            "peak_gb": None, "n_visual_tokens": (keep or N_PATCH) * K,
                            "seq_len": None, "error": "timeout"})
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            d = {"K": K, "keep": keep or N_PATCH, "batch": batch, "mode": mode,
                 "grad_ckpt": gc, "ok": False, "peak_gb": None,
                 "n_visual_tokens": (keep or N_PATCH) * K, "seq_len": None,
                 "error": (r.stderr or "")[-200:]}
        results.append(d)
        flag = f"{d['peak_gb']:.1f} GB" if d.get("peak_gb") else "OOM"
        warn = ""
        if gc and d.get("ok") and not d.get("grad_ckpt_active"):
            warn = "  ⚠️ 梯度检查点未生效"
        if d.get("error"):
            warn += f"  [{d['error'][:80]}]"
        print(f"tok={d['n_visual_tokens']:>4}  {flag}{warn}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")

    # 决策摘要
    def best(mode, keep, gc):
        ks = [d["K"] for d in results
              if d["mode"] == mode and d["keep"] == keep and d["grad_ckpt"] == gc
              and d["batch"] == 1 and d.get("ok")]
        return max(ks) if ks else None

    print("\n=== 决策表（batch=1 时能跑到的最大 K）===")
    for mode in ("infer", "train"):
        for keep in (256, 96):
            for gc in (False, True):
                k = best(mode, keep, gc)
                if k is not None:
                    print(f"  {mode:>5} keep={keep:>3} "
                          f"{'梯度检查点' if gc else '普通      '}: K ≤ {k}")


if __name__ == "__main__":
    main()
