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
# 与 download_checkpoints.sh 一致走镜像: trust_remote_code 要回源拉
# modeling_prismatic.py，直连 huggingface.co 会超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def pick_attn() -> str:
    """
    用**当前环境实际能用的**注意力实现，而不是硬编码 flash_attention_2。

    ⚠️ 这不是可有可无的适配: flash-attn 与 sdpa 的显存差异在长序列上最大，
       而 K=8 正是 2048 token 的长序列区间。用错实现测出来的"装不下"，
       换个实现可能就装得下——结论会反过来。所以结果里必须带上用的是哪个。
    """
    import importlib.util as iu
    forced = os.environ.get("PROBE_ATTN")
    if forced:
        return forced
    return "flash_attention_2" if iu.find_spec("flash_attn") is not None else "sdpa"
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


def measure(K: int, keep: int, batch: int, mode: str, ckpt_grad: bool,
            freeze_vision: bool = False, cameras: int = 1) -> dict:
    """
    cameras=2 时每个时间步有两张图（第三人称 + 腕部），显存上等价于把帧数翻倍:
    2K 次视觉主干前向 + 2K 份 token。所以内部按 K*cameras 帧处理，
    但结果里 K 与 cameras 分开记录，事后才看得懂。
    """
    from transformers import AutoModelForVision2Seq

    # 每个时间步 cameras 张图 → 视觉主干实际要前向 K*cameras 次。
    # ⚠️ 这一行是加 --cameras 时漏掉的：docstring 写了"内部按 K*cameras 帧处理"，
    #    但变量没定义，于是每个配置都以 NameError 崩掉——而父进程把任何失败
    #    都报成 OOM，整张决策表全是假的 OOM。两个 bug 叠在一起才没被立刻看穿。
    n_img = K * cameras

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    attn = pick_attn()
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation=attn,
    ).to("cuda")
    patch_multiframe(model, n_img, keep)

    n_vis = (keep if keep and keep < N_PATCH else N_PATCH) * n_img
    weights_gb = torch.cuda.max_memory_allocated() / 1e9

    if mode == "train":
        from peft import LoraConfig, get_peft_model

        from common.lora import lora_targets
        # ⚠️ 必须与 finetune_single.py 用同一份目标层逻辑。
        #    freeze_vision=True 时视觉主干不挂 LoRA，其激活不进反传图——
        #    这才是实际训练的配置。用 all-linear 测出来的表描述的是别的东西。
        model = get_peft_model(model, LoraConfig(
            r=32, lora_alpha=16, lora_dropout=0.0,
            target_modules=lora_targets(model, include_vision=not freeze_vision),
            init_lora_weights="gaussian"))
        if ckpt_grad:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)
        model.train()
    else:
        model.eval()

    px = torch.randn(batch, n_img * 6, 224, 224, dtype=torch.bfloat16, device="cuda")
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

    return {"K": K, "cameras": cameras, "images_encoded": n_img,
            "keep": keep or N_PATCH, "batch": batch, "mode": mode,
            "attn": attn, "freeze_vision": freeze_vision,
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

# 批量扫描: batch=1 跑 7B 的 GPU 利用率很低，找出能吃下的最大 batch，
# 直接决定那次 20-45 小时的训练能压到多久。
# 默认扫 K=1(单帧复现)，但主线是 K=8 —— 用 `--sweep_batch --K 8` 扫主线配置，
# 因为**多帧下的 batch 上限不能由 K=1 外推**: 每加一个 batch 要多存一整份
# K 帧的视觉激活，K=8 时这份开销是 K=1 的八倍。
def sweep_batch(K=1, keep=0):
    return [(K, keep, b, "train", gc) for b in (2, 4, 8, 16) for gc in (False, True)]


def main():
    # ⚠️ 管道输出(| tee)会让 stdout 变成块缓冲，整个 sweep 的输出填不满一个
    #    缓冲区，于是跑 25 分钟一个字都看不到，看起来像卡死。强制行缓冲。
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="跑完整决策表（每个配置一个子进程）")
    ap.add_argument("--sweep_batch", action="store_true",
                    help="只扫 K=1 的批量，给单帧复现选吞吐最优的配置")
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--keep", type=int, default=0, help="0 = 不压缩")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--mode", default="infer", choices=["infer", "train"])
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--freeze_vision", action="store_true",
                    help="视觉主干不挂 LoRA —— 与 finetune_single.py 的实际配置一致")
    ap.add_argument("--cameras", type=int, default=1, choices=[1, 2],
                    help="2 = 加腕部相机，每帧 token 翻倍（docs/06 C.0 ③）")
    ap.add_argument("--out", default="results/tables/vram_probe.json")
    ap.add_argument("--timeout", type=int, default=900,
                    help="单个配置的超时(秒)；卡住就当失败继续，不拖垮整轮")
    args = ap.parse_args()

    if not (args.sweep or args.sweep_batch):
        print(json.dumps(measure(args.K, args.keep, args.batch, args.mode,
                                 args.grad_ckpt, args.freeze_vision, args.cameras)))
        return

    # 每个配置起独立子进程: OOM 之后显存碎片会污染同进程内的后续测量
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    attn = pick_attn()
    print(f"显卡: {torch.cuda.get_device_name(0)}  显存 {total:.1f} GB")
    print(f"注意力实现: {attn}   视觉主干: "
          f"{'冻结（与实际训练一致）' if args.freeze_vision else '挂 LoRA（官方 all-linear）'}"
          f"   相机数: {args.cameras}")
    if attn == "sdpa":
        print("  ⚠️ 未装 flash-attn，测的是 sdpa 的显存。长序列上 sdpa 比 flash-attn"
              "\n     费显存，K 大时差距明显——这组数字是**保守下界**。")
    plan = sweep_batch(args.K, args.keep) if args.sweep_batch else SWEEP
    print(f"共 {len(plan)} 个配置，每个都要重新加载 7B 模型（约 1 分钟），"
          f"预计 {len(plan)}~{len(plan)*2} 分钟\n")
    results = []
    for i, (K, keep, batch, mode, gc) in enumerate(plan, 1):
        cmd = ([sys.executable, "-u", __file__, "--K", str(K), "--keep", str(keep),
                "--batch", str(batch), "--mode", mode, "--cameras", str(args.cameras)]
               + (["--grad_ckpt"] if gc else [])
               + (["--freeze_vision"] if args.freeze_vision else []))
        # 测之前就报，否则加载模型那一分钟里用户只能盯着空屏幕猜是不是卡了
        print(f"[{i:>2}/{len(plan)}] K={K:<2} keep={keep or N_PATCH:<3} b={batch:<2} "
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
            # 子进程崩了(未被捕获的 OOM 等)，也要带上 attn，否则事后无法核对
            # 整轮用的是不是同一个实现
            d = {"K": K, "cameras": args.cameras, "images_encoded": K * args.cameras,
                 "keep": keep or N_PATCH, "batch": batch, "mode": mode,
                 "attn": attn, "freeze_vision": args.freeze_vision,
                 "grad_ckpt": gc, "ok": False, "peak_gb": None,
                 "n_visual_tokens": (keep or N_PATCH) * K * args.cameras,
                 "seq_len": None, "error": (r.stderr or "")[-600:],
                 "returncode": r.returncode}
        results.append(d)

        # ⚠️ **不能把任何失败都当成 OOM。** 初版这么写过，结果一次签名/环境错误
        # 让整张表全是 "OOM"，包括 K=1——而 K=1 的实际训练只用 16.8 GB。
        # 那张表会直接导出"连单帧都塞不下、必须压缩"的错误结论，
        # 而"压缩是使能而非优化"这个叙事本项目已经撤回过一次（docs/06 阶段 B ①）。
        # 之所以当场识破，是因为手上有独立的实测事实可对照；没有对照的话
        # 这类静默错误会一路带进决策。
        err = d.get("error") or ""
        # 显存耗尽有好几种长相: Python 异常、CUDA driver 报错、cuBLAS 分配失败。
        # 只认 `OutOfMemoryError` 会把后两种误判成代码错误。
        is_oom = (d.get("oom") is True
                  or any(k in err for k in ("OutOfMemoryError",
                                            "CUBLAS_STATUS_ALLOC_FAILED",
                                            "cudaErrorMemoryAllocation"))
                  or "out of memory" in err.lower())
        # stderr 全空 + 非零退出 = 进程被信号杀掉，连 traceback 都没来得及写。
        # Python 异常一定会留 traceback，所以这一支只可能是外部杀进程:
        # SIGKILL(9) 通常是宿主机/容器内存不足或驱动级 OOM。**这不是代码错误**,
        # 早先把它记成「❌崩溃」，等于把一个显存结论藏进了"环境有问题"里。
        rc = d.get("returncode")
        killed = (not err.strip()) and isinstance(rc, int) and rc < 0
        if d.get("peak_gb"):
            flag = f"{d['peak_gb']:.1f} GB"
        elif d.get("error") == "timeout":
            flag = "超时"
        elif is_oom:
            flag = "OOM"
        elif killed:
            flag = f"被杀 signal {-rc}" + ("（多半是显存/内存耗尽）" if -rc == 9 else "")
        else:
            flag = "❌崩溃"          # 不是显存问题，是代码/环境错了
        warn = ""
        if gc and d.get("ok") and not d.get("grad_ckpt_active"):
            warn = "  ⚠️ 梯度检查点未生效"
        if d.get("error") and d["error"] != "timeout":
            # 取**末尾**：异常类型与消息在 traceback 最后一行，
            # 取开头只会拿到中间的文件路径，最没有信息量的那段
            tail = " ".join(err.strip().splitlines()[-1:])[:120]
            warn += f"  [{tail}]"
        print(f"tok={d['n_visual_tokens']:>4}  {flag}{warn}")

    suffix = (f"_batch_K{args.K}" if args.sweep_batch else "")
    suffix += ("_frozen" if args.freeze_vision else "")
    suffix += (f"_cam{args.cameras}" if args.cameras != 1 else "")
    out = args.out.replace(".json", f"{suffix}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\n-> {out}")

    if args.sweep_batch:
        ok = [d for d in results if d.get("ok")]
        if ok:
            best = max(ok, key=lambda d: (d["batch"], not d["grad_ckpt"]))
            print(f"\n=== K={args.K} keep={args.keep or N_PATCH} 的批量建议 ===")
            print(f"  batch={best['batch']} "
                  f"{'+ 梯度检查点' if best['grad_ckpt'] else '(不开检查点)'}"
                  f"  峰值 {best['peak_gb']} GB")
            print(f"  梯度累积设为 {max(16 // best['batch'], 1)}，凑够官方的有效批 16")
        return

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
