"""
把 LoRA adapter 合并进基座权重，产出可直接评测的完整 checkpoint。

    python scripts/merge_lora.py --adapter runs/<exp>/adapter --out runs/<exp>/merged

从训练里拆出来单独做，是因为官方 finetune.py 每次存 checkpoint 都要合并一次:
每份 15 GB，写盘慢且很快吃光磁盘。训练只存几百 MB 的 adapter，
需要评测时再合并一次即可。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="openvla/openvla-7b")
    args = ap.parse_args()

    base = AutoModelForVision2Seq.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out)

    # 评测要用到 processor 与 dataset_statistics，一并带上，否则
    # run_libero_eval_traj.py 会因为找不到 norm_stats 而断言失败
    src = Path(args.adapter).parent
    for fn in ("dataset_statistics.json",):
        if (src / fn).exists():
            (out / fn).write_bytes((src / fn).read_bytes())
    try:
        AutoProcessor.from_pretrained(src, trust_remote_code=True).save_pretrained(out)
    except Exception:
        AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(out)

    print(f"合并完成 -> {out}")
    print(f"评测: bash scripts/run_eval.sh single libero_spatial 50 7 False")
    print(f"      （把 CKPT 里的路径改成 {out}）")


if __name__ == "__main__":
    main()
