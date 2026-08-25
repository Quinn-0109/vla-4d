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

    # ⚠️ dataset_statistics.json 必须一并带上。
    #    openvla_utils.py:60-64: checkpoint 目录里有这个文件时会**整体替换**
    #    vla.norm_stats。没有它，模型带的是基座 openvla-7b 的 OXE 统计量，
    #    里面没有 libero_*_no_noops，评测会在 unnorm_key 断言处失败；
    #    更糟的情况是用错统计量做动作反归一化，静默产出垃圾动作。
    #
    #    它由 finetune_single.py 写在 run_dir，而 adapter 存在 run_dir/adapter/stepN，
    #    所以要向上找两层。
    adapter = Path(args.adapter)
    stats = next((d / "dataset_statistics.json"
                  for d in (adapter, adapter.parent, adapter.parent.parent)
                  if (d / "dataset_statistics.json").exists()), None)
    if stats is None:
        raise SystemExit(
            f"❌ 在 {adapter} 及其上两层都找不到 dataset_statistics.json。\n"
            f"   没有它评测会用错动作反归一化统计量。请确认训练是否正常完成。")
    (out / "dataset_statistics.json").write_bytes(stats.read_bytes())
    print(f"动作统计量: {stats}")

    for d in (adapter.parent.parent, adapter.parent, adapter):
        try:
            AutoProcessor.from_pretrained(d, trust_remote_code=True).save_pretrained(out)
            break
        except Exception:
            continue
    else:
        AutoProcessor.from_pretrained(args.base, trust_remote_code=True).save_pretrained(out)

    print(f"合并完成 -> {out}")
    print("\n评测（5 trials/task，约 17 分钟，可与满量 500 局的基线子集直接对比）:")
    print(f"  python scripts/run_libero_eval_traj.py \\\n"
          f"    --pretrained_checkpoint {out} \\\n"
          f"    --task_suite_name libero_spatial --center_crop True \\\n"
          f"    --num_trials_per_task 5 --save_video False \\\n"
          f"    --run_id_note ft-$(basename {args.adapter})")


if __name__ == "__main__":
    main()
