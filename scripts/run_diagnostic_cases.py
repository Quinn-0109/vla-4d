#!/usr/bin/env python
"""按冻结 manifest 启动一个臂的六局诊断；不训练、不产生成功率结论。"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.diagnostic_dump import load_case_manifest  # noqa: E402
from common.runs import list_adapters, resolve_adapter  # noqa: E402

ARMS = ("G0", "G2", "G3")
DEFAULT_MANIFEST = ROOT / "results/cases/g3_fixed_diagnostic_cases.json"


def command(arm: str, adapter: Path, manifest: Path, run_note: str,
            python: str = sys.executable) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"arm 必须是 {ARMS}")
    return [
        python, str(ROOT / "scripts/run_eval_kframe.py"),
        "--arm", arm,
        "--K", "1" if arm == "G0" else "8",
        "--stride", "16",
        "--budget", "256",
        "--n_t", "2",
        "--partition", "quantile_fixed",
        "--adapter", str(adapter),
        "--task_suite_name", "libero_10",
        "--seed", "7",
        "--center_crop", "True",
        "--vision_cache", "True",
        "--eval_batch", "1",
        "--verify_batch", "1",
        "--dump_traj", "True",
        "--case_manifest", str(manifest),
        "--dump_dir", "results/cases",
        "--dump_frames", "True",
        "--dump_frame_every", "1",
        "--run_note", run_note,
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--adapter", required=True,
                   help="必须给确切 adapter/stepN；不存在时会列出所有可用 checkpoint")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--run_note", default="diag-cases-v1")
    p.add_argument("--run_root", default="runs")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    manifest = args.manifest.expanduser().resolve()
    load_case_manifest(manifest, ROOT)
    try:
        adapter = resolve_adapter(args.adapter, args.run_root).resolve()
    except SystemExit as err:
        print(err, file=sys.stderr)
        have = list_adapters(args.run_root)
        if have:
            print("\n请从上面清单复制与该 arm、step30000 对应的精确路径。", file=sys.stderr)
        raise
    cmd = command(args.arm, adapter, manifest, args.run_note)
    print("机制诊断与演示材料；固定六局、batch=1，不是成功率评测：")
    print(shlex.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
