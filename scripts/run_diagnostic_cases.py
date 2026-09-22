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
DEFAULT_CHECKPOINTS = ROOT / "results/cases/diagnostic_checkpoints.json"


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
        "--dump_traj", "1",
        "--case_manifest", str(manifest),
        "--dump_dir", "results/cases",
        "--dump_frames", "True",
        "--dump_frame_every", "1",
        "--run_note", run_note,
    ]


def registered_adapter(arm: str, path: Path) -> str:
    import json
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("schema_version") != 1
            or doc.get("purpose") != "fixed_case_diagnostic_checkpoint_identity"):
        raise ValueError("不是受支持的诊断 checkpoint 清单")
    try:
        row = doc["arms"][arm]
    except KeyError as err:
        raise ValueError(f"checkpoint 清单没有 {arm}") from err
    digest = row.get("adapter_sha256", "")
    if len(digest) != 64 or row.get("adapter_weight_bytes") != 162015576:
        raise ValueError(f"{arm} 的 checkpoint 身份字段不完整")
    return row["adapter"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--adapter", help="可选覆盖；默认读取已冻结的 checkpoint 清单")
    p.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--run_note", default="diag-cases-v1")
    p.add_argument("--run_root", default="runs")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    manifest = args.manifest.expanduser().resolve()
    load_case_manifest(manifest, ROOT)
    try:
        adapter_arg = args.adapter or registered_adapter(args.arm, args.checkpoints)
        adapter = resolve_adapter(adapter_arg, args.run_root).resolve()
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
