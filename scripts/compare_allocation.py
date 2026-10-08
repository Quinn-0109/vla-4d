#!/usr/bin/env python
"""G2 / G3 / G3Q 的离线分配对照：真实帧数 1..8 时各方法怎样分 token。不需要 GPU 与模型。

    python scripts/compare_allocation.py

走与训练、评测完全相同的 `wire._pool_and_coords`，统计口径与固定案例诊断表相同
（`analysis.alloc_stats.summarize_assignment`）。网格方法的分配只由真实帧数决定、
与画面内容无关，所以用随机特征即可得到确定的分配。

若 `results/tables/diagnostic_cases_by_history.csv`（10-08 真实诊断）存在，G2 与 G3
两行会逐项与之对拍：离线结果必须与真实运行一致，否则这张对照表不可信，退出码为 1。

输出：results/tables/allocation_g2_g3_g3q.md 与 .csv
"""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("G2", "G3", "G3Q")
STRIDE = 16


def allocation_rows(k: int = 8) -> list[dict]:
    import torch
    from analysis.alloc_stats import summarize_assignment
    from pooling.wire import N_PATCH, WireConfig, _Batch, _pool_and_coords

    torch.manual_seed(0)
    emb = torch.randn(1, k * N_PATCH, 8)
    steps = [s * STRIDE for s in range(k)]
    rows = []
    for arm in ARMS:
        cfg = WireConfig(arm=arm, K=k)
        for real in range(1, k + 1):
            mask = (torch.arange(k) >= k - real).unsqueeze(0)
            sink = {}
            _pool_and_coords(emb, cfg, _Batch(frame_pad_mask=mask), sink=sink)
            st = summarize_assignment(sink["assign"][0].tolist(), mask[0].tolist(), steps, N_PATCH)
            lat = [sl["latest_frame_weight"] for sl in st["slots"] if sl["latest_frame_weight"] > 0]
            rows.append({
                "arm": arm, "real_frames": real,
                "tokens_used": st["slots_used"],
                "latest_slots": len(lat),
                "latest_purity": round(sum(lat) / len(lat), 4),
                "latest_share": round(st["per_frame"][st["latest_real_frame"]]["slot_weight_sum"], 4),
                "latest_exclusive": st["latest_frame_exclusive_slots"],
                "mixed_fraction": round(st["mixed_frame_slot_fraction"], 4),
                "step_span_mean": round(st["mean_step_span"], 2),
                "dropped_valid_patches": st["valid_dropped_patches"],
            })
    return rows


def cross_check(rows: list[dict], path: Path) -> list[str]:
    """离线 G2/G3 与真实诊断逐项比较；返回不一致清单。"""
    if not path.is_file():
        return []
    real = {}
    for r in csv.DictReader(open(path, encoding="utf-8")):
        real[(r["arm"], int(r["real_frames"]))] = r
    problems = []
    for r in rows:
        ref = real.get((r["arm"], r["real_frames"]))
        if ref is None:
            continue
        for ours, theirs in (("latest_slots", "latest_slots_mean"),
                             ("latest_purity", "latest_purity_mean"),
                             ("mixed_fraction", "mixed_fraction_mean")):
            if abs(float(r[ours]) - float(ref[theirs])) > 1e-3:
                problems.append(f"{r['arm']} 真实帧 {r['real_frames']}：{ours} 离线 {r[ours]}，真实 {ref[theirs]}")
    return problems


def markdown(rows: list[dict], problems: list[str], checked: bool) -> str:
    cols = [("arm", "方法"), ("real_frames", "真实帧"), ("tokens_used", "使用 token"),
            ("latest_slots", "最新帧所在"), ("latest_purity", "最新帧纯度"),
            ("latest_share", "最新帧份额"), ("latest_exclusive", "最新帧独占"),
            ("mixed_fraction", "跨帧混合"), ("step_span_mean", "平均跨度（步）")]
    lines = ["# G2 / G3 / G3Q 离线分配对照", "",
             "> 由 `scripts/compare_allocation.py` 生成，走训练与评测同一份池化代码；"
             "网格方法的分配只由真实帧数决定。这是设计检查，不是实验结果。", ""]
    if checked:
        lines.append("与 10-08 真实诊断对拍：" + ("G2、G3 全部一致。" if not problems
                                              else "**不一致**：" + "；".join(problems)))
        lines.append("")
    lines += ["| " + " | ".join(t for _, t in cols) + " |",
              "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r[k]) for k, _ in cols) + " |")
    lines += ["", "定义与固定案例诊断表相同：纯度是最新帧在含它的 token 里所占比例的平均；"
              "份额按 patch 计数做分数归属；跨帧混合是含两个及以上真实帧的 token 占已用 token 的比例。", ""]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=ROOT / "results/tables")
    p.add_argument("--real", type=Path, default=ROOT / "results/tables/diagnostic_cases_by_history.csv")
    args = p.parse_args()
    rows = allocation_rows()
    problems = cross_check(rows, args.real)
    args.out.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    (args.out / "allocation_g2_g3_g3q.csv").write_text(buf.getvalue(), encoding="utf-8")
    md = markdown(rows, problems, args.real.is_file())
    (args.out / "allocation_g2_g3_g3q.md").write_text(md, encoding="utf-8")
    print(md)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
