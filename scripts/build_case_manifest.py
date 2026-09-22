#!/usr/bin/env python
"""从正式逐局记录中冻结机制诊断案例；纯标准库，不需要 GPU。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path) -> list[dict]:
    rows, seen = [], set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if (type(row.get("task_id")) is not int or type(row.get("episode")) is not int
                or row.get("success") not in (0, 1)):
            raise ValueError(f"{path}:{lineno}: 非法逐局记录")
        key = (row["task_id"], row["episode"])
        if key in seen:
            raise ValueError(f"{path}:{lineno}: 重复键 {key}")
        seen.add(key)
        rows.append(row)
    return rows


def build(path: Path, tasks: list[int], expected_arm: str = "G3") -> dict:
    if not tasks or len(set(tasks)) != len(tasks) or any(type(t) is not int or t < 0 for t in tasks):
        raise ValueError("tasks 必须是互不重复的非负整数")
    rows = load_rows(path)
    if not rows:
        raise ValueError("逐局记录为空")
    arms = {row.get("arm") for row in rows}
    if arms != {expected_arm}:
        raise ValueError(f"正式记录应只有 arm={expected_arm}，实际为 {sorted(map(str, arms))}")

    cases = []
    for task in tasks:
        task_rows = [r for r in rows if r["task_id"] == task]
        for success, label in ((1, "success"), (0, "failure")):
            candidates = sorted((r for r in task_rows if r["success"] == success),
                                key=lambda r: r["episode"])
            if not candidates:
                raise ValueError(f"task {task} 没有 {label} 案例")
            row = candidates[0]
            cases.append({"task_id": task, "episode": row["episode"],
                          "g3_success": success, "stratum": label})

    return {
        "schema_version": 1,
        "purpose": "mechanism_diagnosis_and_demo_only",
        "statistical_use": "forbidden",
        "source": {
            "path": path.as_posix(),
            "sha256": sha256(path),
            "arm": expected_arm,
            "rows": len(rows),
            "unique_keys": len({(r["task_id"], r["episode"]) for r in rows}),
            "successes": sum(int(r["success"]) for r in rows),
        },
        "selection": {
            "tasks": tasks,
            "rule": "lowest episode id in each (task_id, g3_success) stratum",
            "chosen_before_new_inference": True,
        },
        "cases": cases,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", type=int, nargs="+", default=[2, 4, 5])
    p.add_argument("--arm", default="G3")
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖已冻结 manifest: {args.output}")
    doc = build(args.episodes, args.tasks, args.arm)
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(f"已冻结 {len(doc['cases'])} 个案例 -> {args.output}")


if __name__ == "__main__":
    main()
