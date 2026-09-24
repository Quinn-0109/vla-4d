#!/usr/bin/env python
"""阶段 C 验收 + 阶段 D1 诊断表：读固定案例目录，一条命令出表。不需要 GPU。

    python scripts/diagnose_cases.py                       # 自动读 results/cases/ 下全部已发布目录
    python scripts/diagnose_cases.py results/cases/EVAL-...G3... results/cases/EVAL-...G2...
    python scripts/diagnose_cases.py --check_only          # 只做 C2 验收，不写表

输出（默认 results/tables/，均在 .gitignore 放行范围内）：
  diagnostic_cases.md              验收结果 + 三张表 + 列定义，可直接贴进实验记录
  diagnostic_cases_episodes.csv    逐局
  diagnostic_cases_by_history.csv  按有效帧数 1..K 与本次成败分组
  diagnostic_cases_frame_profile.csv  满历史步上按距最新帧的位置分组的 token 份额
  diagnostic_cases_summary.json    输入目录、commit、权重与 manifest 哈希、验收问题

验收有问题时仍会写表（便于排查），但退出码为 1。
资源：三种方法、每种六局、最长 520 步时，读入约需 1.5 GB 内存、半分钟（实测于合成数据）。
⚠️ 这是机制诊断与演示材料，不更新成功率、不做显著性检验。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.case_report import (BOUNDARY, check_runs, discover,  # noqa: E402
                                  load_run, write_report)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="*", type=Path, help="案例目录；不给则扫描 --root")
    p.add_argument("--root", type=Path, default=ROOT / "results/cases")
    p.add_argument("--out", type=Path, default=ROOT / "results/tables")
    p.add_argument("--prefix", default="diagnostic_cases")
    p.add_argument("--check_only", action="store_true")
    args = p.parse_args()

    partial = []
    dirs = list(args.dirs)
    if not dirs:
        dirs, partial = discover(args.root)
    if not dirs:
        print(f"没有找到已发布的案例目录（{args.root}）", file=sys.stderr)
        return 2
    runs = [load_run(d) for d in dirs]
    problems = check_runs(runs)

    print(BOUNDARY)
    for r in runs:
        eps = {(t["task_id"], t["episode"]) for t in r["trajectory"]}
        print(f"  {r['arm']}: {r['dir'].name}  {len(eps)} 局 / {len(r['trajectory'])} 步")
    for p_ in partial:
        print(f"  ⚠️ 未完成目录（未读取）: {p_.name}")
    if problems:
        print("验收问题：")
        for msg in problems:
            print(f"  - {msg}")
    else:
        print("验收通过。")
    if not args.check_only:
        paths = write_report(runs, problems, partial, args.out, args.prefix)
        for k, v in paths.items():
            print(f"  {k}: {v}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
