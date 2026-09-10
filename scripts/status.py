#!/usr/bin/env python
"""
2×2 四格的进度一览 —— **一条命令看完全部**，不用逐臂拼 shell。

    python scripts/status.py

它回答的是每次都要问的那三个问题：跑到哪了、还活着吗、下一步敲什么。

⚠️ **判"跑完了"以 `adapter/step{max_steps}` 存在为准**，不是以 metrics 的最后一条
   为准：容器约每 22 小时重启一次（`docs/05` §13.3 ⑥），进程被杀时 metrics 停在
   哪里就是哪里，看着很像"跑完了"。三臂都在 27000 步附近断过一次。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ARMS = ("G3", "M3", "M2", "G4")
MAX_STEPS = 30_000
PE_AXES = {"G3": 3, "M3": 4, "M2": 3, "G4": 4}   # 与 wire.WireConfig.pe_axes 一致
DATA_ROOT = "/root/autodl-tmp/datasets/modified_libero_rlds"


def running_arms() -> dict:
    """正在跑的臂 → pid。从 ps 的完整命令行里抠 --arm。"""
    out = {}
    try:
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                            text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    for line in ps.splitlines():
        if "finetune_kframe" not in line or "ps -eo" in line:
            continue
        parts = line.split()
        if "--arm" in parts:
            out[parts[parts.index("--arm") + 1]] = parts[0]
    return out


def arm_dir(arm: str) -> Path | None:
    """runs/ 下属于这一臂的运行目录。exp_id 里带 arm 名与 --run_id_note。"""
    cands = [d for d in Path("runs").glob(f"*{arm}*sub255*") if d.is_dir()]
    return max(cands, key=lambda d: d.stat().st_mtime) if cands else None


def main() -> None:
    live = running_arms()
    rows, done_all = [], True
    for arm in ARMS:
        d = arm_dir(arm)
        if d is None:
            rows.append((arm, "—", "未开始", ""))
            done_all = False
            continue

        ck = sorted(int(x.name[4:]) for x in (d / "adapter").glob("step*")
                    if x.name[4:].isdigit()) if (d / "adapter").is_dir() else []
        finished = MAX_STEPS in ck

        rec = {}
        mp = d / "metrics.jsonl"
        if mp.is_file():
            lines = mp.read_text().strip().splitlines()
            if lines:
                rec = json.loads(lines[-1])

        step = rec.get("step", 0)
        if finished:
            state = "✅ 完成"
        elif arm in live:
            sps = rec.get("steps_per_sec") or 1e-9
            state = f"⏳ 跑中 pid={live[arm]}  剩 {(MAX_STEPS - step) / sps / 3600:.1f} h"
            done_all = False
        else:
            state = f"❌ 断了（最新 checkpoint step{max(ck) if ck else 0}）"
            done_all = False

        detail = ""
        if rec:
            detail = (f"loss {rec.get('loss', 0):.3f}  acc "
                      f"{rec.get('action_accuracy', 0):.3f}  "
                      f"{1 / (rec.get('steps_per_sec') or 1e-9):.2f}s/步")
            if "cg_anon_gb" in rec:
                # ⚠️ 只看 anon。cg_used 含页缓存，会贴着限额，那是正常的（§13.3 ⑦）
                detail += f"  anon {rec['cg_anon_gb']:.0f}GB"
        rows.append((arm, f"{step}/{MAX_STEPS}", state, detail))

    w = max(len(r[2]) for r in rows)
    print(f"\n{'臂':<4}{'PE':<5}{'进度':<14}{'状态':<{w + 2}}")
    print("─" * (w + 40))
    for arm, prog, state, detail in rows:
        print(f"{arm:<4}{PE_AXES[arm]:<5}{prog:<14}{state:<{w + 2}}{detail}")

    print()
    if done_all:
        print("四格全部训完 → 下一步是四臂满量评测（各 2.6 h）与配对分析。")
        return

    # 下一步该动哪一臂：先补断掉的（丢的步数最少），再开没起过的
    def finished(a: str) -> bool:
        d = arm_dir(a)
        return d is not None and (d / "adapter" / f"step{MAX_STEPS}").is_dir()

    broken = [a for a in ARMS
              if arm_dir(a) is not None and a not in live and not finished(a)]
    fresh = [a for a in ARMS if arm_dir(a) is None]
    todo = (broken or fresh or [None])[0]
    if live:
        print(f"（{', '.join(live)} 在跑，等它。）")
    if todo and todo not in live:
        resume = " --resume_from auto" if (arm_dir(todo)
                                           and (arm_dir(todo) / "adapter").is_dir()) else ""
        print(f"下一步 —— {todo}{'（续训）' if resume else '（首次）'}：\n")
        print(f"  nohup python scripts/finetune_kframe.py --arm {todo} "
              f"--max_steps {MAX_STEPS} --run_id_note sub255 \\\n"
              f"    --micro 8 --save_steps 1000{resume} \\\n"
              f"    --data_root_dir {DATA_ROOT} \\\n"
              f"    > results/logs/{todo}.log 2>&1 &\n  disown\n")
        print(f"  起来后确认：grep -E '✓ 形状|PE |留存率' results/logs/{todo}.log")
        print(f"  {todo} 必须打出 **PE {PE_AXES[todo]} 轴** —— "
              "错配臂与它的同池化伙伴在日志里其余部分完全一样。")


if __name__ == "__main__":
    sys.exit(main())
