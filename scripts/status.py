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
import re
import subprocess
import sys
from pathlib import Path

ARMS = ("G3", "M3", "M2", "G4")
MAX_STEPS = 30_000
N_TASKS = 10                                      # libero_10
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
        eval_status()
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


def eval_status() -> None:
    """
    训练全绿之后看评测。日志是 `results/logs/EVAL-<suite>-<arm>-...txt`，
    每个 task 一行、结尾一行 FINAL 或 PARTIAL。

    ⚠️ **PARTIAL 不是判据数**（脚本自己也这么写）：它是只跑了部分 task 的合计，
       判据认的是满 10 task。这里照原样显示，不把它当成绩。
    """
    live_eval = None
    try:
        ps = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                            text=True, timeout=10).stdout
        for line in ps.splitlines():
            if "run_eval_kframe" in line and "ps -eo" not in line:
                parts = line.split()
                a = parts[parts.index("--arm") + 1] if "--arm" in parts else "?"
                live_eval = (a, parts[0])
    except (OSError, subprocess.SubprocessError):
        pass

    print("四格全部训完。评测：\n")
    logs = (sorted(Path("results/logs").glob("EVAL-*.txt"))
            if Path("results/logs").is_dir() else [])
    final = {}                      # arm → 满 10 task 的成绩
    for arm in ARMS:
        cand = [f for f in logs if f"-{arm}-" in f.name and "sub255" in f.name]
        if not cand:
            print(f"  {arm:<4}" + ("⏳ 跑中（还没写日志）"
                                   if live_eval and live_eval[0] == arm else "— 未评"))
            continue

        # ⚠️ **分段跑会写到多个文件**（run_id 里带 -t{start}_{end}），
        #    每个各自 PARTIAL。只看最新那一个，会把"两段合起来已经跑完"
        #    误报成"断在第 5 个 task"，并给出一个**错误的续跑位置**。
        #    所以按 task_id 取并集：同一个 task 出现在多个文件里时以最新的为准。
        seen, whole = {}, None
        for f in sorted(cand, key=lambda x: x.stat().st_mtime):
            txt = f.read_text(errors="ignore").splitlines()
            for line in txt:
                m = re.match(r"task (\d+) .*?: (\d+)/(\d+) =", line)
                if m:
                    seen[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), f.name)
            for line in txt:
                if line.startswith("FINAL"):
                    whole = line
        if whole:                                   # 一次整跑到底
            final[arm] = whole
            print(f"  {arm:<4}✅ {whole}")
            continue

        ok = sum(v[0] for v in seen.values())
        n = sum(v[1] for v in seen.values())
        miss = [t for t in range(N_TASKS) if t not in seen]
        nseg = len({v[2] for v in seen.values()})
        seg = f"（{nseg} 段）" if nseg > 1 else ""
        if not miss:
            # 各段加起来已覆盖全部 task —— 这是判据数，脚本自己不会打 FINAL
            final[arm] = f"合并 {ok}/{n} = {ok / max(n, 1):.4f}{seg}"
            print(f"  {arm:<4}✅ {final[arm]}  ← 分段合并，非单个 FINAL")
        elif live_eval and live_eval[0] == arm:
            print(f"  {arm:<4}⏳ 跑中 pid={live_eval[1]}  "
                  f"{len(seen)}/{N_TASKS} task  {ok}/{n}{seg}")
        else:
            # 续跑位置取**缺口的第一个**，不是"已完成个数"——分段之后两者不同
            print(f"  {arm:<4}❌ 断了  {len(seen)}/{N_TASKS} task{seg}  "
                  f"缺 {miss}  → 补跑 --start_task {miss[0]}"
                  + (f" --end_task {miss[-1] + 1}" if len(miss) > 1 else ""))

    print()
    if len(final) == len(ARMS):
        print("四臂结果齐了 → 配对（McNemar）分析，输入是 *.episodes.jsonl。")
        print("⚠️ 主分析是 **9-task**（事前登记），10-task 只供与文献并列。")
    else:
        print("⚠️ 判据数 = 覆盖全部 10 个 task 的成绩（单段 FINAL 或多段并集）；"
              "单个 PARTIAL 不是。")


if __name__ == "__main__":
    sys.exit(main())
