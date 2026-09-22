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

# Windows 的默认 GBK 终端不能编码日志里的 ✓/⚠️。保留终端编码，只把无法编码的
# 字符替换掉，避免只读状态命令在打印到一半时崩溃；UTF-8 终端不受影响。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

ARMS = ("G3", "M3", "M2", "G4")
MAX_STEPS = 30_000
N_TASKS = 10                                      # libero_10
PE_AXES = {"G3": 3, "M3": 4, "M2": 3, "G4": 4}   # 与 wire.WireConfig.pe_axes 一致
DATA_ROOT = "/root/autodl-tmp/datasets/modified_libero_rlds"


def completed_mainline_status() -> bool:
    """正式修正版 G3 已归档时，显示当前结项路线而非历史四臂启动命令。"""
    root = Path("results/logs")
    logs = sorted(root.glob("EVAL-*-G3-*--fixed-main.txt")) if root.is_dir() else []
    for log in reversed(logs):
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        final = next((line for line in reversed(lines) if line.startswith("FINAL ")), None)
        episodes = log.with_suffix(".episodes.jsonl")
        meta = log.with_suffix(".meta.json")
        if final is None or not episodes.is_file() or not meta.is_file():
            continue
        rows = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        keys = {(r.get("task_id"), r.get("episode")) for r in rows}
        successes = sum(int(r.get("success", 0)) for r in rows)
        if len(rows) != 500 or len(keys) != 500:
            print(f"正式 G3 记录不完整：rows={len(rows)} unique={len(keys)}，请先审计 {episodes}")
            return True
        print("\n当前主线：修正版 G3 已完成训练与正式评测")
        print(f"  {final}")
        print(f"  逐局记录 500/500，成功 {successes}/500")
        print(f"  来源元数据 {meta}")
        print("\n下一步：按 docs/09-当前任务执行清单.md 完成固定案例、分配诊断和结项演示。")
        print("不要重新启动历史 2×2 四臂训练。")
        return True
    return False


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
    if completed_mainline_status():
        return
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
        fulldata_status()
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
    fulldata_status()


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

        # 只允许同一运行的分段合并；不同 step/batch/note 不按修改时间混合。
        groups = {re.sub(r"-t\d+_-?\d+(?=--|\.txt$)", "", f.name) for f in cand}
        if len(groups) != 1:
            print(f"  {arm:<4}⚠️ 有 {len(groups)} 组评测配置，拒绝自动合并；请显式选择来源")
            continue

        # ⚠️ **分段跑会写到多个文件**（run_id 里带 -t{start}_{end}），
        #    每个各自 PARTIAL。只看最新那一个，会把"两段合起来已经跑完"
        #    误报成"断在第 5 个 task"，并给出一个**错误的续跑位置**。
        #    所以仅合并同一运行的不重叠分段；重复 task 拒绝汇总。
        seen, whole, duplicate = {}, None, False
        for f in sorted(cand, key=lambda x: x.stat().st_mtime):
            txt = f.read_text(errors="ignore").splitlines()
            for line in txt:
                m = re.match(r"task (\d+) .*?: (\d+)/(\d+) =", line)
                if m:
                    if int(m.group(1)) in seen:
                        duplicate = True
                    seen[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), f.name)
            for line in txt:
                if line.startswith("FINAL"):
                    whole = line
        if duplicate or any(t not in range(N_TASKS) or v[1] != 50 or not 0 <= v[0] <= v[1]
                            for t, v in seen.items()):
            print(f"  {arm:<4}⚠️ 聚合记录重复或不是每任务 50 局，拒绝自动汇总")
            continue
        # 逐局 JSONL 是配对检验（McNemar）的唯一输入，聚合之后恢复不出来。
        ep = list(Path("results/logs").glob(f"EVAL-*-{arm}-*--sub255.episodes.jsonl"))
        tag = ("  有逐局文件（完整性与来源须用 --manifest 核验）" if ep
               else "  未发现 sub255 逐局文件（检查来源或补评）")
        if whole and set(seen) == set(range(N_TASKS)):
            final[arm] = whole
            print(f"  {arm:<4}✅ {whole}{tag}")
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
        print("聚合日志已覆盖四臂；配对分析须指定 --manifest，不能据此认定逐局数据齐全。")
        print("⚠️ 主分析是 **9-task**（事前登记），10-task 只供与文献并列。")
    else:
        print("⚠️ 判据数 = 覆盖全部 10 个 task 的成绩（单段 FINAL 或多段并集）；"
              "单个 PARTIAL 不是。")


def fulldata_status() -> None:
    """
    G3 全量数据（`--no_subset`，exp_id 带 `+fulldata`）。

    ⚠️ **它不属于 2×2**（`docs/05` §13.8）：纪律 1b 要求四格看同样的数据，
       它与 M3/M2/G4 的差里混着数据量。合法用途只有两个 ——
       与全量训练的 G2 比粒度、与 G3-subset 比纯数据量效应。
       所以单列一节，绝不混进上面那张表。
    """
    runs = [d for d in Path("runs").glob("*fulldata*") if d.is_dir()]
    logs = ([f for f in Path("results/logs").glob("EVAL-*.txt")
             if "fullG3" in f.name] if Path("results/logs").is_dir() else [])
    if not runs and not logs:
        return
    print("\n" + "─" * 52)
    print("G3 全量数据（**不属于 2×2**，`docs/05` §13.8 事前登记）")
    for d in runs:
        ck = sorted(int(x.name[4:]) for x in (d / "adapter").glob("step*")
                    if x.name[4:].isdigit()) if (d / "adapter").is_dir() else []
        rec = {}
        mp = d / "metrics.jsonl"
        if mp.is_file() and mp.read_text().strip():
            rec = json.loads(mp.read_text().strip().splitlines()[-1])
        step = rec.get("step", 0)
        if MAX_STEPS in ck:
            print(f"  训练 ✅ 完成")
        else:
            sps = rec.get("steps_per_sec") or 1e-9
            print(f"  训练 {step}/{MAX_STEPS}  剩 {(MAX_STEPS - step) / sps / 3600:.1f} h"
                  f"  （断了就加 --resume_from auto 重跑同一条命令）")
    for f in logs:
        fin = [l for l in f.read_text(errors="ignore").splitlines()
               if l.startswith(("FINAL", "PARTIAL"))]
        print(f"  评测 {fin[-1] if fin else '跑中'}")
        if fin and fin[-1].startswith("FINAL"):
            try:
                sr = float(fin[-1].split("success_rate=")[1].split()[0])
            except (IndexError, ValueError):
                continue
            print(f"       ⚠️ 判读要用 **9-task**（排除 task 1），"
                  f"10-task 的 {sr:.2%} 不是判据数")
    print("  判读表（事前写死，看到数字后不得移动边界）：")
    print("    ≥38%  优先检查数据子集的贡献；不能据此证明等价")
    print("    33–38%  归因仍未定，不作精确比例分解")
    print("    ≤33%  优先检查跨帧压缩实现与信息损失")
    print("    以上是资源决策阈值；结论需差值区间，单次训练不覆盖训练种子方差。")


if __name__ == "__main__":
    sys.exit(main())
