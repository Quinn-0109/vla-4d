"""固定案例诊断目录 → 验收检查与统计表（阶段 C 的验收、阶段 D1 的表）。

输入是 `run_eval_kframe.py --case_manifest ...` 发布的案例目录（每个方法一个），
由 `diagnostic_dump.read_case_dump` 读取并校验逐步对齐。本模块只做两件事：

1. **验收**（`docs/09` C2）：每个目录恰好包含 manifest 的那几局、元数据齐全、
   引用的关键帧存在、数值全部有限、动作步连续；各方法用的是同一份 manifest。
   参考标签（正式 b8）与本次观察（b1）不一致时**记录，不报错**。
2. **汇总**（`docs/09` D1）：逐局表、按有效帧数分组的表、按帧距离的贡献剖面。

纯标准库，不依赖 torch / numpy / 仿真器，可在任何机器上离线运行。

⚠️ 这些表是**机制诊断与演示材料**：每个方法只有六局，不更新成功率、不计算
   总体置信区间、不做显著性检验。输出的 Markdown 开头会写明这一点。

列的定义（与 `alloc_stats.summarize_assignment` 一致）：

- 最新帧所在 token 数 `latest_slots`：含至少一个最新帧 patch 的输出 token 数。
- 最新帧纯度 `latest_purity`：只在这些 token 上，最新帧 patch 所占比例的平均。
  1.0 表示最新帧没有被旧帧稀释；0.25 表示每个含最新帧的 token 里它平均只占四分之一。
- 最新帧 token 份额 `latest_share`：按 patch 计数做分数归属时最新帧分到的 token 数
  （各帧份额之和 = 实际使用的 token 数；`latest_share = latest_slots × latest_purity`）。
  这是一个**定义上的选择**，随结论一同报告。
- 全槽平均权重 `latest_weight`：最新帧比例在**全部**已用 token 上的平均（导出里的原字段）。
  ⚠️ 它把“分到的 token 少”和“被旧帧稀释”混在一起：帧独立池化里最新帧独占 32 个 token、
  纯度 1.0，全槽平均却只有 0.125，与跨帧池化里被三帧稀释的情形数值相同。
  所以表里以纯度与所在 token 数为主，这一列只保留在 CSV 里。
- 最新帧独占 token `latest_exclusive`：只由最新帧构成的 token 数。
- 跨帧混合比例 `mixed_fraction`：含两个及以上真实帧的 token 占实际使用 token 的比例。
- 时间跨度 `step_span`：一个 token 内最早与最晚来源帧的环境步差。
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

from analysis.diagnostic_dump import read_case_dump

BOUNDARY = ("机制诊断与演示材料：每个方法只有固定的六局，"
            "不用于更新成功率、不计算总体置信区间、不做显著性检验。")


# ---------------------------------------------------------------- 读取与验收
def discover(root: Path) -> tuple[list[Path], list[Path]]:
    """`root` 下已发布的案例目录，以及残留的 `.partial` 目录（后者只报告，不读取）。"""
    root = Path(root)
    if not root.is_dir():
        return [], []
    dirs = sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "meta.json").is_file() and not p.name.endswith(".partial"))
    partial = sorted(p for p in root.iterdir() if p.is_dir() and p.name.endswith(".partial"))
    return dirs, partial


def _finite(x) -> bool:
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return True
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return all(_finite(v) for v in x)
    return True


def load_run(path: Path) -> dict:
    doc = read_case_dump(path)
    meta = doc["meta"]
    arms = {r["arm"] for r in doc["trajectory"]} | {r["arm"] for r in doc["allocation"]}
    cfg_arm = (meta.get("config") or {}).get("arm")
    doc["dir"] = Path(path)
    doc["arm"] = cfg_arm if cfg_arm else (next(iter(arms)) if len(arms) == 1 else None)
    doc["row_arms"] = arms
    return doc


def check_run(run: dict) -> list[str]:
    """单个案例目录的验收问题清单；空表示通过。"""
    problems = []
    meta, traj, alloc = run["meta"], run["trajectory"], run["allocation"]
    name = run["dir"].name
    diag = meta.get("diagnostic_dump", {})
    cases = (diag.get("manifest") or {}).get("cases")
    if not cases:
        problems.append(f"{name}: meta 中没有 manifest 案例列表")
        cases = []
    if run["row_arms"] != {run["arm"]}:
        problems.append(f"{name}: 行内 arm {sorted(map(str, run['row_arms']))} 与配置 arm={run['arm']} 不一致")
    for key in ("config", "code"):
        if not meta.get(key):
            problems.append(f"{name}: meta 缺少 {key}")
    if not (meta.get("code") or {}).get("commit"):
        problems.append(f"{name}: meta 没有记录代码 commit")
    if meta.get("adapter") and not meta.get("adapter_sha256"):
        problems.append(f"{name}: 有 adapter 但没有权重哈希")

    expected = {(c["task_id"], c["episode"]) for c in cases}
    seen = {(r["task_id"], r["episode"]) for r in traj}
    if seen - expected:
        problems.append(f"{name}: manifest 之外的局 {sorted(seen - expected)}")
    if expected - seen:
        problems.append(f"{name}: 缺少 manifest 中的局 {sorted(expected - seen)}")

    by_ep = defaultdict(list)
    for r in traj:
        by_ep[(r["task_id"], r["episode"])].append(r)
    for key, rows in sorted(by_ep.items()):
        steps = sorted(r["env_step"] for r in rows)
        if steps != list(range(len(steps))):
            problems.append(f"{name}: task {key[0]} ep {key[1]} 的动作步不连续（应为 0..{len(steps) - 1}）")
        rows = sorted(rows, key=lambda r: r["env_step"])
        if any(r["success"] for r in rows[:-1]):
            problems.append(f"{name}: task {key[0]} ep {key[1]} 在最后一步之前已标记成功")
        for r in rows:
            if "g3_reference_success" not in r:
                problems.append(f"{name}: task {key[0]} ep {key[1]} 缺少正式参考标签")
                break
        for r in rows:
            frame = r.get("frame")
            if frame and not (run["dir"] / frame).is_file():
                problems.append(f"{name}: 缺少关键帧 {frame}")
                break
    for kind, rows in (("trajectory", traj), ("alloc", alloc)):
        bad = [r for r in rows if not _finite(r)]
        if bad:
            r = bad[0]
            problems.append(f"{name}: {kind} 含非有限数值（首个在 task {r['task_id']} "
                            f"ep {r['episode']} step {r['env_step']}，共 {len(bad)} 行）")
    return problems


def check_runs(runs: list[dict]) -> list[str]:
    problems = []
    for run in runs:
        problems += check_run(run)
    arms = [r["arm"] for r in runs]
    dup = sorted({a for a in arms if arms.count(a) > 1})
    if dup:
        problems.append(f"同一方法出现多个案例目录 {dup}；请显式指定要用的目录")
    shas = {r["meta"].get("diagnostic_dump", {}).get("manifest_sha256") for r in runs}
    if len(shas) > 1:
        problems.append("各方法使用的 manifest 不同（SHA-256 不一致），不能放在同一张表里比较")
    return problems


# ---------------------------------------------------------------- 汇总
def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _latest_share(a: dict) -> float:
    return float(a["per_frame"][a["latest_real_frame"]]["slot_weight_sum"])


def _latest_slots(a: dict) -> int:
    return sum(1 for sl in a["slots"] if sl["latest_frame_weight"] > 0)


def _latest_purity(a: dict):
    ws = [sl["latest_frame_weight"] for sl in a["slots"] if sl["latest_frame_weight"] > 0]
    return sum(ws) / len(ws) if ws else None


def _alloc_cols(al: list[dict]) -> dict:
    """一组动作步上的分配指标（逐局表与按有效帧数表共用同一份定义）。"""
    return {
        "slots_used_mean": _mean(a["slots_used"] for a in al),
        "latest_slots_mean": _mean(_latest_slots(a) for a in al),
        "latest_purity_mean": _mean(x for x in (_latest_purity(a) for a in al) if x is not None),
        "latest_share_mean": _mean(_latest_share(a) for a in al),
        "latest_exclusive_mean": _mean(a["latest_frame_exclusive_slots"] for a in al),
        "mixed_fraction_mean": _mean(a["mixed_frame_slot_fraction"] for a in al),
        "step_span_mean": _mean(a["mean_step_span"] for a in al),
        "latest_weight_mean": _mean(a["latest_frame_mean_slot_weight"] for a in al),
    }


def _episode_actions(rows: list[dict]) -> dict:
    rows = sorted(rows, key=lambda r: r["env_step"])
    deltas = []
    for prev, cur in zip(rows, rows[1:]):
        a, b = prev["raw_action"][:6], cur["raw_action"][:6]
        deltas.append(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))
    grip = [r["executed_action"][6] for r in rows if len(r["executed_action"]) > 6]
    flips = sum(1 for x, y in zip(grip, grip[1:]) if (x > 0) != (y > 0))
    return {"action_change_mean": _mean(deltas), "gripper_flips": flips}


def episode_table(runs: list[dict]) -> list[dict]:
    out = []
    for run in runs:
        traj = defaultdict(list)
        for r in run["trajectory"]:
            traj[(r["task_id"], r["episode"])].append(r)
        alloc = defaultdict(list)
        for a in run["allocation"]:
            alloc[(a["task_id"], a["episode"])].append(a)
        for key in sorted(traj):
            rows = sorted(traj[key], key=lambda r: r["env_step"])
            al = alloc[key]
            observed = int(rows[-1]["success"])
            reference = int(rows[-1].get("g3_reference_success", -1))
            out.append({
                "arm": run["arm"],
                "task_id": key[0],
                "episode": key[1],
                "stratum": rows[-1].get("stratum"),
                "g3_reference_success": reference,
                "observed_success": observed,
                "matches_reference": observed == reference,
                "steps": len(rows),
                **_alloc_cols(al),
                "step_span_max": max((a["max_step_span"] for a in al), default=None),
                "valid_dropped_patches_total": sum(a["valid_dropped_patches"] for a in al),
                **_episode_actions(rows),
            })
    return out


def history_table(runs: list[dict]) -> list[dict]:
    """按有效帧数（1..K）与本次观察成败分组。历史不足阶段与满历史阶段的行为分开看。"""
    success = {}
    for run in runs:
        for r in run["trajectory"]:
            if r["success"]:
                success[(run["arm"], r["task_id"], r["episode"])] = 1
    groups = defaultdict(list)
    for run in runs:
        for a in run["allocation"]:
            outcome = "success" if success.get((run["arm"], a["task_id"], a["episode"])) else "failure"
            groups[(run["arm"], outcome, a["real_frames"])].append(a)
    out = []
    for (arm, outcome, real), al in sorted(groups.items()):
        out.append({"arm": arm, "observed_outcome": outcome, "real_frames": real,
                    "steps": len(al), **_alloc_cols(al)})
    return out


def frame_profile(runs: list[dict]) -> list[dict]:
    """满历史步上，距最新帧 d 个位置的那一帧平均分到多少 token（分数归属）。"""
    groups = defaultdict(list)
    for run in runs:
        for a in run["allocation"]:
            if a["real_frames"] != a["frames"]:
                continue
            latest = a["latest_real_frame"]
            for f in a["per_frame"]:
                if f["is_real"]:
                    groups[(run["arm"], latest - f["frame_index"])].append(f)
    return [{
        "arm": arm, "offset_from_latest": d, "steps": len(fs),
        "token_share_mean": _mean(f["slot_weight_sum"] for f in fs),
        "patch_retention_mean": _mean(f["patch_retention"] for f in fs),
    } for (arm, d), fs in sorted(groups.items())]


# ---------------------------------------------------------------- 输出
def _fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(t for _, t in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_fmt(r[k]) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def render_markdown(runs, problems, partial, episodes, history, profile) -> str:
    lines = ["# 固定案例机制诊断", "", f"> {BOUNDARY}", ""]
    lines += ["## 输入", ""]
    for run in runs:
        m = run["meta"]
        lines.append(f"- {run['arm']}：`{run['dir'].name}`，commit `{(m.get('code') or {}).get('commit')}`，"
                     f"adapter SHA-256 `{m.get('adapter_sha256')}`")
    sha = {r["meta"].get("diagnostic_dump", {}).get("manifest_sha256") for r in runs}
    lines += ["", f"manifest SHA-256：`{', '.join(sorted(map(str, sha)))}`", ""]
    lines += ["## 验收（docs/09 C2）", ""]
    if partial:
        lines.append(f"- 存在未完成的 `.partial` 目录（未读取）：{', '.join(p.name for p in partial)}")
    lines += [f"- 问题：{p}" for p in problems] or ["- 全部通过：每个目录恰好包含 manifest 的各局，"
                                                   "元数据、关键帧与数值检查无误。"]
    mismatch = [e for e in episodes if not e["matches_reference"]]
    if mismatch:
        lines.append("- 本次观察与正式 b8 参考标签不一致（记录，不替换案例）：" + "；".join(
            f"{e['arm']} task {e['task_id']} ep {e['episode']} 参考 {e['g3_reference_success']} → "
            f"本次 {e['observed_success']}" for e in mismatch))
    lines += ["", "## 逐局", "", _md_table(episodes, [
        ("arm", "方法"), ("task_id", "task"), ("episode", "ep"), ("stratum", "分层"),
        ("g3_reference_success", "参考"), ("observed_success", "本次"), ("steps", "步数"),
        ("latest_slots_mean", "最新帧所在"), ("latest_purity_mean", "最新帧纯度"),
        ("latest_share_mean", "最新帧份额"), ("latest_exclusive_mean", "最新帧独占"),
        ("mixed_fraction_mean", "跨帧混合"), ("step_span_mean", "平均跨度"), ("step_span_max", "最大跨度"),
        ("action_change_mean", "动作变化"), ("gripper_flips", "夹爪翻转")])]
    lines += ["", "## 按有效帧数", "", _md_table(history, [
        ("arm", "方法"), ("observed_outcome", "本次成败"), ("real_frames", "有效帧"), ("steps", "步数"),
        ("latest_slots_mean", "最新帧所在"), ("latest_purity_mean", "最新帧纯度"),
        ("latest_share_mean", "最新帧份额"), ("latest_exclusive_mean", "最新帧独占"),
        ("mixed_fraction_mean", "跨帧混合"), ("step_span_mean", "平均跨度")])]
    lines += ["", "## 满历史步的逐帧份额（0 = 最新帧）", "", _md_table(profile, [
        ("arm", "方法"), ("offset_from_latest", "距最新"), ("steps", "步数"),
        ("token_share_mean", "token 份额"), ("patch_retention_mean", "patch 保留率")])]
    lines += ["", "## 列定义", "",
              "- 最新帧所在：含至少一个最新帧 patch 的输出 token 数。",
              "- 最新帧纯度：只在这些 token 上，最新帧 patch 所占比例的平均；1.0 表示没有被旧帧稀释。",
              "- 最新帧份额：按 patch 计数做分数归属时最新帧分到的 token 数，等于所在数 × 纯度；"
              "各帧份额之和等于实际使用的 token 数。这是一个定义上的选择，换归属方式数值会变。",
              "- 全部 token 上的最新帧比例平均（原字段 latest_frame_mean_slot_weight）只保留在 CSV："
              "它把“分到的 token 少”和“被稀释”混在一起，不单独用于判读。",
              "- 最新帧独占：只由最新帧构成的 token 数。",
              "- 跨帧混合：含两个及以上真实帧的 token 占实际使用 token 的比例。",
              "- 跨度：一个 token 内最早与最晚来源帧的环境步差。",
              "- 动作变化：相邻两步原始动作前 6 维的 L2 差的平均；夹爪翻转：执行动作第 7 维的符号变化次数。",
              "- 分层与参考：manifest 按正式 b8 的 G3 成败分层；本次为 b1 观察，二者可能不同。", ""]
    return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def write_report(runs, problems, partial, out_dir: Path, prefix: str = "diagnostic_cases") -> dict:
    episodes, history, profile = episode_table(runs), history_table(runs), frame_profile(runs)
    out_dir = Path(out_dir)
    paths = {
        "episodes": out_dir / f"{prefix}_episodes.csv",
        "by_history": out_dir / f"{prefix}_by_history.csv",
        "frame_profile": out_dir / f"{prefix}_frame_profile.csv",
        "markdown": out_dir / f"{prefix}.md",
        "summary": out_dir / f"{prefix}_summary.json",
    }
    _write_atomic(paths["episodes"], _csv_text(episodes))
    _write_atomic(paths["by_history"], _csv_text(history))
    _write_atomic(paths["frame_profile"], _csv_text(profile))
    _write_atomic(paths["markdown"], render_markdown(runs, problems, partial, episodes, history, profile))
    summary = {
        "schema_version": 1,
        "statistical_use": "forbidden",
        "boundary": BOUNDARY,
        "inputs": [{"arm": r["arm"], "dir": r["dir"].name,
                    "commit": (r["meta"].get("code") or {}).get("commit"),
                    "adapter_sha256": r["meta"].get("adapter_sha256"),
                    "manifest_sha256": r["meta"].get("diagnostic_dump", {}).get("manifest_sha256")}
                   for r in runs],
        "partial_dirs": [p.name for p in partial],
        "acceptance_problems": problems,
        "episodes": episodes,
    }
    _write_atomic(paths["summary"], json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return paths
