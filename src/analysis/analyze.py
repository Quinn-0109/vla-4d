"""
汇总 run_libero_eval_traj.py 落盘的轨迹，计算指标并出图。

    python src/analysis/analyze.py --traj_dir results/trajectories

产出 (results/figures/ 与 results/tables/):
    metrics.csv                 每 episode 全部指标的原始表
    summary_by_suite.csv        按 suite 汇总
    fig1_success_rate.png       成功率 vs OpenVLA 论文值
    fig2_smoothness.png         成功/失败轨迹的平滑度分布
    fig3_trajectory_3d.png      末端执行器 3D 轨迹样例
    fig4_velocity_profile.png   速度剖面 —— 展示启停抖动的核心图
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.analysis.metrics import DEFAULT_DT, compute_all, velocity  # noqa: E402

# 已验证的分类配色（dataviz 规范，light 模式全项 PASS）
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"   # 蓝 / 橙 / 青
INK, MUTED, GRID = "#1a1a19", "#5c5b55", "#e5e4df"

# OpenVLA 论文 Appendix E Table 12 的官方数字，用作对标基线
PAPER_SR = {
    "libero_spatial": 84.7,
    "libero_object": 88.4,
    "libero_goal": 79.2,
    "libero_10": 53.7,
}
SUITE_LABEL = {
    "libero_spatial": "Spatial",
    "libero_object": "Object",
    "libero_goal": "Goal",
    "libero_10": "Long",
}


#: 图上是否可用中文。由 _style() 探测系统字体后设置；缺字体时全部标签回退英文。
CJK_OK = False


def L(zh: str, en: str) -> str:
    """标签取中/英 —— 服务器常缺 CJK 字体，缺了就自动用英文，避免出现方块。"""
    return zh if CJK_OK else en


def _setup_font() -> bool:
    """探测系统里可用的 CJK 字体并挂到 matplotlib。找不到返回 False。"""
    from matplotlib import font_manager
    candidates = [
        "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
        "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei",
        "PingFang SC", "SimHei", "Droid Sans Fallback",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False   # 中文字体下负号会变方块
            return True
    return False


def _style():
    global CJK_OK
    CJK_OK = _setup_font()
    if not CJK_OK:
        print("提示: 未找到中文字体，图表标签使用英文。"
              "\n      安装中文字体可显示中文: apt install -y fonts-wqy-zenhei")
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        # 多数 CJK 字体没有 bold 字重，强行指定会刷屏 findfont 警告
        "axes.titleweight": "normal" if CJK_OK else "bold",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
    })


def _despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def load_trajectories(traj_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    """遍历所有 run 目录，算指标并保留原始轨迹供画图。"""
    rows, raw = [], []
    files = sorted(traj_dir.rglob("task*_ep*.json"))
    if not files:
        raise SystemExit(f"在 {traj_dir} 下没找到轨迹文件。先跑 scripts/run_eval.sh")
    print(f"找到 {len(files)} 条轨迹")
    for fp in files:
        with open(fp) as f:
            traj = json.load(f)
        rows.append(compute_all(traj, DEFAULT_DT))
        raw.append(traj)
    return pd.DataFrame(rows), raw


def fig_success_rate(df: pd.DataFrame, out: Path):
    """图1: 各 suite 成功率与 OpenVLA 论文值对比。"""
    suites = [s for s in PAPER_SR if s in set(df["task_suite"])]
    if not suites:
        return
    ours = [df[df.task_suite == s]["success"].mean() * 100 for s in suites]
    paper = [PAPER_SR[s] for s in suites]

    x = np.arange(len(suites))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x - w / 2, ours, w, label=L("本次复现", "This run"), color=C1, zorder=3)
    b2 = ax.bar(x + w / 2, paper, w, label=L("OpenVLA 论文", "OpenVLA paper"), color=C2, zorder=3)

    # 直接标注数值 —— 规范要求为对比图提供 relief
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(f"{b.get_height():.1f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=9, color=MUTED,
                        xytext=(0, 2), textcoords="offset points")

    ax.set_xticks(x, [SUITE_LABEL[s] for s in suites])
    ax.set_ylabel(L("任务成功率 (%)", "Success rate (%)"))
    ax.set_ylim(0, 105)
    ax.set_title(L("OpenVLA 在 LIBERO 上的复现结果", "OpenVLA reproduction on LIBERO"))
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.savefig(out / "fig1_success_rate.png")
    plt.close(fig)


def fig_smoothness(df: pd.DataFrame, out: Path):
    """图2: 成功 vs 失败 episode 的平滑度指标分布。"""
    metrics = [
        ("normalized_jerk", L("归一化 Jerk\n(越低越平滑)", "Normalized jerk\n(lower = smoother)"), True),
        ("velocity_reversals", L("速度方向反转次数", "Velocity reversals"), False),
        ("idle_ratio", L("空转步数占比", "Idle-step ratio"), False),
        ("path_efficiency", L("路径效率\n(越高越直接)", "Path efficiency\n(higher = straighter)"), False),
    ]
    avail = [(k, lbl, lg) for k, lbl, lg in metrics if k in df.columns and df[k].notna().any()]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(3.3 * len(avail), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (key, label, logscale) in zip(axes, avail):
        groups, colors, names = [], [], []
        for flag, color, name in ((True, C1, L("成功", "Success")), (False, C2, L("失败", "Failure"))):
            vals = df[(df.success == flag)][key].dropna().values
            if len(vals):
                groups.append(vals); colors.append(color); names.append(f"{name}\n(n={len(vals)})")
        if not groups:
            continue
        bp = ax.boxplot(groups, patch_artist=True, widths=0.5, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.6),
                        whiskerprops=dict(color=MUTED, linewidth=1),
                        capprops=dict(color=MUTED, linewidth=1))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c); patch.set_alpha(0.75); patch.set_edgecolor(c)
        ax.set_xticks(range(1, len(names) + 1), names, fontsize=9)
        ax.set_title(label, fontsize=10)
        if logscale:
            ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
        _despine(ax)
    fig.suptitle(L("轨迹平滑度: 成功与失败 episode 对比", "Trajectory smoothness: success vs failure"), fontweight=None if CJK_OK else "bold")
    fig.tight_layout()
    fig.savefig(out / "fig2_smoothness.png")
    plt.close(fig)


def fig_trajectory_3d(raw: list[dict], out: Path, n: int = 3):
    """图3: 末端执行器 3D 轨迹样例，成功与失败并排。"""
    succ = [t for t in raw if t.get("success") and len(t.get("steps", [])) > 10][:n]
    fail = [t for t in raw if not t.get("success") and len(t.get("steps", [])) > 10][:n]
    cases = ([(t, C1, L("成功", "Success")) for t in succ]
             + [(t, C2, L("失败", "Failure")) for t in fail])
    if not cases:
        return

    ncol = min(len(cases), 6)
    fig = plt.figure(figsize=(3.1 * ncol, 3.4))
    for i, (traj, color, tag) in enumerate(cases[:ncol]):
        pos = np.array([s["eef_pos"] for s in traj["steps"]])
        ax = fig.add_subplot(1, ncol, i + 1, projection="3d")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, linewidth=1.4)
        ax.scatter(*pos[0], color=INK, s=22, marker="o", label=L("起点", "start"))
        ax.scatter(*pos[-1], color=color, s=40, marker="*", label=L("终点", "end"))
        ax.set_title(f"{tag} · {traj['task_description'][:22]}…", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x", fontsize=7); ax.set_ylabel("y", fontsize=7); ax.set_zlabel("z", fontsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(L("末端执行器 3D 轨迹", "End-effector 3D trajectories"), fontweight=None if CJK_OK else "bold")
    fig.tight_layout()
    fig.savefig(out / "fig3_trajectory_3d.png")
    plt.close(fig)


def fig_velocity_profile(raw: list[dict], out: Path, n: int = 4):
    """
    图4: 速度剖面 —— 论证"时序不连贯"的核心证据图。
    平滑策略应呈现钟形速度曲线；抖动策略会看到高频锯齿和频繁触零。
    """
    cases = [t for t in raw if len(t.get("steps", [])) > 20][:n]
    if not cases:
        return

    fig, axes = plt.subplots(len(cases), 1, figsize=(7.6, 1.9 * len(cases)), sharex=False)
    axes = np.atleast_1d(axes)
    for ax, traj in zip(axes, cases):
        pos = np.array([s["eef_pos"] for s in traj["steps"]])
        spd = np.linalg.norm(velocity(pos, DEFAULT_DT), axis=1)
        tt = np.arange(len(spd)) * DEFAULT_DT
        color = C1 if traj.get("success") else C2
        ax.plot(tt, spd, color=color, linewidth=1.2)
        ax.fill_between(tt, spd, color=color, alpha=0.13)

        # 标出速度触零(空转)的位置
        idle = spd < 1e-3
        if idle.any():
            ax.scatter(tt[idle], spd[idle], s=6, color=MUTED, zorder=4)

        tag = L("成功", "Success") if traj.get("success") else L("失败", "Failure")
        ax.set_title(f"[{tag}] {traj['task_description'][:52]}", fontsize=9, loc="left")
        ax.set_ylabel(L("速度 (m/s)", "Speed (m/s)"), fontsize=9)
        ax.grid(axis="x", visible=False)
        _despine(ax)
    axes[-1].set_xlabel(L("时间 (s)", "Time (s)"))
    fig.suptitle(L("末端执行器速度剖面 —— 灰点为速度触零(空转)",
                   "End-effector speed profile (grey = zero-speed / idle)"), fontweight=None if CJK_OK else "bold")
    fig.tight_layout()
    fig.savefig(out / "fig4_velocity_profile.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="results/trajectories")
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    _style()
    traj_dir = Path(args.traj_dir)
    fig_dir = Path(args.out_dir) / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = Path(args.out_dir) / "tables"; tab_dir.mkdir(parents=True, exist_ok=True)

    df, raw = load_trajectories(traj_dir)
    df.to_csv(tab_dir / "metrics.csv", index=False)

    num_cols = [c for c in df.columns if df[c].dtype.kind in "fi" and c != "task_id"]
    summary = df.groupby("task_suite").agg(
        episodes=("success", "size"),
        success_rate=("success", lambda s: s.mean() * 100),
        **{c: (c, "mean") for c in num_cols},
    ).round(4)
    summary.to_csv(tab_dir / "summary_by_suite.csv")

    print("\n=== summary by suite ===")
    cols = [c for c in ["episodes", "success_rate", "normalized_jerk", "velocity_reversals",
                        "idle_ratio", "path_efficiency", "completion_time_s"] if c in summary.columns]
    print(summary[cols].to_string())

    for s in summary.index:
        if s in PAPER_SR:
            got, want = summary.loc[s, "success_rate"], PAPER_SR[s]
            print(f"  {SUITE_LABEL.get(s, s):8s} 复现 {got:5.1f}%  论文 {want:5.1f}%  差 {got-want:+5.1f}")

    fig_success_rate(df, fig_dir)
    fig_smoothness(df, fig_dir)
    fig_trajectory_3d(raw, fig_dir)
    fig_velocity_profile(raw, fig_dir)
    print(f"\n图表 -> {fig_dir}\n表格 -> {tab_dir}")


if __name__ == "__main__":
    main()
