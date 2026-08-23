"""
把 measure_redundancy.py 的输出画成图。

    python src/analysis/plot_redundancy.py --in_dir results/redundancy --tag pixel

产出:
    fig_r1_concentration.png  累积集中度曲线 —— 核心图，直接回答"是否只有一小部分 patch 在动"
    fig_r2_vs_delta.png       集中度随时间间隔 Δt 的变化
    fig_r3_by_suite.png       各 suite 的集中度对比
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# dataviz 规范已验证的分类配色（light 模式相邻配对全项 PASS）
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID_C = "#1a1a19", "#5c5b55", "#e5e4df"

SUITE_LABEL = {"libero_spatial": "Spatial", "libero_object": "Object",
               "libero_goal": "Goal", "libero_10": "Long"}
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

CJK_OK = False


def L(zh: str, en: str) -> str:
    return zh if CJK_OK else en


def _style():
    global CJK_OK
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC",
              "Microsoft YaHei", "SimHei", "Droid Sans Fallback"):
        if n in avail:
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            CJK_OK = True
            break
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.edgecolor": GRID_C, "axes.labelcolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "normal" if CJK_OK else "bold",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID_C, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "legend.frameon": False,
    })


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_concentration(profiles: dict, out: Path, dt: int = 1):
    """
    核心图：横轴为按变化幅度降序排列的 patch 百分位，纵轴为累积变化占比。
    对角线 y=x 是"变化均匀分布"的基准；曲线越靠左上，冗余越大。
    """
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    x = np.linspace(0, 100, 256)
    ax.plot([0, 100], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
    ax.annotate(L("变化均匀分布（无冗余）", "uniform change (no redundancy)"),
                (62, 0.60), color=MUTED, fontsize=9, rotation=27, ha="center")

    drawn, legend_rows = 0, []
    for suite in ORDER:
        key = f"{suite}__dt{dt}"
        if key not in profiles:
            continue
        cum = np.cumsum(profiles[key])
        c = PALETTE[drawn % len(PALETTE)]
        ax.plot(x, cum, color=c, lw=2, zorder=3)
        legend_rows.append((SUITE_LABEL.get(suite, suite), cum[int(256 * 0.10) - 1], c))
        drawn += 1

    # 曲线彼此贴近时逐线标注会重叠，改用右下角的色码数值块。
    # 低对比度配色要求 relief，规范允许以"表格视图"形式提供。
    if legend_rows:
        ax.text(0.585, 0.30, L("前 10% patch 的变化占比", "share of change in top 10% patches"),
                transform=ax.transAxes, fontsize=9, color=INK, fontweight="bold")
        for i, (name, val, c) in enumerate(legend_rows):
            yy = 0.245 - i * 0.052
            ax.plot([0.60], [yy + 0.012], marker="s", ms=7, color=c,
                    transform=ax.transAxes, clip_on=False)
            ax.text(0.635, yy, f"{name:<9s}{val*100:5.1f}%", transform=ax.transAxes,
                    fontsize=9.5, color=INK, family="monospace")

    ax.axvline(10, color=INK, lw=0.8, ls=":", zorder=2)
    ax.annotate(L("前 10% 的 patch", "top 10% of patches"), (10, 0.04),
                xytext=(4, 0), textcoords="offset points", fontsize=9, color=INK)

    ax.set_xlim(0, 100); ax.set_ylim(0, 1.02)
    ax.set_xlabel(L("patch 百分位（按变化幅度降序）", "patch percentile (sorted by change, desc)"))
    ax.set_ylabel(L("累积变化占比", "cumulative share of total change"))
    ax.set_title(L(f"视觉变化的集中度（Δt={dt} 帧）",
                   f"Concentration of visual change (Δt={dt} frames)"))
    _despine(ax)
    fig.savefig(out / "fig_r1_concentration.png")
    plt.close(fig)


def fig_vs_delta(df: pd.DataFrame, out: Path):
    """集中度随时间间隔的变化：间隔越大，变化越分散还是越集中？"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (key, label) in zip(axes, [
        ("top10pct_share", L("前 10% patch 的变化占比", "share of change in top 10% patches")),
        ("gini", L("Gini 系数（越高越集中）", "Gini coefficient (higher = more concentrated)")),
    ]):
        for i, suite in enumerate(ORDER):
            sub = df[df.task_suite == suite].groupby("delta_t")[key].mean()
            if sub.empty:
                continue
            c = PALETTE[i % len(PALETTE)]
            ax.plot(sub.index, sub.values, marker="o", ms=6, lw=2, color=c)
            ax.annotate(SUITE_LABEL.get(suite, suite), (sub.index[-1], sub.values[-1]),
                        xytext=(6, 0), textcoords="offset points",
                        color=c, fontsize=9, va="center", fontweight="bold")
        ax.set_xlabel(L("时间间隔 Δt（帧）", "frame gap Δt"))
        ax.set_title(label, fontsize=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(df.delta_t.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(axis="x", visible=False)
        _despine(ax)
    fig.suptitle(L("集中度随时间间隔的变化", "Concentration vs. temporal gap"),
                 fontweight=None if CJK_OK else "bold")
    fig.tight_layout()
    fig.savefig(out / "fig_r2_vs_delta.png")
    plt.close(fig)


def fig_by_suite(df: pd.DataFrame, out: Path, dt: int = 1):
    """各 suite 在 Δt=1 时的关键指标对比。"""
    sub = df[df.delta_t == dt]
    if sub.empty:
        return
    keys = [("top5pct_share", L("前5%", "top 5%")),
            ("top10pct_share", L("前10%", "top 10%")),
            ("top25pct_share", L("前25%", "top 25%"))]
    suites = [s for s in ORDER if s in set(sub.task_suite)]
    x = np.arange(len(suites)); w = 0.26

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for i, (k, lab) in enumerate(keys):
        vals = [sub[sub.task_suite == s][k].mean() for s in suites]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab,
                      color=PALETTE[i], zorder=3)
        for b in bars:
            ax.annotate(f"{b.get_height():.2f}",
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=MUTED,
                        xytext=(0, 2), textcoords="offset points")
    # 均匀分布下的期望值参考线
    for i, (frac, lab) in enumerate([(0.05, "5%"), (0.10, "10%"), (0.25, "25%")]):
        ax.axhline(frac, color=MUTED, lw=0.7, ls=":", zorder=1)

    ax.set_xticks(x, [SUITE_LABEL.get(s, s) for s in suites])
    ax.set_ylabel(L("累积变化占比", "cumulative share of change"))
    ax.set_ylim(0, 1.05)
    ax.set_title(L(f"各 suite 的变化集中度（Δt={dt}；虚线为均匀分布基准）",
                   f"Change concentration by suite (Δt={dt}; dotted = uniform baseline)"))
    ax.legend(loc="lower right")
    ax.grid(axis="x", visible=False)
    _despine(ax)
    fig.savefig(out / "fig_r3_by_suite.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/redundancy")
    ap.add_argument("--tag", default="pixel", help="pixel / feature_dino / feature_siglip")
    ap.add_argument("--dt", type=int, default=1)
    args = ap.parse_args()

    _style()
    d = Path(args.in_dir)
    df = pd.read_csv(d / f"redundancy_{args.tag}.csv")
    profiles = dict(np.load(d / f"profiles_{args.tag}.npz"))

    out = d / "figures"; out.mkdir(parents=True, exist_ok=True)
    fig_concentration(profiles, out, args.dt)
    fig_vs_delta(df, out)
    fig_by_suite(df, out, args.dt)

    print(f"\n=== 关键数字（Δt={args.dt}）===")
    sub = df[df.delta_t == args.dt].groupby("task_suite")[
        ["top5pct_share", "top10pct_share", "top25pct_share", "gini", "frac_above_0.1"]].mean()
    print(sub.round(4).to_string())
    print(f"\n参照：若变化在 256 个 patch 上均匀分布，")
    print(f"      top5%=0.05  top10%=0.10  top25%=0.25  gini=0")
    print(f"\n图表 -> {out}")


if __name__ == "__main__":
    main()
