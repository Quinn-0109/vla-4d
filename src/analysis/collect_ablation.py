"""
汇总 token 预算消融的结果 —— 把一堆 summary.json 变成一条"预算 vs 成功率"曲线。

    python src/analysis/collect_ablation.py --traj_dir results/trajectories \
           --suite libero_spatial --trials 5

基线不重跑: 评测完全确定(docs/05 3.2)，满量 500 局里 episode_idx < trials 的那些
局与消融跑的是同一批初始状态，直接从已有轨迹里筛出来即可。

输出:
    results/tables/token_ablation.csv
    fig_a1_budget_curve.png    成功率 vs token 预算(主图)
    fig_a2_methods.png         同一预算下各算子对比(存在阶段二数据时)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cf6"]
INK, MUTED, GRID_C = "#1a1a19", "#5c5b55", "#e5e4df"
N_PATCH = 256

METHOD_LABEL = {"random": "随机", "uniform": "均匀网格", "norm": "L2 范数 top-k",
                "avgpool": "网格平均池化", "tome": "ToMe 合并",
                "expand": "ToMe 合并后广播回 256 位", "shuffle": "打乱 256 个 token 的顺序",
                "tome+fixpos": "ToMe 合并 + 位置修正"}
METHOD_LABEL_EN = {"random": "Random", "uniform": "Uniform grid", "norm": "L2-norm top-k",
                   "avgpool": "Grid avg-pool", "tome": "ToMe merge",
                   "expand": "ToMe, broadcast back to 256", "shuffle": "Shuffled 256 tokens",
                   "tome+fixpos": "ToMe + corrected positions"}

# 对照/诊断算子: 输出仍是 256 个 token，不是候选方案，不进主曲线和算子对比图
DIAG_METHODS = {"expand", "shuffle"}


def _setup_font() -> bool:
    """有中文字体就用，没有则整张图退回英文——总比画一堆豆腐块强。"""
    from matplotlib import font_manager
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for n in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "Source Han Sans SC",
              "Microsoft YaHei", "SimHei", "Droid Sans Fallback"):
        if n in avail:
            plt.rcParams["font.sans-serif"] = [n, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return True
    return False


CJK_OK = _setup_font()
L = (lambda zh, en: zh if CJK_OK else en)


# ------------------------------------------------------------------ 统计
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 区间。成功率接近 0/1 时正态近似会给出越界的下限，Wilson 不会。"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """两比例 z 检验(合并标准误)。返回 (z, 双侧 p)。"""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    if se == 0:
        return (0.0, 1.0)
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2))
    return (z, p)


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    return "<1e-4" if p < 1e-4 else (f"{p:.2e}" if p < 0.01 else f"{p:.3f}")


# ------------------------------------------------------------------ 读取
def load_runs(traj_dir: Path, suite: str) -> tuple[list[dict], list[dict]]:
    """返回 (消融 run 列表, 基线 run 列表)。按 summary.json 里的字段分类。"""
    abl, base = [], []
    for s in sorted(traj_dir.glob("*/summary.json")):
        try:
            d = json.loads(s.read_text())
        except json.JSONDecodeError:
            continue
        if d.get("task_suite") != suite:
            continue
        d["_dir"] = s.parent
        (abl if d.get("token_keep", 0) > 0 else base).append(d)
    return abl, base


def baseline_subset(runs: list[dict], trials: int) -> tuple[int, int, str]:
    """
    从基线 run 里筛出 episode_idx < trials 的那些局。
    取 episode 数最多的那次基线(满量 500 局)作为来源。
    """
    if not runs:
        return (0, 0, "")
    src = max(runs, key=lambda d: d.get("total_episodes", 0))
    k = n = 0
    for f in sorted(src["_dir"].glob("task*_ep*.json")):
        ep = int(f.stem.split("_ep")[1])
        if ep >= trials:
            continue
        n += 1
        k += bool(json.loads(f.read_text())["success"])
    return (k, n, src["run_id"])


# ------------------------------------------------------------------ 作图
def _ax(ax, xlab, ylab, title):
    ax.set_title(title, color=INK, fontsize=13, pad=12, loc="left")
    ax.set_xlabel(xlab, color=MUTED, fontsize=10)
    ax.set_ylabel(ylab, color=MUTED, fontsize=10)
    ax.grid(True, color=GRID_C, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID_C)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_budget(df: pd.DataFrame, base_p: float, base_ci, out: Path, suite: str):
    series = [(m, df[df.method == m].sort_values("keep"))
              for m in ("tome", "tome+fixpos")]
    series = [(m, d) for m, d in series if not d.empty]
    if not series:
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    if not math.isnan(base_p):
        ax.axhline(base_p * 100, color=MUTED, ls="--", lw=1.2, zorder=2)
        ax.fill_between([0, N_PATCH], base_ci[0] * 100, base_ci[1] * 100,
                        color=MUTED, alpha=0.12, zorder=1)
        ax.text(N_PATCH, base_p * 100 + 1.5,
                f"{L('未压缩基线', 'uncompressed baseline')} {base_p*100:.0f}%",
                ha="right", color=MUTED, fontsize=9)
    lab = METHOD_LABEL if CJK_OK else METHOD_LABEL_EN
    for i, (m, d) in enumerate(series):
        lo = d.rate - np.array([wilson(int(k), int(n))[0] for k, n in zip(d.succ, d.n)])
        hi = np.array([wilson(int(k), int(n))[1] for k, n in zip(d.succ, d.n)]) - d.rate
        ax.errorbar(d.keep, d.rate * 100, yerr=[lo * 100, hi * 100], fmt="o-",
                    color=PALETTE[i], lw=2, ms=6, capsize=3, zorder=3 + i)
        r = d.iloc[-1]
        ax.annotate(lab.get(m, m), (r.keep, r.rate * 100), xytext=(6, 4),
                    textcoords="offset points", color=PALETTE[i], fontsize=9,
                    fontweight="bold")
    _ax(ax, L("保留的视觉 token 数（共 256）", "Visual tokens kept (of 256)"),
        L("成功率 (%)", "Success rate (%)"),
        f"{L('冻结 OpenVLA 的 token 预算曲线', 'Frozen OpenVLA token-budget curve')}"
        f" · {suite}")
    ax.set_xlim(0, N_PATCH + 8)
    ax.set_ylim(-2, 100)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / N_PATCH * 100,
                                               lambda x: x * N_PATCH / 100))
    sec.set_xlabel(L("保留比例 (%)", "Fraction kept (%)"), color=MUTED, fontsize=9)
    sec.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


def plot_methods(df: pd.DataFrame, base_p: float, out: Path, suite: str):
    df = df[~df.method.isin(DIAG_METHODS)]
    keeps = sorted(df.groupby("keep").method.nunique().pipe(lambda s: s[s > 1]).index)
    if not keeps:
        return
    keep = keeps[-1] if len(keeps) == 1 else keeps[0]
    d = df[df.keep == keep].sort_values("rate")
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(d) + 2.4), dpi=150)
    y = np.arange(len(d))
    ax.barh(y, d.rate * 100, color=PALETTE[0], height=0.6, zorder=3)
    if not math.isnan(base_p):
        ax.axvline(base_p * 100, color=MUTED, ls="--", lw=1.2, zorder=4)
        ax.text(base_p * 100, len(d) - 0.3,
                f" {L('基线', 'baseline')} {base_p*100:.0f}%",
                color=MUTED, fontsize=9, va="top")
    ax.set_yticks(y)
    lab = METHOD_LABEL if CJK_OK else METHOD_LABEL_EN
    ax.set_yticklabels([lab.get(m, m) for m in d.method])
    for yi, r in zip(y, d.rate):
        ax.text(r * 100 + 1, yi, f"{r*100:.0f}%", va="center", color=INK, fontsize=9)
    _ax(ax, L("成功率 (%)", "Success rate (%)"), "",
        f"{L('同一预算下的算子对比', 'Operators at equal budget')} · keep={keep} · {suite}")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {out}")


# ------------------------------------------------------------------ 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="results/trajectories")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--trials", type=int, default=5,
                    help="每 task 的试验数；基线按此值从满量轨迹里取子集")
    ap.add_argument("--out_dir", default="results/tables")
    ap.add_argument("--fig_dir", default="results/figures")
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    abl, base = load_runs(traj_dir, args.suite)
    if not abl:
        raise SystemExit(f"{traj_dir} 下没有找到 {args.suite} 的消融 run "
                         f"(带 token_keep>0 的 summary.json)")

    bk, bn, bsrc = baseline_subset(base, args.trials)
    base_p = bk / bn if bn else float("nan")
    base_ci = wilson(bk, bn) if bn else (float("nan"),) * 2
    if bn:
        print(f"基线子集: {bk}/{bn} = {base_p*100:.1f}%  "
              f"[{base_ci[0]*100:.1f}, {base_ci[1]*100:.1f}]   来源 {bsrc}")
    else:
        print("⚠️ 没找到未压缩的基线 run，将只报告消融各点的绝对成功率")

    # 同一 (keep, method) 若跑了多次，取最新的一次
    latest: dict[tuple, dict] = {}
    for d in abl:
        latest[(d["token_keep"], d["token_method"])] = d

    rows = []
    for (keep, method), d in sorted(latest.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        n, k = d["total_episodes"], d["total_successes"]
        ci = wilson(k, n)
        z, p = two_prop_z(k, n, bk, bn) if bn else (float("nan"),) * 2
        rows.append({
            "keep": keep, "keep_pct": keep / N_PATCH * 100,
            "compression": N_PATCH / keep, "method": method,
            "succ": k, "n": n, "rate": k / n,
            "ci_lo": ci[0], "ci_hi": ci[1],
            "delta_vs_base": (k / n - base_p) if bn else float("nan"),
            "z": z, "p": p,
        })
    df = pd.DataFrame(rows)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    csv = out_dir / f"token_ablation_{args.suite}.csv"
    df.to_csv(csv, index=False)

    print(f"\n{args.suite}  (每配置 n={df.n.iloc[0] if len(df) else 0})")
    print(f"{'keep':>5} {'占比':>7} {'压缩':>6} {'算子':>9} {'成功率':>8} "
          f"{'95%CI':>15} {'Δ基线':>8} {'p':>8}")
    print("-" * 78)
    for r in df.itertuples():
        print(f"{r.keep:>5} {r.keep_pct:>6.1f}% {r.compression:>5.1f}x "
              f"{r.method:>9} {r.rate*100:>7.1f}% "
              f"[{r.ci_lo*100:>5.1f},{r.ci_hi*100:>5.1f}] "
              f"{r.delta_vs_base*100:>+7.1f} {fmt_p(r.p):>8}")

    # 拐点: 第一个显著低于基线的预算(从大到小扫)
    if bn:
        print()
        for m in ("tome", "tome+fixpos"):
            sub = df[df.method == m].sort_values("keep", ascending=False)
            if sub.empty:
                continue
            knee = next((r for r in sub.itertuples()
                         if r.p < 0.05 and r.delta_vs_base < 0), None)
            name = METHOD_LABEL.get(m, m) if CJK_OK else METHOD_LABEL_EN.get(m, m)
            if knee is None:
                print(f"[{name}] 扫过的所有预算都没有显著掉点"
                      f"(最低到 keep={sub.keep.min()})")
            else:
                ok = sub[sub.keep > knee.keep]
                safe = ok.keep.min() if len(ok) else None
                print(f"[{name}] 拐点 keep={knee.keep} ({knee.keep_pct:.0f}%) "
                      f"({knee.delta_vs_base*100:+.1f}, p={fmt_p(knee.p)})；"
                      f"不掉点的最低预算 keep={safe}"
                      + (f"，压缩 {N_PATCH/safe:.1f}x" if safe else ""))
        print("注: 单侧解读时 n 较小，拐点位置只是区间估计；确认请在拐点两侧加量。")

    # ---- 诊断: 掉点来自信息损失还是位置错位 ----
    diag = df[df.method.isin(DIAG_METHODS)]
    if not diag.empty:
        print("\n【诊断】压缩同时改了信息量和序列位置，这一组把位置固定住:")
        sh = diag[diag.method == "shuffle"]
        if not sh.empty:
            r = sh.iloc[0]
            print(f"  打乱 256 个 token 的顺序(信息量不变): {r['rate']*100:.1f}% "
                  f"({r['delta_vs_base']*100:+.1f}, p={fmt_p(r['p'])})"
                  f"  <- 位置敏感度的上界参照")
        ex = diag[diag.method == "expand"].sort_values("keep", ascending=False)
        tm = df[df.method == "tome"].set_index("keep")
        if not ex.empty:
            print(f"  {'不同值':>6} {'expand(长度256)':>16} {'直接压(长度k)':>15} {'差值':>8}")
            for r in ex.itertuples():
                t = f"{tm.loc[r.keep, 'rate']*100:.1f}%" if r.keep in tm.index else "  n/a"
                gap = (f"{(r.rate - tm.loc[r.keep, 'rate'])*100:+.1f}"
                       if r.keep in tm.index else "   n/a")
                print(f"  {r.keep:>6} {r.rate*100:>15.1f}% {t:>15} {gap:>8}")
            print("  读法: expand 不掉点而直接压掉点 -> 掉的是位置，不是信息；"
                  "两者都掉 -> 模型确实需要这些细节")

    print(f"\n表格 -> {csv}")
    plot_budget(df, base_p, base_ci, fig_dir / f"fig_a1_budget_curve_{args.suite}.png", args.suite)
    plot_methods(df, base_p, fig_dir / f"fig_a2_methods_{args.suite}.png", args.suite)


if __name__ == "__main__":
    main()
