"""
成功 vs 失败 episode 的统计对比 —— 把箱线图里的视觉差异变成可引用的数字。

    python src/analysis/compare_groups.py --metrics results/tables/metrics.csv

对每个指标输出: 两组中位数/IQR、Mann-Whitney U 检验 p 值、
以及 rank-biserial 效应量(取值 -1~1，绝对值越大差异越大)。

用非参数检验而非 t 检验: jerk 等指标分布严重右偏且有长尾，
均值和正态假设都不成立，秩检验对此稳健。
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

METRICS = [
    ("normalized_jerk",     "归一化 Jerk",        "越低越平滑"),
    ("sparc",               "SPARC",              "越接近0越平滑"),
    ("velocity_reversals",  "速度方向反转次数",    "越低越好"),
    ("idle_ratio",          "空转步数占比",        "越低越好"),
    ("path_efficiency",     "路径效率",            "越高越直接"),
    ("path_length",         "路径长度 (m)",        ""),
    ("action_jitter",       "动作抖动",            "越低越好"),
    ("gripper_flips",       "夹爪翻转次数",        "越低越好"),
    ("mean_speed",          "平均速度 (m/s)",      ""),
    ("completion_time_s",   "完成时间 (s)",        "越低越好"),
]


def _rank(x: np.ndarray) -> np.ndarray:
    """平均秩(处理并列)，等价于 scipy.stats.rankdata，避免引入 scipy 依赖。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1)
    # 并列取平均秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def mann_whitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """返回 (U, p 值双侧, rank-biserial 效应量)。大样本正态近似 + 并列校正。"""
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan"), float("nan")

    allv = np.concatenate([a, b])
    r = _rank(allv)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2

    mu = n1 * n2 / 2
    # 并列校正的方差
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum()
    n = n1 + n2
    var = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return u1, float("nan"), float("nan")

    z = (u1 - mu) / math.sqrt(var)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    rb = 2 * u1 / (n1 * n2) - 1        # rank-biserial: +1 表示 a 普遍大于 b
    return u1, p, rb


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.2e}" if p < 0.01 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/tables/metrics.csv")
    ap.add_argument("--out", default="results/tables/success_vs_failure.csv")
    ap.add_argument("--by_suite", action="store_true", help="按 suite 分别统计")
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    groups = df.groupby("task_suite") if args.by_suite else [("全部", df)]

    rows = []
    for suite, sub in groups:
        succ = sub[sub.success]
        fail = sub[~sub.success]
        print(f"\n{'='*88}")
        print(f"{suite}   成功 n={len(succ)}   失败 n={len(fail)}   "
              f"成功率 {len(succ)/max(len(sub),1)*100:.1f}%")
        print(f"{'='*88}")
        print(f"{'指标':<22}{'成功 中位数':>14}{'失败 中位数':>14}"
              f"{'比值':>9}{'p 值':>11}{'效应量':>9}")
        print("-" * 88)

        for key, label, note in METRICS:
            if key not in sub.columns:
                continue
            a = succ[key].dropna().values
            b = fail[key].dropna().values
            if len(a) < 3 or len(b) < 3:
                continue

            ma, mb = float(np.median(a)), float(np.median(b))
            _, p, rb = mann_whitney(a, b)
            ratio = mb / ma if abs(ma) > 1e-12 else float("nan")
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

            # 成功中位数为 0 时比值无意义(如速度反转次数)，显示为 —
            ratio_s = "—" if (math.isnan(ratio) or math.isinf(ratio)) else f"{ratio:.2f}"
            print(f"{label:<22}{ma:>14.4g}{mb:>14.4g}{ratio_s:>9}"
                  f"{fmt_p(p):>11}{rb:>8.2f} {star}")
            rows.append(dict(suite=suite, metric=key, label=label, note=note,
                             n_success=len(a), n_fail=len(b),
                             median_success=ma, median_fail=mb,
                             q1_success=float(np.percentile(a, 25)),
                             q3_success=float(np.percentile(a, 75)),
                             q1_fail=float(np.percentile(b, 25)),
                             q3_fail=float(np.percentile(b, 75)),
                             ratio_fail_over_success=ratio,
                             p_value=p, rank_biserial=rb))

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\n比值 = 失败中位数 / 成功中位数   效应量 = rank-biserial (|值|越大差异越明显)")
    print(f"显著性: * p<0.05   ** p<0.01   *** p<0.001")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
