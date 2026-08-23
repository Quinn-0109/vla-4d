"""
从冗余剖面反推 token 预算：覆盖 X% 的视觉变化，需要保留多少 patch？

    python src/analysis/token_budget.py --tag pixel

这是 4D 池化方案里最直接可用的数字 —— 它决定"历史帧保留多少 token"这个
超参该取多少，而不是拍脑袋。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SUITE_LABEL = {"libero_spatial": "Spatial", "libero_object": "Object",
               "libero_goal": "Goal", "libero_10": "Long"}
ORDER = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
COVERAGES = (0.80, 0.90, 0.95, 0.99)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/redundancy")
    ap.add_argument("--tag", default="pixel")
    ap.add_argument("--dt", type=int, default=1)
    args = ap.parse_args()

    prof = dict(np.load(Path(args.in_dir) / f"profiles_{args.tag}.npz"))
    n_patch = len(next(iter(prof.values())))

    print(f"\n{'='*78}")
    print(f"覆盖 X% 的视觉变化所需的 patch 数（共 {n_patch} 个，Δt={args.dt}，{args.tag}）")
    print(f"{'='*78}")
    header = f"{'Suite':<10}" + "".join(f"{int(c*100)}%".rjust(16) for c in COVERAGES)
    print(header)
    print("-" * 78)

    rows = {}
    for suite in ORDER:
        key = f"{suite}__dt{args.dt}"
        if key not in prof:
            continue
        cum = np.cumsum(prof[key])
        cells, vals = [], []
        for c in COVERAGES:
            k = int(np.searchsorted(cum, c) + 1)
            cells.append(f"{k:>4d} ({k/n_patch*100:4.1f}%)".rjust(16))
            vals.append(k)
        rows[suite] = vals
        print(f"{SUITE_LABEL.get(suite, suite):<10}" + "".join(cells))

    if rows:
        print("-" * 78)
        mx = {i: max(v[i] for v in rows.values()) for i in range(len(COVERAGES))}
        print(f"{'最坏情况':<10}" +
              "".join(f"{mx[i]:>4d} ({mx[i]/n_patch*100:4.1f}%)".rjust(16)
                      for i in range(len(COVERAGES))))

        print(f"\n均匀分布下需要 {int(n_patch*0.90)} 个 patch 才能覆盖 90%；"
              f"实测最坏情况仅需 {mx[1]} 个，压缩比 {n_patch/mx[1]:.1f}×")

        # 按 K 帧历史估算总 token 数
        print(f"\n{'='*78}\nK 帧历史的 token 预算估算（当前帧全保留 + 历史帧按 90% 覆盖压缩）\n{'='*78}")
        keep = mx[1]
        print(f"{'K':<5}{'朴素堆叠':>12}{'本方案':>12}{'压缩比':>10}")
        print("-" * 40)
        for K in (2, 4, 8, 16):
            naive = n_patch * K
            ours = n_patch + keep * (K - 1)
            print(f"{K:<5}{naive:>12}{ours:>12}{naive/ours:>9.1f}×")


if __name__ == "__main__":
    main()
