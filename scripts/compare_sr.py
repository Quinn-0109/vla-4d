#!/usr/bin/env python
"""
成功率的两两比较 —— 主线对照的通用判读工具。

    python scripts/compare_sr.py 官方=421/500 20k=370/500 25k=?/500 30k=?/500

对每一对做两独立比例 z 检验，并给出 Wilson 置信区间与检验功效。
后面 G0–G4 + M2 的每一对比较都用它，别再手算。

**为什么要有这个脚本**：本项目已经三次因为「看差值不看功效」得出错误结论
（"2.7× 压缩免费"、step15000 的 56% 被当成退化、"硬门槛通过"）。
把功效和置信区间和差值一起打出来，就没法只盯着差值了。
"""

from __future__ import annotations

import sys
from math import erf, erfc, sqrt

Z95 = 1.959964
Z80 = 0.8416


def wilson(s: int, n: int, z: float = Z95) -> tuple[float, float]:
    p = s / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def ztest(s1: int, n1: int, s2: int, n2: int):
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1, p2, 0.0, 1.0
    z = (p2 - p1) / se
    return p1, p2, z, erfc(abs(z) / sqrt(2))


def power(p1: float, d: float, n: int) -> float:
    """给定基线 p1、真实差异 d、每组 n，双侧 α=0.05 下的检验功效。"""
    p2 = p1 + d
    if not (0 < p2 < 1) or d == 0:
        return float("nan")
    pb = (p1 + p2) / 2
    se0 = sqrt(2 * pb * (1 - pb) / n)
    se1 = sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / n)
    return 0.5 * (1 + erf(((abs(d) - Z95 * se0) / se1) / sqrt(2)))


def need_n(p1: float, d: float) -> float:
    """80% 功效检出 d 所需的每组样本量。"""
    p2 = p1 + d
    pb = (p1 + p2) / 2
    return (Z95 * sqrt(2 * pb * (1 - pb))
            + Z80 * sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / d ** 2


def main() -> None:
    args = [a for a in sys.argv[1:] if "=" in a and "?" not in a]
    if len(args) < 2:
        print(__doc__)
        print("⚠️ 至少要两组有效数据（`?` 会被跳过，方便先填一个）。")
        raise SystemExit(1)

    groups = []
    for a in args:
        name, frac = a.split("=", 1)
        s, n = frac.split("/")
        groups.append((name, int(s), int(n)))

    print(f"{'组':<10} {'成功率':>8}   {'95% CI (Wilson)':>18}")
    for name, s, n in groups:
        lo, hi = wilson(s, n)
        print(f"{name:<10} {s/n:>7.1%}   [{lo:>6.1%}, {hi:>6.1%}]   ({s}/{n})")

    print(f"\n{'对比':<22} {'差值':>8} {'z':>7} {'p':>10}   判读")
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            n1, s1, t1 = groups[i]
            n2, s2, t2 = groups[j]
            p1, p2, z, pv = ztest(s1, t1, s2, t2)
            d = 100 * (p2 - p1)
            mark = "**显著**" if pv < 0.05 else "不显著"
            # 不显著时必须同时报功效，否则「不显著」会被误读成「相同」
            extra = ""
            if pv >= 0.05:
                pw = power(p1, 0.10, min(t1, t2))
                extra = f"（检出 10 点的功效 {pw:.0%}）"
                if pw < 0.8:
                    extra += " ⚠️ 功效不足，不能声称相同"
            print(f"{n1+' vs '+n2:<22} {d:>+7.1f}点 {z:>7.2f} {pv:>10.2e}   {mark}{extra}")

    print("\n⚠️ p > 0.05 不等于「两者相同」。上面每个不显著的对比都附了功效——"
          "功效不足时，正确的表述是「本实验无法区分」，不是「没有差异」。")
    print(f"   参考：基线 50% 时，80% 功效检出 10 个点需每组 n≈{need_n(0.5, 0.10):.0f}；"
          f"检出 5 个点需 n≈{need_n(0.5, 0.05):.0f}")


if __name__ == "__main__":
    main()
