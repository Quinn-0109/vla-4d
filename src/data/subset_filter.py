"""
训练侧的固定子集过滤 —— **2×2 四格的前提**（`docs/06` §4.1 纪律 1b、§8.5b 的方案 A′）。

判据只有一条：**一个样本可用，当且仅当我们有它的深度。**
所以键就是 `img_hash ∈ depth_cache` —— 它不依赖 episode 下标（RLDS 训练时
打乱 + 交错读，下标没有意义），而且它**就是"能不能用"的定义本身**，
不是另造一个可能与之不一致的清单。

⚠️⚠️ **G3 也必须套同一个过滤器。** G3 用不到深度，但它必须和另外三格看**同样的
数据** —— 这是纪律 1b 在 2×2 内部的全部意义。漏了它：四格数据量不同，
`G4 vs G3` 的差里混进"谁的训练数据多"，而**不会报任何错**。

⚠️ 为什么在迭代器这一层过滤，而不是在 RLDS 里：octo 的 `make_dataset_from_rlds`
确实有 `filter_functions` 钩子，挂在那里更省（被丢的样本不必解码增广）。
但它要 `ModuleSpec` 序列化、且 openvla 没有把它透出到 `RLDSDataset` 的签名里，
接进去要改上游。这一层每步只多解码约三分之一的样本，而训练是 GPU 受限的
（2.79 s/步），代价可以忽略 —— **先要正确，再要快**。
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


class SubsetFiltered(IterableDataset):
    """
    包住 `RLDSDataset`，只放行深度缓存里有的样本。

    `keep_rate_range` 是**防静默失效的闸**：过滤器最危险的两种坏法都不会报错 ——
    键对不上则一条都不放行（0%），判断写反则全部放行（100%，等于没过滤）。
    两者都会让训练照常跑完，一个训不出东西、一个悄悄用了全量数据。
    所以头 `warmup` 个样本之后强制检查一次留存率。
    """

    def __init__(self, base, cache, keep_rate_range=(0.45, 0.90), warmup: int = 400):
        self.base = base
        # DepthCache 暴露的是 idx（哈希 → 行号）；也接受裸的哈希数组（自测用）
        if hasattr(cache, "idx"):
            self._h = set(int(k) for k in cache.idx)
        else:
            # int64 载 uint64：与 KFrameBatchTransform 里的转换一致（见 depth_cache）
            self._h = set(np.asarray(cache).astype(np.int64).tolist())
        if not self._h:
            raise ValueError("深度缓存是空的 —— 先跑 build_subset.py --commit")
        self.lo, self.hi = keep_rate_range
        self.warmup = warmup
        self.n_seen = self.n_kept = 0
        self._checked = False

    def __len__(self):
        return len(self.base)

    def _check(self) -> None:
        r = self.n_kept / max(self.n_seen, 1)
        if not (self.lo <= r <= self.hi):
            raise RuntimeError(
                f"子集过滤的留存率 {r:.1%}（{self.n_kept}/{self.n_seen}）落在 "
                f"[{self.lo:.0%}, {self.hi:.0%}] 之外。\n"
                f"  · 接近 0% → 帧指纹与深度缓存对不上（先跑 "
                f"scripts/check_hash_key.py）\n"
                f"  · 接近 100% → 判断写反了，等于没过滤，四格会各自用全量数据\n"
                f"  子集覆盖率应当是 255/379 ≈ 67%，抽样留存率该在它附近。")
        print(f"✓ 子集过滤留存率 {r:.1%}（{self.n_kept}/{self.n_seen}），"
              f"与 255/379 ≈ 67.3% 相符", flush=True)
        self._checked = True

    def __iter__(self) -> Iterator[dict]:
        for s in self.base:
            hs = s.get("img_hash")
            if hs is None:
                raise RuntimeError(
                    "样本里没有 img_hash —— 子集过滤没有键可用。"
                    "分窗补丁必须先给每帧打指纹（见 data/kframe.py 的 ⓪）。")
            self.n_seen += 1
            # 窗口内 K 帧同属一条 episode，所以是全有或全无；全查一遍不额外花钱
            if all(int(x) in self._h for x in torch.as_tensor(hs).reshape(-1).tolist()):
                self.n_kept += 1
                yield s
            if not self._checked and self.n_seen >= self.warmup:
                self._check()


def _selftest() -> None:
    class _Fake:
        def __init__(self, hs):
            self.hs = hs
        def __iter__(self):
            for h in self.hs:
                yield {"img_hash": torch.tensor(h, dtype=torch.int64)}
        def __len__(self):
            return len(self.hs)

    keep = np.arange(100, dtype=np.int64)
    # 约 67% 的样本落在缓存里
    rng = np.random.default_rng(0)
    hs = [[int(x)] * 8 for x in rng.choice(150, size=600)]
    d = SubsetFiltered(_Fake(hs), keep, warmup=300)
    out = list(d)
    print(f"✅ 1/3 放行 {len(out)}/600，留存率 {len(out)/600:.1%}（缓存覆盖 100/150）")
    assert all(int(s["img_hash"][0]) < 100 for s in out), "放行了缓存外的样本"

    # 键全不匹配 → 必须抛错，不能静默训一个空数据集
    bad = SubsetFiltered(_Fake([[9999] * 8] * 600), keep, warmup=300)
    try:
        list(bad)
        raise SystemExit("✗ 留存率 0% 竟然没抛错")
    except RuntimeError as e:
        assert "对不上" in str(e)
    print("✅ 2/3 键全不匹配（0%）→ 抛错，不会静默训空数据集")

    # 全部匹配 → 也必须抛错：那等于没过滤
    allin = SubsetFiltered(_Fake([[1] * 8] * 600), keep, warmup=300)
    try:
        list(allin)
        raise SystemExit("✗ 留存率 100% 竟然没抛错")
    except RuntimeError as e:
        assert "写反" in str(e)
    print("✅ 3/3 全部放行（100%）→ 抛错，那等于没过滤")
    print("\n全部通过。")


if __name__ == "__main__":
    _selftest()
