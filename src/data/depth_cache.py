"""
训练集深度缓存的**键** —— G4/M2 靠它把离线渲染的深度接回训练流。

问题：RLDS 训练时打乱 + 交错读，位置下标没有意义；`episode_metadata` 里只有
`file_path`，而一个 hdf5 装 50 条 demo，它标识的是 task 不是 episode；
`KFrameBatchTransform` 拿到的图像**已经过增广**，按内容做指纹也不成立。

解法：**在轨迹变换阶段按 JPEG 字节做指纹**。octo 的链路用 `SkipDecoding` 载入，
解码与增广都在其后的 frame transform 里 —— 所以 `apply_trajectory_transforms`
看到的 `observation/image_primary` 仍是未解码的字节：逐帧唯一，且在增广之前。

    离线（scripts/build_subset.py）：hash(帧字节) → patch 级深度 (16,16)
    训练（data/kframe.py）：        分窗时把 hash 一并 gather，随 batch 带下去

⚠️ **两边必须用同一个哈希函数**，所以它只在这里定义一次。
⚠️ 查不到就**硬抛**（`DepthCache.lookup`）。查不到时退回零深度不会报错，
   G4 会照跑，只是它看的是一个和图像无关的三维世界 —— 这个项目已经为
   同一类静默失效付过 19.6 小时。
"""

import numpy as np

BUCKETS = 1 << 62          # 2^62，5 万帧的碰撞概率可忽略
GRID = 16


def hash_bytes(b):
    """
    JPEG 字节 → uint64 指纹。**离线与训练两边都走这一个函数。**

    用 `tf.strings.to_hash_bucket_fast`：它是 farmhash 的确定性实现，
    跨进程、跨机器、跨版本稳定（`hash()` 不是 —— Python 的字符串哈希带随机种子）。
    """
    import tensorflow as tf

    return tf.strings.to_hash_bucket_fast(b, BUCKETS)


def hash_np(b: bytes) -> np.uint64:
    return np.uint64(int(hash_bytes([b])[0].numpy()))


class DepthCache:
    """`hash → (16,16) float16` 的只读表。"""

    def __init__(self, hashes: np.ndarray, depth: np.ndarray):
        assert len(hashes) == len(depth), (len(hashes), len(depth))
        assert depth.shape[1:] == (GRID, GRID), depth.shape
        self.idx = {int(h): i for i, h in enumerate(hashes)}
        if len(self.idx) != len(hashes):
            raise ValueError(
                f"{len(hashes) - len(self.idx)} 个哈希碰撞 —— 换更大的 BUCKETS。")
        self.depth = depth

    @classmethod
    def load(cls, path) -> "DepthCache":
        z = np.load(str(path))
        return cls(z["hash"], z["depth"])

    def lookup(self, hashes) -> np.ndarray:
        """(K,) uint64 → (K,16,16) float32。**任何一个查不到就抛。**"""
        out = np.empty((len(hashes), GRID, GRID), np.float32)
        for i, h in enumerate(hashes):
            j = self.idx.get(int(h))
            if j is None:
                raise KeyError(
                    f"深度缓存里没有第 {i} 帧（hash {int(h)}）。"
                    "要么这条 episode 不在固定子集里（训练必须只用子集，"
                    "docs/06 §4.1 1b），要么缓存与数据集版本对不上。"
                    "退回零深度不会报错，但 G4 看的就是另一个三维世界 —— 所以硬抛。")
            out[i] = self.depth[j]
        return out

    def __len__(self) -> int:
        return len(self.idx)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    h = np.arange(5, dtype=np.uint64) * 7919
    d = rng.random((5, GRID, GRID)).astype(np.float16)
    c = DepthCache(h, d)
    assert np.allclose(c.lookup(h[[3, 1]])[0], d[3].astype(np.float32))
    print("✅ 1/3 查得到的按哈希取，顺序跟着输入")

    try:
        c.lookup([np.uint64(12345)])
    except KeyError as e:
        assert "深度缓存里没有" in str(e)
        print("✅ 2/3 查不到硬抛，不退回零深度")
    else:
        raise AssertionError("应当抛 KeyError")

    try:
        DepthCache(np.array([1, 1], np.uint64), d[:2])
    except ValueError as e:
        assert "碰撞" in str(e)
        print("✅ 3/3 哈希碰撞在建表时就拦下")
    else:
        raise AssertionError("应当拒绝重复哈希")


if __name__ == "__main__":
    _selftest()
