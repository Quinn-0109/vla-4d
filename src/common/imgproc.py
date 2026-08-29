"""
评测侧的图像预处理 —— **与训练侧的增广对齐**。

训练用 `image_aug=True`（`scripts/finetune_kframe.py`）。openvla 的 RLDS 链路
把它展开成 `random_resized_crop(scale=[0.9, 0.9], ratio=[1.0, 1.0])` + 色彩抖动：
**每一帧都被裁掉外围 ~5%，再拉回 224**。模型于是从头到尾只见过"裁过的"画面。

评测若直接喂整张 224，视野比训练时大一圈、物体比训练时小一圈 ——
这不会报任何错，模型照样吐出 7 个合法动作，只是它在一个没见过的分布上。
官方 `run_libero_eval.py` 因此把 `--center_crop True` 定为"训练开了增广就必须开"，
本模块就是那一段（官方实现在 `experiments/robot/openvla_utils.py::crop_and_resize`）。

⚠️ **数值要与官方对齐**，所以这里逐条照抄 `tf.image.crop_and_resize` 的双线性
语义（角点对齐：`in = y1*(H-1) + i*(y2-y1)*(H-1)/(out-1)`），而不是随手 PIL 裁一刀
——插值核与取整都不同，而差异小到只能从成功率上看出来，正是最难查的那类。
`_selftest` 在装了 tensorflow 的机器上会**逐像素与 TF 对拍**；没装就只跑性质检查。

    python src/common/imgproc.py      # 自检
"""

import numpy as np

CROP_SCALE = 0.9        # = RLDS random_resized_crop 的 scale 下界（也是上界）


def center_crop_resize(img, crop_scale: float = CROP_SCALE, out: int = None):
    """
    (H, W, 3) uint8 → 中心裁 `sqrt(crop_scale)` 边长、再双线性拉回 (out, out)。

    `out` 缺省为输入边长（评测里就是 224）。
    """
    img = np.asarray(img)
    assert img.ndim == 3 and img.dtype == np.uint8, (img.shape, img.dtype)
    h, w = img.shape[:2]
    out = out or h

    x = img.astype(np.float32) / 255.0            # = tf.image.convert_image_dtype
    side = float(np.clip(np.sqrt(crop_scale), 0.0, 1.0))
    off = (1.0 - side) / 2.0
    y1, y2 = off, off + side

    def axis(n_in: int):
        # tf.image.crop_and_resize 的坐标：归一化框 × (n_in - 1)，输出角点对齐
        scale = (y2 - y1) * (n_in - 1) / (out - 1) if out > 1 else 0.0
        return y1 * (n_in - 1) + np.arange(out, dtype=np.float64) * scale

    def lerp(a, lo, hi, frac, ax):
        f = np.expand_dims(frac, tuple(i for i in range(a.ndim) if i != ax))
        return np.take(a, lo, axis=ax) * (1 - f) + np.take(a, hi, axis=ax) * f

    for ax, n_in in ((0, h), (1, w)):
        c = axis(n_in)
        lo = np.floor(c).astype(np.int64)
        hi = np.minimum(lo + 1, n_in - 1)
        x = lerp(x, lo, hi, (c - lo).astype(np.float32), ax)

    # = tf.image.convert_image_dtype(..., uint8, saturate=True)
    return np.clip(np.floor(np.clip(x, 0.0, 1.0) * 255.0 + 0.5), 0, 255).astype(np.uint8)


def _selftest() -> None:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    c = center_crop_resize(img)
    assert c.shape == img.shape and c.dtype == np.uint8
    print("✅ 1/4 形状与 dtype 不变")

    flat = np.full((224, 224, 3), 137, np.uint8)
    assert np.abs(center_crop_resize(flat).astype(int) - 137).max() <= 1
    print("✅ 2/4 纯色不变（插值无偏置）")

    # 裁的是外围：把外框涂白，裁完应当几乎看不到白边
    marked = np.zeros((224, 224, 3), np.uint8)
    b = int(224 * (1 - np.sqrt(CROP_SCALE)) / 2)
    marked[:b] = marked[-b:] = marked[:, :b] = marked[:, -b:] = 255
    got = center_crop_resize(marked)
    assert got.mean() < marked.mean() / 4, (got.mean(), marked.mean())
    assert b >= 5, b
    print(f"✅ 3/4 裁掉的是外围 {b} 像素带（{marked.mean():.1f} → {got.mean():.1f}）")

    try:
        import tensorflow as tf
    except ImportError:
        print("⚠️ 4/4 跳过：本机没有 tensorflow，无法与官方实现对拍。"
              "**在训练机上再跑一次这个自检**——这一条才是真正的判据。")
        return
    x = tf.image.convert_image_dtype(tf.convert_to_tensor(img), tf.float32)
    side = float(np.clip(np.sqrt(CROP_SCALE), 0.0, 1.0))
    off = (1.0 - side) / 2.0
    box = tf.constant([[off, off, off + side, off + side]], tf.float32)
    ref = tf.image.crop_and_resize(x[None], box, tf.range(1), (224, 224))[0]
    ref = tf.image.convert_image_dtype(tf.clip_by_value(ref, 0, 1), tf.uint8,
                                       saturate=True).numpy()
    d = np.abs(c.astype(int) - ref.astype(int))
    assert d.max() <= 1, f"与 TF 差 {d.max()}（均值 {d.mean():.3f}）"
    print(f"✅ 4/4 与 tf.image.crop_and_resize 对拍：最大差 {d.max()}，"
          f"不同像素占 {(d > 0).mean():.4%}")


if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    _selftest()
