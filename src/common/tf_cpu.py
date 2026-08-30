"""
把 GPU 从 TensorFlow 眼前藏起来 —— **凡是"用 tfds 读数据 + 用 GPU 干别的"的脚本都要调**。

TF 默认在第一次用到 GPU 时把**整张卡的显存全部预留**（不是按需分配）。
本项目里 tfds 只用来读 tfrecord 和解 JPEG，一个 GPU 算子都不需要，
但那 22 GB 一占，同机的训练/评测就直接 OOM ——

    torch.cuda.OutOfMemoryError: Tried to allocate 20.00 MiB.
    Process 1332 has 22.01 GiB memory in use.        ← 那是 build_subset 里的 TF

⚠️ **不能用 `CUDA_VISIBLE_DEVICES=""` 代替**：MuJoCo 的 EGL 渲染要真的用 GPU，
   整个进程屏蔽掉就只能退回 CPU 渲染，慢一个量级。要藏的只是 TF 那一半。

⚠️ 必须在**第一个 TF 算子之前**调用，晚了 TF 已经初始化，`set_visible_devices`
   会抛 RuntimeError。所以放在 import tensorflow 之后立刻调。
"""


def hide_gpu_from_tf() -> None:
    try:
        import tensorflow as tf
    except ImportError:
        return
    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as e:                 # 已经初始化过了
        print(f"⚠️ TF 已初始化，屏蔽 GPU 失败（{e}）—— 它可能已经占住显存了。"
              "把这个调用挪到更靠前的位置。")
