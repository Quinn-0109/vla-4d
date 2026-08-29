"""
K 帧输入的数据侧 —— 带**间隔**的历史窗口。

官方 RLDS 链路只支持**连续**窗口（`traj_transforms.chunk_act_obs` 里
`tf.range(-window_size+1, 1)`）。而 `docs/06` §4.1 实测定下 stride=16：
K=8 取连续帧只覆盖 0.4 秒，跑出"历史没用"是必然的假阴性。

两条路，选了第二条：

  ① `window_size = (K-1)*stride + 1 = 113`，再在 batch transform 里隔取。
     **否决**：解码发生在分窗**之后**（`apply_frame_transforms` 用 `dl.vmap`
     对窗口里每一帧解码+resize），113 帧全解码只用 8 帧，14 倍的白工。
  ② 换掉分窗函数本身，只 gather 需要的 8 个下标。**本模块做的就是这个。**

    from data.kframe import patch_strided_chunking, KFrameBatchTransform
    patch_strided_chunking(stride=16)      # 必须在构造 RLDSDataset 之前调用
    dataset = RLDSDataset(..., 见 scripts/finetune_kframe.py)

`python src/data/kframe.py` 跑自检（纯 numpy，不需要 tensorflow）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Type

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


# ---------------------------------------------------------------- 索引数学
def strided_chunk_indices(traj_len: int, k: int, stride: int) -> np.ndarray:
    """
    (traj_len, k) 的**未截断**下标：第 t 行是 [t-(k-1)s, …, t-s, t]。

    这是纯 numpy 的**参考实现**，下面 TF 版逐行对应它。
    索引数学放在能在 CPU 上自检的地方，是因为它错了不会崩——
    只会让历史窗口指向别的时刻，然后训练出一个"历史没用"的结论。
    """
    offs = np.arange(-(k - 1) * stride, 1, stride)          # (k,)
    return np.arange(traj_len)[:, None] + offs[None, :]     # (T, k)


def strided_pad_fraction(traj_len: int, k: int, stride: int) -> float:
    """窗口里被截断（回到 episode 开头之前）的帧占比。"""
    idx = strided_chunk_indices(traj_len, k, stride)
    return float((idx < 0).mean())


# ---------------------------------------------------------------- TF 侧替换
def strided_chunk_act_obs(traj: Dict, window_size: int, stride: int = 1,
                          k: int = None, future_action_window_size: int = 0) -> Dict:
    """
    `traj_transforms.chunk_act_obs` 的带间隔替换版。签名兼容，多 `stride` 与 `k`。

    ⚠️⚠️ **`k` 必须由我们自己带进来，不能指望 `window_size`。**
    openvla 的 `RLDSDataset.__init__` 把 `traj_transform_kwargs=dict(window_size=1, …)`
    **写死**在构造函数里，外面传不进去；我们只替换了分窗函数，拿到的 window_size
    永远是 1，`tf.range(0, 1, stride)` 于是只剩当前帧 —— **K 静默退化成 1**。
    下游不会报任何错：pixel_values 变成 (B, 6, H, W)，视觉主干照跑，
    池化把 256 个 patch 池成 256 个槽，序列长仍是 302，loss 照降、acc 照升。
    只有把它和 K=8 的评测放在一起时才会露馅。

    与官方的**两处**差异，都要写清楚：

    1. **观测按 stride 取**：偏移 `range(-(K-1)*s, 1, s)` 而非 `range(-K+1, 1)`。
       越界处与官方一样 clamp 到 0（即重复第 0 帧），并由 `pad_mask` 标出。

    2. ⚠️ **动作只取当前帧（及 future）**，不跟着 window_size 变长。
       官方 window_size=1 时 `action` 轴长 1，于是 `RLDSBatchTransform` 里
       `rlds_batch["action"][0]` 就是当前动作。**一旦 window_size=K，
       `action[0]` 变成 K 步之前的动作**，而这不会报任何错——
       模型会被训成"用当前画面预测很久以前的动作"，loss 照样下降，
       只是学到的东西是错的。与其在下游记得改成 `action[-1]`，
       不如在这里把动作轴钉死成"当前 + future"，让官方 batch transform
       的 `action[0]` 语义**继续成立**。
    """
    import tensorflow as tf

    if k is None:
        k = window_size
    elif window_size != 1:
        raise RuntimeError(
            f"官方传进来的 window_size={window_size}，不再是写死的 1 —— "
            "openvla 改了 RLDSDataset，这个覆盖式的补丁要重新对一遍。")

    traj_len = tf.shape(traj["action"])[0]

    # ⓪ ⭐ **在这里给每帧打指纹** —— G4/M2 靠它把离线渲染的深度查回来。
    #    这个位置是唯一可行的：更早没有统一的 image_primary 键，更晚图像已被
    #    解码 + 增广（frame transform 里），按内容做指纹就不成立了。
    #    指纹随观测一起被 gather，于是每个窗口的 K 帧各自带着自己的键。
    #    见 data/depth_cache.py —— 哈希函数只在那里定义一次，两边共用。
    img = traj["observation"].get("image_primary")
    if img is not None and img.dtype == tf.string:
        from data.depth_cache import hash_bytes
        traj["observation"]["img_hash"] = hash_bytes(img)
    elif img is not None:
        # 不报错就意味着 G4 拿不到深度而没人知道 —— octo 改了解码时机就得改这里
        raise RuntimeError(
            f"image_primary 在分窗时已不是未解码字节（dtype={img.dtype}）——"
            "指纹的前提没了。octo 的解码时机变了，depth_cache 的键要重新设计。")

    # ① 观测：带间隔的历史窗口
    obs_offsets = tf.range(-(k - 1) * stride, 1, stride)                    # (K,)
    chunk_indices = tf.broadcast_to(obs_offsets, [traj_len, k]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, k]
    )
    floored = tf.maximum(chunk_indices, 0)
    traj["observation"] = tf.nest.map_structure(
        lambda x: tf.gather(x, floored), traj["observation"])
    traj["observation"]["pad_mask"] = chunk_indices >= 0

    # ② 动作：当前 + future，长度与 window_size 无关（见 docstring 第 2 条）
    act_offsets = tf.range(0, 1 + future_action_window_size)
    n_act = 1 + future_action_window_size
    action_indices = tf.broadcast_to(act_offsets, [traj_len, n_act]) + tf.broadcast_to(
        tf.range(traj_len)[:, None], [traj_len, n_act]
    )
    if "timestep" in traj["task"]:
        goal_timestep = traj["task"]["timestep"]
    else:
        goal_timestep = tf.fill([traj_len], traj_len - 1)
    action_indices = tf.minimum(tf.maximum(action_indices, 0), goal_timestep[:, None])
    traj["action"] = tf.gather(traj["action"], action_indices)
    return traj


def patch_strided_chunking(stride: int, k: int) -> None:
    """
    把官方链路里的分窗函数换成带间隔的版本。**必须在构造 RLDSDataset 之前调用。**

    `apply_trajectory_transforms` 里是 `partial(traj_transforms.chunk_act_obs, …)`，
    在**构造数据集时**取属性，所以替换模块属性即可，不必改 openvla 仓库的文件
    （改了的话每次 `git pull` 官方仓库都要重打补丁，而且显存实测与训练两边
     容易走岔——这个项目在 LoRA 目标层上已经栽过一次）。
    """
    from functools import partial

    from prismatic.vla.datasets.rlds import traj_transforms

    assert k >= 1 and stride >= 1, (k, stride)
    if getattr(traj_transforms.chunk_act_obs, "_strided", False):
        raise RuntimeError("已经打过补丁了，重复调用会把 stride 叠起来")
    patched = partial(strided_chunk_act_obs, stride=stride, k=k)
    patched._strided = True                                  # type: ignore[attr-defined]
    traj_transforms.chunk_act_obs = patched                  # type: ignore[assignment]


# ---------------------------------------------------------------- batch transform
@dataclass
class KFrameBatchTransform:
    """
    官方 `RLDSBatchTransform` 的 K 帧版。差别只有图像那一段。

    `pixel_values` 排成 **(K*6, H, W)**：每帧 6 通道（DINOv2 3 + SigLIP 3），
    沿通道拼接。这与 `scripts/probe_vram.py:patch_multiframe` 用的约定一致——
    两边不一致的话，显存实测测的就是另一个模型。
    """
    action_tokenizer: Any
    base_tokenizer: Any
    image_transform: Any
    prompt_builder_fn: Type
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from PIL import Image

        dataset_name = rlds_batch["dataset_name"]
        action = rlds_batch["action"][0]          # 当前动作，见 strided_chunk_act_obs ②
        frames = rlds_batch["observation"]["image_primary"]        # (K, H, W, 3)
        pad_mask = np.asarray(rlds_batch["observation"]["pad_mask"])  # (K,) bool
        img_hash = rlds_batch["observation"].get("img_hash")           # (K,) uint64
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        prompt_builder = self.prompt_builder_fn("openvla")
        for turn in ({"from": "human",
                      "value": f"What action should the robot take to {lang}?"},
                     {"from": "gpt", "value": self.action_tokenizer(action)}):
            prompt_builder.add_turn(turn["from"], turn["value"])
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(),
                                        add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        pv = torch.cat([self.image_transform(Image.fromarray(np.asarray(f)))
                        for f in frames], dim=0)              # (K*6, H, W)

        out = dict(pixel_values=pv, input_ids=input_ids, labels=labels,
                   frame_pad_mask=torch.from_numpy(pad_mask.astype(np.bool_)),
                   dataset_name=dataset_name)
        if img_hash is not None:
            # int64 载 uint64：torch 没有 uint64，查表时再转回去（见 depth_cache）
            out["img_hash"] = torch.from_numpy(
                np.asarray(img_hash).astype(np.int64))
        return out


# ---------------------------------------------------------------- collator
@dataclass
class PaddedCollatorKFrame:
    """
    官方 `PaddedCollatorForActionPrediction` 的 K 帧版，多带一个 `frame_pad_mask`。

    ⚠️ **`frame_pad_mask` 必须一路传到池化算子**。episode 开头的窗口会把第 0 帧
    重复若干次（官方 clamp 行为），不屏蔽的话：K 份完全相同的内容落在 K 个不同的
    t 箱里，白占 token 预算，而"跨帧合并率"这个断言反而被这些假跨帧撑高——
    §3.0.5 ③ 那道闸门会失效。
    """
    model_max_length: int
    pad_token_id: int
    padding_side: str = "right"

    def __call__(self, instances) -> Dict[str, torch.Tensor]:
        assert self.padding_side == "right", "训练期只支持 right padding（与官方一致）"
        input_ids = pad_sequence([x["input_ids"] for x in instances],
                                 batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence([x["labels"] for x in instances],
                              batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        return dict(
            pixel_values=torch.stack([x["pixel_values"] for x in instances]),
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
            labels=labels,
            frame_pad_mask=torch.stack([x["frame_pad_mask"] for x in instances]),
            dataset_names=[x["dataset_name"] for x in instances],
            **({"img_hash": torch.stack([x["img_hash"] for x in instances])}
               if "img_hash" in instances[0] else {}),
        )


# ---------------------------------------------------------------- 自检
def _selftest() -> None:
    k, s = 8, 16
    idx = strided_chunk_indices(200, k, s)
    assert idx.shape == (200, k)
    assert idx[199].tolist() == [199 - s * i for i in range(k - 1, -1, -1)]
    assert idx[199][-1] == 199 and idx[199][0] == 199 - 112
    assert (np.diff(idx[199]) == s).all()
    print("✅ 1/4 索引：最后一帧是当前帧，间隔恒为 stride")

    assert (strided_chunk_indices(50, 1, s) == np.arange(50)[:, None]).all()
    assert (strided_chunk_indices(50, k, 1) ==
            np.arange(50)[:, None] + np.arange(-k + 1, 1)[None, :]).all()
    print("✅ 2/4 退化：K=1 还原单帧，stride=1 还原官方连续窗口")

    # 填充率：不是边角情况。stride=16/K=8 的窗口跨 113 帧，
    # 而 libero_spatial 的 episode 平均才 106 帧。
    for name, T, want_all_padded in (("libero_10", 388, False),
                                     ("libero_object", 140, False),
                                     ("libero_spatial", 106, True)):
        f = strided_pad_fraction(T, k, s)
        full = (strided_chunk_indices(T, k, s) >= 0).all(axis=1).mean()
        print(f"   {name:<15} 平均 {T:>3} 帧 → 窗口内填充帧占 {f:>5.1%}，"
              f"历史完整的时刻占 {full:>5.1%}")
        if want_all_padded:
            assert full == 0.0, "spatial 应当没有任何一个时刻拥有完整历史"
    print("✅ 3/4 填充率：spatial 上无一时刻有完整历史 —— 必须屏蔽，不是边角情况")

    class _T:
        def __call__(self, img):
            return torch.zeros(6, 4, 4)

    tf_ = KFrameBatchTransform.__new__(KFrameBatchTransform)
    pv = torch.cat([_T()(None) for _ in range(k)], dim=0)
    assert pv.shape == (k * 6, 4, 4), pv.shape
    del tf_
    coll = PaddedCollatorKFrame(model_max_length=32, pad_token_id=0)
    batch = coll([dict(pixel_values=pv, input_ids=torch.arange(5),
                       labels=torch.arange(5), frame_pad_mask=torch.ones(k, dtype=torch.bool),
                       dataset_name="d") for _ in range(2)])
    assert batch["pixel_values"].shape == (2, k * 6, 4, 4)
    assert batch["frame_pad_mask"].shape == (2, k)
    print("✅ 4/4 打包：pixel_values (B, K*6, H, W)，与 probe_vram 的约定一致")


if __name__ == "__main__" and __package__ in (None, ""):
    # 直接 `python src/.../x.py` 跑自检时把 src/ 放上 sys.path。
    # 各脚本用的都是 `from common.x import ...` 这种以 src/ 为根的写法
    # （scripts/*.py 里那句 sys.path.insert(..., "src") 就是干这个的），
    # 少了这几行就只有 rope4d 一个模块得换个跑法，是纯粹的绊脚石。
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    _selftest()
