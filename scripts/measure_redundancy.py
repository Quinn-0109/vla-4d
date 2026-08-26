"""
measure_redundancy.py —— 测量 LIBERO 观测中 patch 级的时空冗余

动机（见 docs/04）：VLA 要做时空推理就得吃历史帧，但每帧只有很小一块区域
在动，其余是静止背景。本脚本用数据检验这个假设。

**为什么可以回放而不必重跑模型**：评测是确定性的（见 docs/05 第 3.2 节），
回放已记录的 action_env 序列可得到逐帧完全相同的画面，
但无需任何 7B 前向 —— 只跑 MuJoCo 步进与渲染，快两个数量级。

两种度量：
  pixel   —— patch 内像素的平均绝对差。CPU 即可，用于快速看分布形态
  feature —— DINOv2 / SigLIP 倒数第二层 patch token 的余弦距离。
             需 GPU，但这才是模型真正"看到"的冗余

用法:
    export OPENVLA_ROOT=... MUJOCO_GL=egl
    export HF_ENDPOINT=https://hf-mirror.com   # feature 模式要下 timm 权重，直连常超时
    python scripts/measure_redundancy.py --mode pixel   --max_episodes 100
    python scripts/measure_redundancy.py --mode feature --max_episodes 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库路径>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))

from libero.libero import benchmark  # noqa: E402
from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
)

IMG_SIZE = 224      # OpenVLA 输入尺寸
PATCH = 14          # DINOv2 / SigLIP 的 patch 大小
GRID = IMG_SIZE // PATCH        # 16
N_PATCH = GRID * GRID           # 256

# 变化幅度阈值（相对该帧对内 patch 变化的最大值），用于统计"多少 patch 真的在动"
REL_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)


# ---------------------------------------------------------------- 像素度量
def patch_change_pixel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两帧 (H,W,3) uint8 -> 每个 patch 的平均绝对差，返回 (N_PATCH,)"""
    d = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(axis=2)   # (H,W)
    d = d[: GRID * PATCH, : GRID * PATCH]
    return d.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).reshape(-1)


# ---------------------------------------------------------------- 特征度量
class FeatureExtractor:
    """复用 OpenVLA 的视觉骨干：DINOv2 + SigLIP，均取倒数第二层 patch token。"""

    def __init__(self, which: str = "dino", device: str = "cuda"):
        import timm
        import torch

        self.torch = torch
        self.device = device
        name = {
            "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
            "siglip": "vit_so400m_patch14_siglip_224",
        }[which]
        try:
            self.model = timm.create_model(name, pretrained=True, num_classes=0, img_size=IMG_SIZE)
        except Exception as e:
            if "hf-mirror" not in os.environ.get("HF_ENDPOINT", ""):
                raise SystemExit(
                    f"下载 {name} 失败: {type(e).__name__}\n"
                    f"直连 huggingface.co 常超时，请先设镜像后重试:\n"
                    f"  export HF_ENDPOINT=https://hf-mirror.com\n"
                    f"(权重约 1.2 GB，下完即缓存到 $HF_HOME)"
                ) from e
            raise
        self.model.eval().to(device)
        cfg = timm.data.resolve_model_data_config(self.model)
        cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)
        self.tf = timm.data.create_transform(**cfg, is_training=False)
        # 与 openvla 一致：取倒数第二层
        self.layer_idx = len(self.model.blocks) - 2

    @staticmethod
    def _drop_prefix(tok, n_patch: int):
        """去掉 cls / register token，只留 patch token。"""
        return tok[:, -n_patch:, :] if tok.shape[1] > n_patch else tok

    def encode(self, frames: list[np.ndarray], chunk: int = 32):
        """
        (T,H,W,3) uint8 -> (T, N_PATCH, D) 归一化后的 patch 特征。

        ⚠️ **必须分块**。初版一次前向整条 episode，在 libero_10 上炸了：
        Long 平均 388 帧，DINOv2 的激活涨到 6 GB，加上同卡在跑的训练直接 OOM。
        这不只是"并行才有的问题"——单独跑长 suite 一样会撞，只是之前跑的都是
        短 suite（Spatial 成功 episode 约 106 帧）所以没暴露。

        chunk=32 把峰值钉在与 episode 长度无关的常数上，
        代价只是多几次 kernel 启动。显存紧张时可以再调小。
        """
        from PIL import Image
        torch = self.torch
        outs = []
        for i in range(0, len(frames), chunk):
            batch = frames[i:i + chunk]
            x = torch.stack([self.tf(Image.fromarray(f).convert("RGB"))
                             for f in batch]).to(self.device)
            with torch.no_grad():
                out = self.model.get_intermediate_layers(x, n={self.layer_idx})
                tok = out[0] if isinstance(out, (tuple, list)) else out
            tok = self._drop_prefix(tok, N_PATCH)
            outs.append(torch.nn.functional.normalize(tok.float(), dim=-1).cpu().numpy())
            del x, out, tok
        return np.concatenate(outs, axis=0)


def patch_change_feature(fa: np.ndarray, fb: np.ndarray) -> np.ndarray:
    """两帧归一化 patch 特征 (N_PATCH,D) -> 每 patch 的余弦距离"""
    return 1.0 - (fa * fb).sum(axis=-1)


# ---------------------------------------------------------------- 统计
def summarize(change: np.ndarray) -> dict:
    """把一帧对的 256 个 patch 变化值压成若干标量。"""
    total = float(change.sum())
    srt = np.sort(change)[::-1]
    out = {
        "mean": float(change.mean()),
        "max": float(change.max()),
        "std": float(change.std()),
    }
    if total > 1e-12:
        cum = np.cumsum(srt) / total
        # top-k% 的 patch 贡献了多少比例的总变化 —— 这是核心指标
        for k in (5, 10, 25, 50):
            out[f"top{k}pct_share"] = float(cum[max(int(N_PATCH * k / 100) - 1, 0)])
        # Gini 系数：0=均匀分布，1=完全集中
        idx = np.arange(1, N_PATCH + 1)
        out["gini"] = float((2 * (idx * np.sort(change)).sum()) / (N_PATCH * change.sum()) - (N_PATCH + 1) / N_PATCH)
    else:
        for k in (5, 10, 25, 50):
            out[f"top{k}pct_share"] = float("nan")
        out["gini"] = float("nan")

    mx = change.max()
    for th in REL_THRESHOLDS:
        out[f"frac_above_{th}"] = float((change > th * mx).mean()) if mx > 1e-12 else float("nan")
    return out, srt / total if total > 1e-12 else None


# ---------------------------------------------------------------- 回放
def replay_episode(traj: dict, task_suite, extractor=None, deltas=(1, 2, 4, 8)):
    """回放一条轨迹，返回 (每个 Δt 的统计列表, 平均排序剖面)"""
    task_id, ep_idx = traj["task_id"], traj["episode_idx"]
    task = task_suite.get_task(task_id)
    env, _ = get_libero_env(task, "openvla", resolution=256)
    env.seed(0)                                   # 与评测一致，勿改
    env.reset()
    obs = env.set_init_state(task_suite.get_task_init_states(task_id)[ep_idx])

    for _ in range(traj.get("num_steps_wait", 10)):   # 等物体落稳
        obs, *_ = env.step(get_libero_dummy_action("openvla"))

    frames = []
    for st in traj["steps"]:
        frames.append(get_libero_image(obs, IMG_SIZE))
        obs, *_ = env.step(st["action_env"])
    env.close()

    if len(frames) < max(deltas) + 1:
        return [], {}

    feats = extractor.encode(frames) if extractor is not None else None

    rows, profiles = [], {}
    for dt in deltas:
        stats_list, prof_list = [], []
        for t in range(len(frames) - dt):
            if feats is not None:
                ch = patch_change_feature(feats[t], feats[t + dt])
            else:
                ch = patch_change_pixel(frames[t], frames[t + dt])
            st, prof = summarize(ch)
            stats_list.append(st)
            if prof is not None:
                prof_list.append(prof)
        if not stats_list:
            continue
        row = {k: float(np.mean([s[k] for s in stats_list])) for k in stats_list[0]}
        row.update(task_suite=traj["task_suite"], task_id=task_id, episode_idx=ep_idx,
                   success=traj["success"], delta_t=dt, n_pairs=len(stats_list))
        rows.append(row)
        if prof_list:
            profiles[dt] = np.mean(prof_list, axis=0)
    return rows, profiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", default="results/trajectories")
    ap.add_argument("--mode", choices=["pixel", "feature"], default="pixel")
    ap.add_argument("--backbone", choices=["dino", "siglip"], default="dino")
    ap.add_argument("--max_episodes", type=int, default=100, help="每个 suite 采样多少条")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--out_dir", default="results/redundancy")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.traj_dir).rglob("task*_ep*.json"))
    if not files:
        raise SystemExit(f"{args.traj_dir} 下没有轨迹")

    # 按 suite 分组均匀采样，避免只测到某一个 suite
    by_suite: dict[str, list] = {}
    for f in files:
        with open(f) as fh:
            t = json.load(fh)
        by_suite.setdefault(t["task_suite"], []).append(t)
    for k in by_suite:
        rng = np.random.default_rng(0)
        rng.shuffle(by_suite[k])
        by_suite[k] = by_suite[k][: args.max_episodes]

    extractor = FeatureExtractor(args.backbone) if args.mode == "feature" else None
    print(f"模式={args.mode}" + (f" ({args.backbone})" if extractor else "")
          + f" | patch 网格 {GRID}×{GRID}={N_PATCH} | Δt={args.deltas}")

    all_rows, prof_acc = [], {}
    for suite, trajs in by_suite.items():
        print(f"\n[{suite}] {len(trajs)} 条")
        for i, tr in enumerate(trajs):
            try:
                rows, profs = replay_episode(tr, benchmark.get_benchmark_dict()[suite](),
                                             extractor, tuple(args.deltas))
            except Exception as e:                      # 单条失败不中断整体
                print(f"  ! ep{tr['episode_idx']} 跳过: {type(e).__name__}: {e}")
                continue
            all_rows.extend(rows)
            for dt, p in profs.items():
                prof_acc.setdefault((suite, dt), []).append(p)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(trajs)}")

    import pandas as pd
    df = pd.DataFrame(all_rows)
    tag = args.mode if args.mode == "pixel" else f"{args.mode}_{args.backbone}"
    df.to_csv(out / f"redundancy_{tag}.csv", index=False)

    # 排序剖面：用于画"top X% 的 patch 贡献了多少变化"的累积曲线
    np.savez_compressed(out / f"profiles_{tag}.npz",
                        **{f"{s}__dt{d}": np.mean(v, axis=0) for (s, d), v in prof_acc.items()})

    print(f"\n{'='*72}\n按 suite × Δt 汇总（{tag}）\n{'='*72}")
    cols = ["top5pct_share", "top10pct_share", "top25pct_share", "gini", "frac_above_0.1"]
    print(df.groupby(["task_suite", "delta_t"])[cols].mean().round(4).to_string())
    print(f"\n-> {out}/redundancy_{tag}.csv")


if __name__ == "__main__":
    main()
