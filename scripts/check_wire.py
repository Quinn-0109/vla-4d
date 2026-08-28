#!/usr/bin/env python
"""
挂点验证 —— **在真 7B 上跑一次前向，确认接线真的生效**。

`src/pooling/wire.py` 的 7 项自检用的是桩数据，只证明逻辑自洽。
真正的风险在**挂点**：transformers 的内部结构一变，挂上去的东西就没人调用，
而 Llama 会照用自己的 1D RoPE 跑完，**loss 正常下降、结论全错**。

四项检查，后两项是能证伪的：

  A 序列长度与 loss  —— 形状对不对，数值有没有炸
  B assert_rope_active —— 我们的 rotary_emb 到底被调用了几次
  C ⭐ **改深度 → G4 的 logits 必须变，G3 必须不变**
      G3 的坐标是图像网格，与深度无关；G4 是度量坐标。
      G4 不变 = 度量坐标根本没进到模型里，那"G4"就是个带深度通道的 G3。
  D ⭐ **G4 与 M2 的 logits 必须不同**
      两者池化侧逐位相同，只有 PE 侧坐标不同。
      若输出相同 = PE 侧坐标没起作用，**标题句的机制通路不存在**，
      而 G4 vs M2 正是强命题的唯一直接证据。

    export OPENVLA_ROOT=<openvla 路径>
    python scripts/check_wire.py --arms G1,G3,G4,M2

只做推理，不反传；显存与评测同量级（K=8 不压缩约 16.7 GB）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = os.environ.get("OPENVLA_ROOT")
if _ROOT is None:
    raise SystemExit("请先 export OPENVLA_ROOT=<openvla 仓库路径>")
sys.path.insert(0, str(Path(_ROOT).expanduser().resolve()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from common.camera import Camera  # noqa: E402
from pooling.wire import (WireConfig, assert_rope_active, set_batch,  # noqa: E402
                          unwire, wire)

CKPT = "openvla/openvla-7b"
N_PATCH = 256


def load_depth_and_cams(cache_dir: str, cam_json: str, k: int, b: int):
    """优先用真值深度缓存；没有就用合成的，并**明确说出来**。"""
    cams_raw = json.loads(Path(cam_json).read_text()) if os.path.exists(cam_json) else {}
    cams = {int(t): Camera(fovy=v["fovy"], height=v["height"], width=v["width"],
                           pos=torch.tensor(v["pos"], dtype=torch.float64),
                           rot=torch.tensor(v["rot"], dtype=torch.float64).reshape(3, 3),
                           flipped=v.get("flipped", True))
            for t, v in cams_raw.items()}
    files = sorted(Path(cache_dir).glob("*_t*_e*.npy")) if os.path.isdir(cache_dir) else []
    picked = []
    for f in files:
        import re
        m = re.search(r"_t(\d+)_e\d+\.npy$", f.name)
        if m and int(m.group(1)) in cams:
            picked.append((int(m.group(1)), f))
        if len(picked) >= b:
            break
    if len(picked) < b or not cams:
        print("  ⚠️ 没有可用的真值深度/相机，改用**合成深度**。"
              "形状与流程能验，数值分布不能当结论。")
        cam = next(iter(cams.values()), None) or Camera(
            fovy=45.0, height=224, width=224,
            pos=torch.zeros(3, dtype=torch.float64),
            rot=torch.eye(3, dtype=torch.float64), flipped=True)
        depth = 1.0 + 0.3 * torch.rand(b, k, N_PATCH).double()
        return depth, [cam] * b, False
    depth, out_cams = [], []
    for tid, f in picked[:b]:
        a = np.load(f).astype(np.float32)[..., 1]        # 均值通道
        idx = np.linspace(0, a.shape[0] - 1, k).astype(int)
        depth.append(torch.from_numpy(a[idx].reshape(k, -1)).double())
        out_cams.append(cams[tid])
    return torch.stack(depth), out_cams, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="G1,G3,G4,M2")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cache_dir", default="results/depth_cache")
    ap.add_argument("--camera", default="results/tables/camera_libero.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 GPU"
    dev, K, B = "cuda", args.K, args.batch

    depth, cams, real = load_depth_and_cams(args.cache_dir, args.camera, K, B)
    xyz = torch.cat([cams[i].patch_xyz(depth[i]).reshape(-1, 3) for i in range(B)])
    bbox = torch.stack([torch.quantile(xyz, 0.01, dim=0),
                        torch.quantile(xyz, 0.99, dim=0)]).float()
    print(f"深度来源：{'真值缓存' if real else '合成'}   "
          f"包围盒 " + " ".join(f"{a}[{float(bbox[0][i]):+.2f},{float(bbox[1][i]):+.2f}]"
                                for i, a in enumerate("xyz")))

    processor = AutoProcessor.from_pretrained(CKPT, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True, attn_implementation="flash_attention_2").to(dev).eval()

    prompt = ("In: What action should the robot take to pick up the black bowl?\nOut:")
    enc = processor.tokenizer(prompt, return_tensors="pt")
    ids = enc.input_ids.to(dev).expand(B, -1).contiguous()
    att = torch.ones_like(ids)
    labels = ids.clone()
    px = torch.randn(B, K * 6, 224, 224, dtype=torch.bfloat16, device=dev)

    def run(arm, dep):
        cfg = WireConfig(arm=arm, K=K,
                         bbox=bbox if arm in ("G4", "M2") else None)
        st = wire(model, cfg)
        try:
            set_batch(st, depth=dep, frame_pad_mask=torch.ones(B, K, dtype=torch.bool),
                      cameras=cams)
            with torch.no_grad():
                o = model(input_ids=ids, attention_mask=att, pixel_values=px,
                          labels=labels)
            if arm != "G0":
                assert_rope_active(st)
            return o.logits.float().cpu(), float(o.loss), st.rope_calls
        finally:
            unwire(model, st)

    res = {}
    for arm in args.arms.split(","):
        arm = arm.strip()
        lg, loss, calls = run(arm, depth)
        res[arm] = lg
        n_layers = len(model.language_model.model.layers)
        print(f"\n[{arm}] 序列 {lg.shape[1]}  loss {loss:.4f}  "
              f"rotary_emb 调用 {calls}/{n_layers} 层")
        assert np.isfinite(loss), f"{arm} 的 loss 不是有限数"

    print(f"\n{'='*60}\n=== C: 改深度 → G4 变、G3 不变 ===")
    dep2 = depth + 0.15                      # 整体推远 15 cm
    ok_c = True
    for arm, must_change in (("G3", False), ("G4", True)):
        if arm not in res:
            continue
        lg2, _, _ = run(arm, dep2)
        d = float((lg2 - res[arm]).abs().max())
        changed = d > 1e-3
        flag = "✅" if changed == must_change else "❌"
        ok_c &= changed == must_change
        print(f"  {flag} {arm}: |Δlogits|max = {d:.2e}   "
              f"（应当{'变' if must_change else '不变'}）")
    if not ok_c:
        print("  ❌ G4 不随深度变 = 度量坐标没进到模型里，那个'G4'是带深度通道的 G3。")

    print(f"\n=== D: G4 与 M2 必须不同（池化侧逐位相同，只差 PE 坐标）===")
    if "G4" in res and "M2" in res:
        d = float((res["G4"] - res["M2"]).abs().max())
        if d > 1e-3:
            print(f"  ✅ |Δlogits|max = {d:.2e}，PE 侧坐标确实起作用")
        else:
            print(f"  ❌ |Δlogits|max = {d:.2e} —— PE 侧坐标没起作用。"
                  "\n     G4 vs M2 是强命题的唯一直接证据，这条通路不存在就没得测。")
    else:
        print("  （没同时跑 G4 与 M2，跳过）")


if __name__ == "__main__":
    main()
