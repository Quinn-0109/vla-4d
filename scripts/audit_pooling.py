"""无需 7B/仿真器的池化结构诊断。结果不代表成功率或真实视觉特征损失。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def audit(k=8, budget=256, n_t=2, enforce_n=0, partition="quantile"):
    import torch
    from pooling.coord_pool import coord_bin_pool, grid_coords, grid_extent
    if k <= 0 or budget <= 0 or not 1 <= n_t <= k or not 0 <= enforce_n <= budget:
        raise ValueError("要求 K,budget>0、1<=n_t<=K、0<=enforce_n<=budget")
    coords = grid_coords(k).unsqueeze(0)
    lo, hi = grid_extent(k)
    # 特征只是携带坐标的探针，不用它推断真实模型的信息损失。
    features = coords.clone()
    rows = []
    for real in range(1, k+1):
        valid = (torch.arange(k) >= k - real).repeat_interleave(256).unsqueeze(0)
        for arm, kw in (("G2", dict(group_axes=(0,), n_group=(k,))), ("G3", dict(n_t=n_t))):
            out = coord_bin_pool(features, coords, budget, lo, hi, valid=valid,
                                 enforce_n=enforce_n or None, partition=partition, **kw)
            used = int(out.mask.sum())
            dropped = int((valid & (out.assign < 0)).sum())
            spans = []
            for slot in out.mask[0].nonzero().flatten():
                times = coords[0, out.assign[0] == slot, 0]
                spans.append(float(times.max() - times.min()))
            rows.append(dict(arm=arm,real_frames=real,output_slots=out.feat.shape[1],
                             used_slots=used,empty_slots=out.feat.shape[1]-used,
                             dropped_valid_patches=dropped,
                             latest_frame_kept=int((out.assign[0,-256:] >= 0).sum()),
                             cross_frame_slots=sum(v > 0 for v in spans)))
    return {"K": k,"budget": budget,"n_t": n_t,"enforce_n": enforce_n,"partition": partition,"rows": rows,
            "scope": "规则网格与补帧掩码诊断；不是 LIBERO 评测，不含真实深度或视觉特征"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--n_t", type=int, default=2)
    ap.add_argument("--enforce_n", type=int, default=0)
    ap.add_argument("--partition", choices=("quantile", "quantile_fixed", "voxel", "fps"), default="quantile")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = audit(args.K, args.budget, args.n_t, args.enforce_n, args.partition)
    print("arm real_frames output used empty dropped latest_kept cross_frame")
    for r in report["rows"]:
        print(" ".join(str(r[k]) for k in ("arm","real_frames","output_slots","used_slots",
                                          "empty_slots","dropped_valid_patches","latest_frame_kept","cross_frame_slots")))
    print(report["scope"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
