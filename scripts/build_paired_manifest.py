"""从显式指定的逐局文件及其自动元数据生成分析清单；不猜测旧实验来源。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.provenance import sha256
from paired_analysis import ARMS, load


def build(files_by_arm: dict, output: Path) -> dict:
    arms = {}
    common = None
    for arm, files in files_by_arm.items():
        if not files:
            continue
        entry = None
        for path in files:
            path = Path(path).resolve()
            meta_path = path.with_name(path.name.removesuffix(".episodes.jsonl") + ".meta.json")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            training = meta.get("training")
            if not training or not training.get("historical_config_verified"):
                raise ValueError(f"{path}: 无可核对训练来源，按 docs/paired-analysis.md 人工归档旧日志")
            c, tc, ds = meta["config"], training["config"], training["dataset"]
            shared = {
                "training": {k: tc[k] for k in ("learning_rate", "image_aug", "lora_rank",
                              "lora_dropout", "lora_vision", "shuffle_buffer_size", "vla_path", "partition")},
                "dataset": ds, "action_stats": meta["action_stats"],
                "initial_states": meta["initial_states_sha256"],
                "training_code": training["code"]["source_sha256"],
            }
            if common is not None and shared != common:
                raise ValueError(f"{path}: 训练协议/数据/初始状态/统计量与其他实验臂不一致")
            common = shared
            if c["arm"] != arm or tc["arm"] != arm or ds["selection"] != "sub255":
                raise ValueError(f"{path}: 实验臂或训练数据不属于固定子集 2×2")
            checkpoint = Path(meta["adapter"])
            protocol = {k: c[k] for k in ("K", "stride", "budget", "n_t", "enforce_n",
                                         "center_crop", "eval_batch", "seed", "num_trials_per_task")}
            protocol.update(suite=c["task_suite_name"], dataset_sha256=ds["subset_sha256"],
                            checkpoint_step=int(checkpoint.name.removeprefix("step")),
                            training_seed=tc.get("seed"), eval_code_commit=meta["code"]["source_sha256"])
            candidate = {"dataset": "sub255", "checkpoint": str(checkpoint), "protocol": protocol}
            # 同臂分段必须来自同一权重；不同训练配置不能靠相同的 step 名混用。
            provenance = {"adapter_sha256": meta["adapter_sha256"], "training": training,
                          "action_stats": meta["action_stats"]}
            if entry is None:
                entry = dict(candidate, files=[], provenance=provenance)
            elif any(entry[k] != v for k, v in candidate.items()) or entry["provenance"] != provenance:
                raise ValueError(f"{arm}: 分段来源不一致")
            entry["files"].append({"path": path.as_posix(), "sha256": sha256(path)})
        arms[arm] = entry
    # 不覆盖已有人工清单；先临时核验，成功才交付。
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = {"arms": arms}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "manifest.json"
        p.write_text(json.dumps(spec), encoding="utf-8")
        load(p)
    with output.open("x", encoding="utf-8") as f:
        json.dump(spec, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return spec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        ap.add_argument(f"--{arm}", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    try:
        build({a: getattr(args, a) for a in ARMS}, args.out)
    except (OSError, ValueError, KeyError, TypeError) as e:
        ap.exit(1, f"拒绝生成: {e}\n")
    print(f"已生成并核验 {args.out}")


if __name__ == "__main__":
    main()
