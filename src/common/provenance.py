"""实验配置与来源记录；纯标准库，可在无 GPU 的机器上检查。"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


def config_dict(cfg) -> dict:
    return json.loads(json.dumps(asdict(cfg) if is_dataclass(cfg) else dict(cfg), default=str))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_identity(root: Path) -> dict:
    # 包含未提交代码，避免把同一 HEAD 下的不同补丁误记为同一实现。
    h = hashlib.sha256()
    for folder in ("src", "scripts"):
        for p in sorted((root / folder).rglob("*.py")):
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"commit": commit, "source_sha256": h.hexdigest()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


TRAIN_FIELDS = ("arm", "K", "stride", "budget", "n_t", "enforce_n", "no_subset",
                "dataset_name", "vla_path", "learning_rate", "image_aug", "lora_rank",
                "lora_dropout", "lora_vision", "seed", "shuffle_buffer_size", "partition")
WIRE_FIELDS = ("arm", "K", "stride", "budget", "n_t", "enforce_n", "vla_path", "partition")


def check_config(saved: dict, current: dict, fields=TRAIN_FIELDS) -> None:
    changes = {k: (saved.get(k), current.get(k)) for k in fields
               if saved.get(k) != current.get(k)}
    if changes:
        raise ValueError(f"配置与 checkpoint 来源不一致: {changes}")


def training_record(adapter: Path | None) -> dict | None:
    if adapter is None:
        return None
    p = adapter.parents[1] / "run_config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def prepare_training(cfg, run_dir: Path, repo: Path, resume: Path | None) -> dict:
    current = config_dict(cfg)
    subset = Path(current["subset_dir"]) / "subset.json"
    cache = Path(current["subset_dir"]) / "depth_cache.npz"
    filtered = current["arm"] in ("G3", "M3", "M2", "G4") and not current["no_subset"]
    dataset = {"name": current["dataset_name"], "selection": "sub255" if filtered else "full",
               "subset_sha256": sha256(subset) if filtered and subset.is_file() else None,
               "depth_cache_sha256": sha256(cache) if filtered and cache.is_file() else None}
    p = run_dir / "run_config.json"
    saved = training_record(resume)
    if saved:
        check_config(saved["config"], current)
        if saved.get("dataset") != dataset:
            raise ValueError("续训的数据子集或深度缓存发生变化，拒绝混用")
    elif resume:
        print("旧 checkpoint 没有 run_config.json：无法验证历史配置，续训也不是逐位复现。")
    if not resume and ((run_dir / "adapter").exists() or (run_dir / "metrics.jsonl").exists()):
        raise ValueError(f"{run_dir} 已有训练产物；请续训或使用新的 run_id_note")
    if p.is_file():
        record = json.loads(p.read_text(encoding="utf-8"))
        check_config(record["config"], current)
        if record.get("dataset") != dataset:
            raise ValueError("运行目录记录的数据来源与当前输入不同")
    else:
        record = {"schema_version": 1, "config": current, "dataset": dataset, "code": code_identity(repo),
                  "created_at": datetime.now(timezone.utc).isoformat(),
                  "resume_source": str(resume) if resume else None,
                  "historical_config_verified": not resume or bool(saved and saved.get("historical_config_verified"))}
        write_json(p, record)
    if resume and (not saved or not saved.get("historical_config_verified")
                   or saved.get("code") != code_identity(repo)):
        record["historical_config_verified"] = False
        write_json(p, record)
        print("续训存在历史来源缺失或代码变化：记录为不可自动核验，保留 sessions.jsonl 供人工审计。")
    with (run_dir / "sessions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"started_at": datetime.now(timezone.utc).isoformat(),
                            "config": current, "code": code_identity(repo),
                            "resume_source": str(resume) if resume else None}, ensure_ascii=False) + "\n")
    return record


def validate_config(cfg, training: bool) -> None:
    c = config_dict(cfg)
    if c["arm"] not in ("G0", "G2", "G3", "M2", "M3", "G4"):
        raise ValueError("arm 不支持；G1 的定义尚未确定，不能启动")
    for k in ("K", "stride", "budget", "n_t"):
        if type(c[k]) is not int or c[k] <= 0:
            raise ValueError(f"{k} 必须是正整数")
    if c["arm"] == "G0" and c["K"] != 1:
        raise ValueError("G0 必须 K=1")
    if c["enforce_n"] < 0 or c["enforce_n"] > c["budget"]:
        raise ValueError("enforce_n 必须在 0..budget 内")
    if c["partition"] not in ("quantile", "quantile_fixed", "voxel", "fps"):
        raise ValueError("未知 partition")
    if training:
        if c["no_subset"] and c["arm"] != "G3":
            raise ValueError("no_subset 仅用于 G3-full")
        if c["micro"] < 0 or (c["micro"] and 16 % c["micro"]):
            raise ValueError("micro 必须是 0 或有效批 16 的正因子")
        if c["K"] not in (1, 8) and not c["micro"]:
            raise ValueError("K 不在已测微批表中，请显式给 --micro 并先测吞吐")
        for k in ("max_steps", "save_steps", "log_steps", "eval_steps"):
            if c[k] <= 0:
                raise ValueError(f"{k} 必须 >0")
        if c["keep_last"] < 0 or c["keep_every"] < 0:
            raise ValueError("checkpoint 保留参数不能为负")
    else:
        if not 1 <= c["num_trials_per_task"] <= 50 or c["eval_batch"] <= 0:
            raise ValueError("评测要求每任务 1..50 局、eval_batch >0")
        if c["verify_batch"] and c["eval_batch"] != 1:
            raise ValueError("verify_batch 仅支持 eval_batch=1")
        if c["start_task"] < 0 or c["end_task"] < -1:
            raise ValueError("非法 task 范围")
        if c["end_task"] != -1 and c["end_task"] <= c["start_task"]:
            raise ValueError("task 范围必须非空，end_task 为开区间")


def eval_paths(cfg, root: Path) -> tuple[str, dict]:
    c = config_dict(cfg)
    # 完整 checkpoint 路径、配置及本地源代码参与身份，不仅是 step30000 目录名。
    identity = {k: v for k, v in c.items() if k not in
                ("overwrite", "need_gb", "local_log_dir", "verify_env", "verify_batch", "verify_vision_cache",
                 "start_task", "end_task")}
    identity["code"] = code_identity(root)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    step = Path(c["adapter"]).name if c["adapter"] else "base"
    segment = f"-t{c['start_task']}_{c['end_task']}" if c["start_task"] or c["end_task"] >= 0 else ""
    run_id = (f"EVAL-{c['task_suite_name']}-{c['arm']}-K{c['K']}s{c['stride']}"
              f"-seed{c['seed']}-{step}-b{c['eval_batch']}-N{c['budget']}-nt{c['n_t']}"
              f"-r{digest}{segment}" + (f"--{c['run_note']}" if c["run_note"] else ""))
    paths = {k: Path(c["local_log_dir"]) / (run_id + ext) for k, ext in
             (("log", ".txt"), ("episodes", ".episodes.jsonl"), ("meta", ".meta.json"))}
    if not c["overwrite"] and any(p.exists() for p in paths.values()):
        raise ValueError(f"已有同配置评测记录（含未完成记录）: {run_id}；换 run_note 保留两份，或显式 overwrite")
    return run_id, paths
