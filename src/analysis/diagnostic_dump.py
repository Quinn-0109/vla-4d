"""案例 manifest 校验与可恢复的诊断材料写入；不依赖模型或仿真器。"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import atexit
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_case_manifest(path: Path, repo: Path | None = None) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema_version") != 1 or doc.get("purpose") != "mechanism_diagnosis_and_demo_only":
        raise ValueError("不是受支持的机制诊断 manifest")
    if doc.get("statistical_use") != "forbidden":
        raise ValueError("manifest 必须明确禁止统计推断")
    cases, seen = doc.get("cases"), set()
    if not isinstance(cases, list) or not cases:
        raise ValueError("manifest 没有案例")
    for row in cases:
        key = (row.get("task_id"), row.get("episode"))
        if (type(key[0]) is not int or type(key[1]) is not int or min(key) < 0
                or row.get("g3_success") not in (0, 1)):
            raise ValueError(f"非法案例: {row}")
        if key in seen:
            raise ValueError(f"重复案例: {key}")
        seen.add(key)
    source = doc.get("source", {})
    digest = source.get("sha256", "")
    if len(digest) != 64:
        raise ValueError("manifest 缺少正式逐局记录 SHA-256")
    if repo is not None:
        source_path = Path(source.get("path", ""))
        source_path = source_path if source_path.is_absolute() else repo / source_path
        if not source_path.is_file():
            raise ValueError(f"找不到 manifest 来源文件: {source_path}")
        if sha256(source_path) != digest:
            raise ValueError("manifest 来源文件 SHA-256 已变化")
    return doc


def cases_by_task(doc: dict) -> dict[int, list[int]]:
    out = {}
    for row in doc["cases"]:
        out.setdefault(row["task_id"], []).append(row["episode"])
    return {task: sorted(episodes) for task, episodes in sorted(out.items())}


def observation_state(obs: dict) -> dict:
    """只摘机器人低维状态；图像与深度由单独文件保存。"""
    wanted = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
              "robot0_joint_pos", "ee_pos", "ee_quat", "gripper_states",
              "joint_states")
    out = {}
    for key in wanted:
        if key not in obs:
            continue
        value = obs[key]
        out[key] = value.tolist() if hasattr(value, "tolist") else value
    return out


def _read_rows(path: Path, required: set[str]) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows, seen = [], set()
    with opener(path, "rt", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - set(row)
            if row.get("schema_version") != 1 or missing:
                raise ValueError(f"{path}:{lineno}: schema 不兼容或缺字段 {sorted(missing)}")
            key = (row["arm"], row["task_id"], row["episode"], row["env_step"])
            if key in seen:
                raise ValueError(f"{path}:{lineno}: 重复记录 {key}")
            seen.add(key)
            rows.append(row)
    return rows


def read_case_dump(root: Path) -> dict:
    """读取一个已原子发布的案例目录，并核对轨迹/分配逐步对齐。"""
    root = Path(root)
    if root.name.endswith(".partial"):
        raise ValueError("拒绝把 .partial 目录当成完整诊断产物")
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    diag = meta.get("diagnostic_dump", {})
    if diag.get("schema_version") != 1 or diag.get("statistical_use") != "forbidden":
        raise ValueError("meta 缺少诊断 schema 或统计用途边界")
    base = {"schema_version", "arm", "task_id", "episode", "env_step"}
    trajectory = _read_rows(
        root / "trajectory.jsonl",
        base | {"source_steps", "frame_pad_mask", "raw_action", "executed_action",
                "observation_before", "observation_after", "success", "frame"})
    allocation = _read_rows(
        root / "alloc.jsonl.gz",
        base | {"slots_used", "per_frame", "slots", "latest_frame_mean_slot_weight",
                "mixed_frame_slot_fraction", "padding_assigned_patches"})
    def keys(rows):
        return {(r["arm"], r["task_id"], r["episode"], r["env_step"]) for r in rows}
    if keys(trajectory) != keys(allocation):
        raise ValueError("trajectory 与 alloc 的动作步集合不一致")
    return {"meta": meta, "trajectory": trajectory, "allocation": allocation}


class DiagnosticDump:
    """逐行写入轨迹与 gzip 分配记录，成功后原子发布整个目录。"""

    def __init__(self, final_dir: Path, metadata: dict):
        self.final_dir = Path(final_dir)
        self.partial_dir = self.final_dir.with_name(self.final_dir.name + ".partial")
        if self.final_dir.exists() or self.partial_dir.exists():
            raise FileExistsError(f"诊断输出已存在，拒绝覆盖: {self.final_dir}")
        self.partial_dir.mkdir(parents=True)
        (self.partial_dir / "frames").mkdir()
        self._traj = (self.partial_dir / "trajectory.jsonl").open("x", encoding="utf-8")
        self._alloc = gzip.open(self.partial_dir / "alloc.jsonl.gz", "xt", encoding="utf-8")
        self._closed = False
        self._published = False
        atexit.register(self._atexit_close)
        self.write_json("meta.json", metadata)

    def write_json(self, name: str, value: dict) -> None:
        p = self.partial_dir / name
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(p)

    def write_trajectory(self, row: dict) -> None:
        self._traj.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._traj.flush()

    def write_alloc(self, row: dict) -> None:
        self._alloc.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._alloc.flush()

    def save_frame(self, relative: str, image) -> str:
        from PIL import Image
        p = self.partial_dir / "frames" / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(p, quality=90)
        return (Path("frames") / relative).as_posix()

    def _close_files(self) -> None:
        if not self._closed:
            self._traj.close()
            self._alloc.close()
            self._closed = True

    def finalize(self) -> Path:
        self._close_files()
        os.replace(self.partial_dir, self.final_dir)
        self._published = True
        return self.final_dir

    def abort(self) -> Path:
        """保留 `.partial` 供排错，不把它冒充完整产物。"""
        self._close_files()
        return self.partial_dir

    def _atexit_close(self) -> None:
        # 普通异常退出时补齐 gzip 尾部，同时保留 `.partial` 标识不完整。
        # SIGKILL 无法执行 Python 清理，但仍不会发布成正式目录。
        if not self._published:
            self._close_files()
