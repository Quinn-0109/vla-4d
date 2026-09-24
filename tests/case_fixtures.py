"""合成的固定案例目录，**逐字段模仿** `run_eval_kframe.py` 的诊断导出。

只给测试和本地预览用：它让阶段 D 的表与演示在没有 GPU、没有仿真器时就能写完并测好。
分配规则是示意性的（不跑真实池化算子），但三种结构与真实方法一致：
G0 单帧不池化、G2 每帧独立 32 个槽、G3 按两段时间箱跨帧合并。
文件名不以 test 开头，不会被 unittest 当成测试收集。
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.alloc_stats import summarize_assignment  # noqa: E402
from analysis.diagnostic_dump import DiagnosticDump  # noqa: E402

PATCHES = 256
STRIDE = 16

DEFAULT_CASES = [
    {"task_id": 2, "episode": 0, "g3_success": 1, "stratum": "success"},
    {"task_id": 2, "episode": 1, "g3_success": 0, "stratum": "failure"},
]


def window(step: int, k: int, stride: int = STRIDE):
    """与 run_eval_kframe.build_window 同一条取帧规则：旧 → 新，不足则重复最早一步。"""
    steps, mask = [], []
    for i in range(k - 1, -1, -1):
        j = i * stride
        if j <= step:
            steps.append(step - j)
            mask.append(True)
        else:
            steps.append(0)
            mask.append(False)
    return steps, mask


def assignment(arm: str, mask: list[bool]) -> list[int]:
    k = len(mask)
    real = [f for f in range(k) if mask[f]]
    half = math.ceil(len(real) / 2)
    out = []
    for f in range(k):
        for p in range(PATCHES):
            if not mask[f]:
                out.append(-1)
            elif arm == "G0":
                out.append(p)
            elif arm == "G2":
                out.append(f * 32 + p // 8)
            else:                                   # G3：两段时间箱 × 128 个空间箱
                tb = 0 if real.index(f) < half else 1
                out.append(tb * 128 + p // 2)
    return out


def _image(step: int, arm: str):
    import numpy as np
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    img[..., 0] = (step * 7) % 256
    img[..., 1] = {"G0": 60, "G2": 140, "G3": 220}.get(arm, 100)
    img[40:180, (step * 3) % 180:(step * 3) % 180 + 40, 2] = 255
    return img


def make_case_dir(root: Path, arm: str, cases=None, outcomes=None, *, frames: bool = True,
                  manifest_sha: str = "m" * 64, name: str | None = None,
                  extra_episode: tuple | None = None, nan_at: tuple | None = None) -> Path:
    """写一个已发布的案例目录并返回路径。

    ``outcomes``：{(task, ep): (步数, 本次是否成功)}；默认成功局 40 步、失败局 70 步。
    ``extra_episode`` / ``nan_at`` 用来构造应被验收拦下的坏数据。
    """
    cases = cases or DEFAULT_CASES
    k = 1 if arm == "G0" else 8
    outcomes = outcomes or {(c["task_id"], c["episode"]): (40, 1) if c["g3_success"] else (70, 0)
                            for c in cases}
    meta = {
        "schema_version": 1,
        "config": {"arm": arm, "K": k, "stride": STRIDE, "budget": 256, "n_t": 2,
                   "partition": "quantile_fixed", "eval_batch": 1},
        "code": {"commit": "0" * 40, "source_sha256": "s" * 64},
        "adapter": f"runs/{arm}/adapter/step30000",
        "adapter_sha256": hashlib.sha256(arm.encode()).hexdigest(),
        "diagnostic_dump": {
            "schema_version": 1,
            "purpose": "mechanism_diagnosis_and_demo_only",
            "statistical_use": "forbidden",
            "manifest_path": "results/cases/g3_fixed_diagnostic_cases.json",
            "manifest_sha256": manifest_sha,
            "manifest": {"cases": cases},
        },
    }
    final = Path(root) / (name or f"EVAL-libero_10-{arm}-synthetic")
    dump = DiagnosticDump(final, meta)
    lookup = {(c["task_id"], c["episode"]): c for c in cases}
    episodes = list(outcomes.items())
    if extra_episode:
        episodes.append((extra_episode, (5, 0)))
        lookup[extra_episode] = {"g3_success": 0, "stratum": "failure"}
    for (task, ep), (n_steps, ok) in episodes:
        for s in range(n_steps):
            steps, mask = window(s, k)
            stats = summarize_assignment(assignment(arm, mask), mask, steps, PATCHES)
            stats.update({"arm": arm, "task_id": task, "episode": ep, "env_step": s})
            dump.write_alloc(stats)
            grip = 1.0 if (s // 15) % 2 else -1.0
            raw = [0.01 * math.sin(s / 5 + i) for i in range(6)] + [grip]
            if nan_at == (task, ep, s):
                raw[0] = float("nan")
            frame = dump.save_frame(f"task{task:02d}/ep{ep:02d}/step{s:04d}.jpg",
                                    _image(s, arm)) if frames else None
            dump.write_trajectory({
                "schema_version": 1, "arm": arm, "task_id": task, "episode": ep,
                "stratum": lookup[(task, ep)]["stratum"],
                "g3_reference_success": int(lookup[(task, ep)]["g3_success"]),
                "env_step": s, "source_steps": steps, "frame_pad_mask": mask,
                "raw_action": raw, "executed_action": raw[:6] + [-grip],
                "observation_before": {"robot0_eef_pos": [0.0, 0.0, 1.0]},
                "observation_after": {"robot0_eef_pos": [0.0, 0.0, 1.0]},
                "success": int(bool(ok) and s == n_steps - 1),
                "frame": frame,
            })
    return dump.finalize()


def manifest_json(cases=None) -> str:
    return json.dumps({"cases": cases or DEFAULT_CASES})
