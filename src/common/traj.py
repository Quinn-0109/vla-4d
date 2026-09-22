"""
演示素材的落盘与读回 —— **写与读必须在同一个文件里**。

`run_eval_kframe.py --dump_traj N` 在评测时顺带把每个 task 前 N 局的
(观测, 动作, 分配统计) 抄一份；`read_episode` 是回放侧唯一的入口。
两侧共用这里的格式约定，演示因此**不需要 GPU**：回放已有记录即可
（缺了记录才得重新推理，那要显存）。

每局两个文件，同一个 stem：
  `<stem>.jsonl`  第 1 行表头（run_id/arm/task/success/…），其后逐步一行
                  `{"t", "action"[7], "n_valid", "alloc" | null}`
  `<stem>.npz`    `rgb` (T,224,224,3) uint8 —— **送进模型之前的原图**，
                  `t` (T,) 对应步号。`--dump_rgb False` 时没有这个文件。

⚠️ 表头里的 run_id 是这段轨迹的**唯一凭据**。没有它，两段来自不同臂或
   不同 checkpoint 的轨迹在文件里看起来一模一样 —— 演示时张冠李戴不会报错。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def write_episode(dump_dir: Path, head: dict, steps: list,
                  rgb: Optional[list] = None) -> Path:
    """写一局。返回 jsonl 的路径。`rgb` 为空/None 时不写 npz。"""
    if rgb and len(rgb) != len(steps):
        # ⚠️ 硬拦。两者是逐步一一对应的（同一个 live 行、同一步采的），
        #    对不上说明采集点被改动过 —— 而错位的演示**看起来完全正常**：
        #    画面是这一步的，动作是另一步的，只是解释不通。
        raise ValueError(f"RGB {len(rgb)} 帧与动作 {len(steps)} 步对不上，拒绝落盘")
    dump_dir = Path(dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{head['arm']}-task{head['task_id']}-ep{head['episode']}"
    path = dump_dir / (stem + ".jsonl")
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(dict(head, header=1, steps=len(steps)),
                           ensure_ascii=False) + "\n")
        for st in steps:
            f.write(json.dumps(st) + "\n")
    if rgb:
        np.savez_compressed(dump_dir / (stem + ".npz"),
                            rgb=np.stack(rgb),
                            t=np.array([st["t"] for st in steps]))
    return path


def read_episode(path) -> tuple[dict, list, Optional[np.ndarray]]:
    """读回一局：`(表头, 逐步记录, rgb 或 None)`。"""
    path = Path(path)
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    if not lines or not lines[0].get("header"):
        raise ValueError(f"{path} 没有表头行 —— 不知道它是哪次测量的，不可用于演示")
    head, steps = lines[0], lines[1:]
    if head["steps"] != len(steps):
        # 评测中途被 kill 时会出现。**说出来**，不要默默少几步。
        raise ValueError(f"{path} 表头记 {head['steps']} 步，实有 {len(steps)} 步"
                         "（多半是评测被中断）")
    npz = path.with_suffix(".npz")
    rgb = None
    if npz.is_file():
        with np.load(npz) as z:
            rgb, t = z["rgb"], z["t"]
            if len(rgb) != len(steps) or [int(x) for x in t] != [s["t"] for s in steps]:
                raise ValueError(f"{npz} 的帧与 {path} 的步号不对应")
    return head, steps, rgb
