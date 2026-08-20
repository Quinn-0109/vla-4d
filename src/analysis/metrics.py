"""
轨迹平滑度与时空连贯性指标。

OpenVLA 官方评测只报成功率。本模块提供论文里用来论证"时空不连贯"的量化指标,
是我们自己实验数据的核心。所有指标都在末端执行器轨迹或动作序列上计算。

约定: LIBERO 仿真控制频率 20Hz (robosuite 默认 control_freq)，dt = 0.05s。
"""

from __future__ import annotations

import numpy as np

DEFAULT_DT = 0.05  # LIBERO/robosuite 默认 20Hz


def _diff(x: np.ndarray, dt: float) -> np.ndarray:
    """沿时间轴一阶差分并除以 dt。x: (T, D) -> (T-1, D)"""
    return np.diff(x, axis=0) / dt


def velocity(pos: np.ndarray, dt: float = DEFAULT_DT) -> np.ndarray:
    """速度 (T-1, 3)"""
    return _diff(pos, dt)


def acceleration(pos: np.ndarray, dt: float = DEFAULT_DT) -> np.ndarray:
    return _diff(velocity(pos, dt), dt)


def jerk(pos: np.ndarray, dt: float = DEFAULT_DT) -> np.ndarray:
    return _diff(acceleration(pos, dt), dt)


def normalized_jerk(pos: np.ndarray, dt: float = DEFAULT_DT) -> float:
    """
    归一化 jerk —— 运动学里最标准的平滑度度量，值越小越平滑。

        NJ = sqrt( 0.5 * ∫|jerk|² dt * T⁵ / L² )

    对时长 T 和路径长度 L 做了归一化，因此可以跨任务、跨时长比较。
    参考 Hogan & Sternad 的 dimensionless jerk。
    """
    if len(pos) < 4:
        return float("nan")
    j = jerk(pos, dt)
    duration = len(pos) * dt
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if path_len < 1e-9 or duration < 1e-9:
        return float("nan")
    integral = float(np.sum(np.sum(j ** 2, axis=1)) * dt)
    return float(np.sqrt(0.5 * integral * duration ** 5 / path_len ** 2))


def spectral_arc_length(pos: np.ndarray, dt: float = DEFAULT_DT,
                        fc: float = 10.0, amp_th: float = 0.05) -> float:
    """
    SPARC (Spectral Arc Length) —— 速度谱的弧长，对噪声比 jerk 更鲁棒。
    值为负，越接近 0 越平滑。Balasubramanian et al. 2015。
    """
    v = np.linalg.norm(velocity(pos, dt), axis=1)
    if len(v) < 4:
        return float("nan")

    n = int(2 ** np.ceil(np.log2(len(v))) * 4)     # 补零提高频率分辨率
    spec = np.abs(np.fft.rfft(v, n=n))
    freq = np.fft.rfftfreq(n, d=dt)
    if spec.max() < 1e-12:
        return float("nan")
    spec = spec / spec.max()

    keep = freq <= fc
    freq, spec = freq[keep], spec[keep]
    idx = np.where(spec >= amp_th)[0]
    if len(idx) < 2:
        return float("nan")
    freq, spec = freq[idx[0]:idx[-1] + 1], spec[idx[0]:idx[-1] + 1]

    df = np.diff(freq) / (freq[-1] - freq[0])
    ds = np.diff(spec)
    return float(-np.sum(np.sqrt(df ** 2 + ds ** 2)))


def num_velocity_reversals(pos: np.ndarray, dt: float = DEFAULT_DT,
                           eps: float = 1e-4) -> int:
    """
    速度方向反转次数 —— 直接对应肉眼可见的"抖动/来回蹭"。
    对速度向量做相邻帧点积，符号由正变负即计一次。
    """
    v = velocity(pos, dt)
    if len(v) < 2:
        return 0
    dots = np.sum(v[:-1] * v[1:], axis=1)
    mags = np.linalg.norm(v[:-1], axis=1) * np.linalg.norm(v[1:], axis=1)
    valid = mags > eps
    return int(np.sum(dots[valid] < 0))


def num_idle_steps(pos: np.ndarray, thresh: float = 1e-4) -> int:
    """
    空转步数 —— 末端执行器位移低于阈值的步数。
    对应 VLA-4D 论文点名的 "idle pauses" 问题。
    """
    if len(pos) < 2:
        return 0
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return int(np.sum(step < thresh))


def path_efficiency(pos: np.ndarray) -> float:
    """
    路径效率 = 起终点直线距离 / 实际路径长度，∈(0,1]。
    越低说明绕路/冗余运动越多 —— VLA-4D 说 2D 模型有 "redundant global motion"。
    """
    if len(pos) < 2:
        return float("nan")
    path_len = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    if path_len < 1e-9:
        return float("nan")
    return float(np.linalg.norm(pos[-1] - pos[0]) / path_len)


def action_jitter(actions: np.ndarray) -> float:
    """
    动作抖动 —— 相邻两步动作向量差的均方根。
    直接作用在模型输出上，与仿真动力学无关，能把"模型输出不连贯"和
    "环境执行不连贯"区分开。
    """
    if len(actions) < 2:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=1))))


def gripper_flip_count(actions: np.ndarray, grip_dim: int = -1) -> int:
    """夹爪开合翻转次数 —— 反复开合是典型的时序不连贯表现。"""
    if len(actions) < 2:
        return 0
    g = np.sign(actions[:, grip_dim])
    return int(np.sum(g[:-1] != g[1:]))


def compute_all(traj: dict, dt: float = DEFAULT_DT) -> dict:
    """
    对一条 run_libero_eval_traj.py 落盘的轨迹算全部指标。

    traj: 加载后的 episode JSON。
    """
    steps = traj.get("steps", [])
    out = {
        "task_id": traj.get("task_id"),
        "episode_idx": traj.get("episode_idx"),
        "task_suite": traj.get("task_suite"),
        "task_description": traj.get("task_description"),
        "success": bool(traj.get("success", False)),
        "num_steps": len(steps),
    }
    if len(steps) < 4:
        return out

    pos = np.array([s["eef_pos"] for s in steps], dtype=np.float64)
    acts = np.array([s["action_env"] for s in steps], dtype=np.float64)

    out.update({
        "normalized_jerk": normalized_jerk(pos, dt),
        "sparc": spectral_arc_length(pos, dt),
        "velocity_reversals": num_velocity_reversals(pos, dt),
        "idle_steps": num_idle_steps(pos),
        "idle_ratio": num_idle_steps(pos) / max(len(pos) - 1, 1),
        "path_efficiency": path_efficiency(pos),
        "path_length": float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))),
        "action_jitter": action_jitter(acts),
        "gripper_flips": gripper_flip_count(acts),
        "mean_speed": float(np.mean(np.linalg.norm(velocity(pos, dt), axis=1))),
        "max_speed": float(np.max(np.linalg.norm(velocity(pos, dt), axis=1))),
        # 完成时间 —— 对齐 VLA-4D 论文的 CT 指标，便于横向比较
        "completion_time_s": len(steps) * dt,
    })
    return out
