"""
run_libero_eval_traj.py —— OpenVLA LIBERO 评测 + 轨迹记录

官方的 experiments/robot/libero/run_libero_eval.py 只输出两样东西：
每个 episode 一个 MP4，以及一份记录成功/失败的 txt。
动作序列 (7-DoF) 和末端执行器位姿在循环里每步都在手边，但用完就丢了。

本脚本在保持评测逻辑与官方逐行一致的前提下，额外落盘每一步的:
    - 模型原始输出动作 action_raw (7)
    - 送入环境的动作 action_env (7,  经 gripper 归一化+翻转)
    - 末端执行器位置 eef_pos (3) / 四元数 eef_quat (4)
    - 夹爪关节 gripper_qpos
    - 仿真步索引 t
用于计算轨迹平滑度等官方未提供的指标。

用法:
    export OPENVLA_ROOT=/path/to/openvla
    python scripts/run_libero_eval_traj.py \
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
        --task_suite_name libero_spatial \
        --center_crop True \
        --num_trials_per_task 50
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm

# ---- 让解释器找到 openvla 的 experiments 包 -------------------------------
_OPENVLA_ROOT = os.environ.get("OPENVLA_ROOT")
if _OPENVLA_ROOT is None:
    raise SystemExit(
        "请先设置 OPENVLA_ROOT 环境变量，指向 openvla 仓库根目录。\n"
        "  export OPENVLA_ROOT=$HOME/vla-work/openvla"
    )
sys.path.insert(0, str(Path(_OPENVLA_ROOT).expanduser().resolve()))

from libero.libero import benchmark  # noqa: E402

from experiments.robot.libero.libero_utils import (  # noqa: E402
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.robot.openvla_utils import get_processor  # noqa: E402
from experiments.robot.robot_utils import (  # noqa: E402
    DATE_TIME,
    get_action,
    get_image_resize_size,
    get_model,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)

# LIBERO 各 suite 的最大步数（数值取自官方脚本，勿改）
MAX_STEPS = {
    "libero_spatial": 220,   # 最长训练 demo 193 步
    "libero_object": 280,    # 最长训练 demo 254 步
    "libero_goal": 300,      # 最长训练 demo 270 步
    "libero_10": 520,        # 最长训练 demo 505 步
    "libero_90": 400,        # 最长训练 demo 373 步
}


@dataclass
class GenerateConfig:
    # fmt: off
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # 重要: 官方 checkpoint 用 90% 随机裁剪增强训练，测试时必须开中心裁剪
    center_crop: bool = True

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10          # 等物体在仿真里落稳
    num_trials_per_task: int = 50

    run_id_note: Optional[str] = None
    local_log_dir: str = "./experiments/logs"

    # --- 本脚本新增 ---
    traj_dir: str = "./results/trajectories"   # 逐步轨迹落盘目录
    save_video: bool = True                    # 满量评测时建议关掉(500个MP4很占盘)

    use_wandb: bool = False
    wandb_project: str = "vla-4d"
    wandb_entity: str = ""

    seed: int = 7
    # fmt: on


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint, "必须指定 --pretrained_checkpoint"
    if "image_aug" in str(cfg.pretrained_checkpoint):
        assert cfg.center_crop, "该 checkpoint 用图像增强训练过，center_crop 必须为 True"
    assert not (cfg.load_in_8bit and cfg.load_in_4bit), "8bit 与 4bit 不能同时开"

    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = cfg.task_suite_name

    model = get_model(cfg)

    # 数据集若带 _no_noops 后缀，反归一化的 key 需要跟着改
    if cfg.model_family == "openvla":
        if cfg.unnorm_key not in model.norm_stats and f"{cfg.unnorm_key}_no_noops" in model.norm_stats:
            cfg.unnorm_key = f"{cfg.unnorm_key}_no_noops"
        assert cfg.unnorm_key in model.norm_stats, f"norm_stats 里找不到 {cfg.unnorm_key}"

    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-seed{cfg.seed}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id += f"--{cfg.run_id_note}"

    os.makedirs(cfg.local_log_dir, exist_ok=True)
    traj_dir = Path(cfg.traj_dir) / run_id
    traj_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    print(f"日志: {log_file.name}\n轨迹: {traj_dir}")

    if cfg.use_wandb:
        import wandb
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=run_id)

    task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    resize_size = get_image_resize_size(cfg)
    max_steps = MAX_STEPS[cfg.task_suite_name]

    total_episodes, total_successes = 0, 0
    per_task_results = []

    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"task{task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []
            steps = []   # <-- 本脚本的核心新增

            while t < max_steps + cfg.num_steps_wait:
                try:
                    # 前若干步空转，等仿真器把物体放稳
                    if t < cfg.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    if cfg.save_video:
                        replay_images.append(img)

                    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                    eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64)
                    grip_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate((eef_pos, quat2axisangle(eef_quat), grip_qpos)),
                    }

                    action_raw = get_action(cfg, model, observation, task_description, processor=processor)

                    # 夹爪动作 [0,1] -> [-1,+1]，环境期望后者
                    action = normalize_gripper_action(action_raw, binarize=True)
                    # OpenVLA 的 dataloader 翻转过夹爪符号，执行前翻回来
                    if cfg.model_family == "openvla":
                        action = invert_gripper_action(action)

                    steps.append({
                        "t": t,
                        "action_raw": np.asarray(action_raw, dtype=np.float64).tolist(),
                        "action_env": np.asarray(action, dtype=np.float64).tolist(),
                        "eef_pos": eef_pos.tolist(),
                        "eef_quat": eef_quat.tolist(),
                        "gripper_qpos": grip_qpos.tolist(),
                    })

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:  # noqa: BLE001  与官方脚本保持一致的容错行为
                    print(f"异常: {e}")
                    log_file.write(f"Caught exception: {e}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            # 落盘该 episode 的完整轨迹
            with open(traj_dir / f"task{task_id:02d}_ep{episode_idx:03d}.json", "w") as f:
                json.dump({
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "task_description": task_description,
                    "task_suite": cfg.task_suite_name,
                    "success": bool(done),
                    "num_steps": len(steps),
                    "num_steps_wait": cfg.num_steps_wait,
                    "seed": cfg.seed,
                    "checkpoint": str(cfg.pretrained_checkpoint),
                    "steps": steps,
                }, f)

            if cfg.save_video and replay_images:
                save_rollout_video(replay_images, total_episodes, success=done,
                                   task_description=task_description, log_file=log_file)

            log_file.write(f"episode={total_episodes} success={done} "
                           f"rate={total_successes/total_episodes*100:.1f}%\n")
            log_file.flush()

        sr = task_successes / max(task_episodes, 1)
        per_task_results.append({
            "task_id": task_id,
            "task_description": task_description,
            "num_episodes": task_episodes,
            "num_successes": task_successes,
            "success_rate": sr,
        })
        print(f"[task {task_id}] {task_description}  ->  {sr*100:.1f}%")
        log_file.write(f"[task {task_id}] success_rate={sr:.4f}\n")

    overall = total_successes / max(total_episodes, 1)

    # per-task 分解 —— 官方只给 suite 平均，这份是我们自己的
    summary = {
        "run_id": run_id,
        "task_suite": cfg.task_suite_name,
        "checkpoint": str(cfg.pretrained_checkpoint),
        "seed": cfg.seed,
        "num_trials_per_task": cfg.num_trials_per_task,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall,
        "per_task": per_task_results,
    }
    with open(traj_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}\n{cfg.task_suite_name}  总成功率: {overall*100:.1f}%  "
          f"({total_successes}/{total_episodes})\n{'='*60}")
    log_file.write(f"FINAL success_rate={overall:.4f}\n")
    log_file.close()

    if cfg.use_wandb:
        import wandb
        wandb.log({"success_rate": overall})
        wandb.finish()


if __name__ == "__main__":
    eval_libero()
