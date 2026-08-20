"""
explore_libero.py —— 动手熟悉 LIBERO，不加载任何模型。

在真正跑 OpenVLA 之前先用这个把环境摸清楚：任务长什么样、观测里有什么、
20Hz 是多快、动作各维度分别控制什么。

    python scripts/explore_libero.py list                      # 列出所有 suite 和任务
    python scripts/explore_libero.py inspect                   # 打印观测结构与动作空间
    python scripts/explore_libero.py render --task_id 0         # 随机动作录一段视频
    python scripts/explore_libero.py probe                      # 逐维测试动作的作用
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")   # 必须在 import mujoco 之前设

import numpy as np

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _make_env(suite: str, task_id: int, resolution: int = 256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    task = task_suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl,
                             camera_heights=resolution, camera_widths=resolution)
    # 即使用固定初始状态，seed 仍会影响物体位置 —— 官方脚本也这么做
    env.seed(0)
    return env, task, task_suite


def cmd_list(args):
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    for suite in SUITES:
        ts = bd[suite]()
        print(f"\n{'='*70}\n{suite}  ({ts.n_tasks} tasks)\n{'='*70}")
        for i in range(min(ts.n_tasks, args.limit)):
            print(f"  [{i:2d}] {ts.get_task(i).language}")
        if ts.n_tasks > args.limit:
            print(f"  ... 还有 {ts.n_tasks - args.limit} 个 (--limit 调整)")


def cmd_inspect(args):
    env, task, ts = _make_env(args.suite, args.task_id)
    print(f"\nSuite: {args.suite} | Task {args.task_id}")
    print(f"指令: {task.language}")
    print(f"BDDL: {task.bddl_file}")

    init_states = ts.get_task_init_states(args.task_id)
    print(f"初始状态: {np.asarray(init_states).shape}  (50 个 episode 各一个)")

    env.reset()
    obs = env.set_init_state(init_states[0])

    print(f"\n{'='*70}\n观测字典\n{'='*70}")
    for k in sorted(obs.keys()):
        v = np.asarray(obs[k])
        note = ""
        if k == "agentview_image":
            note = "  <- 第三人称，⚠️ 上下颠倒，用前要 [::-1,::-1]"
        elif k == "robot0_eye_in_hand_image":
            note = "  <- 腕部相机"
        elif k == "robot0_eef_pos":
            note = "  <- 末端执行器位置"
        elif k == "robot0_eef_quat":
            note = "  <- 末端执行器四元数"
        elif k == "robot0_gripper_qpos":
            note = "  <- 夹爪关节"
        print(f"  {k:34s} {str(v.shape):16s} {v.dtype}{note}")

    print(f"\n{'='*70}\n动作空间 (7 维)\n{'='*70}")
    for i, d in enumerate(["Δx  平移-x", "Δy  平移-y", "Δz  平移-z",
                           "Δroll   旋转-r", "Δpitch  旋转-p", "Δyaw    旋转-y",
                           "gripper 夹爪 (-1=张开, +1=闭合)"]):
        print(f"  [{i}] {d}")
    print("\n控制频率 20 Hz -> dt = 0.05 s")
    env.close()


def cmd_render(args):
    import imageio
    env, task, ts = _make_env(args.suite, args.task_id)
    print(f"指令: {task.language}")
    env.reset()
    obs = env.set_init_state(ts.get_task_init_states(args.task_id)[args.episode])

    rng = np.random.default_rng(0)
    frames, done = [], False
    for t in range(args.steps):
        if t < 10:
            action = [0, 0, 0, 0, 0, 0, -1]        # 前 10 步空转，等物体落稳
        else:
            action = np.concatenate([rng.uniform(-0.3, 0.3, 6), [-1.0]]).tolist()
        obs, reward, done, info = env.step(action)
        frames.append(obs["agentview_image"][::-1, ::-1])   # 转 180°
        if done:
            print(f"任务在第 {t} 步完成(随机动作蒙对了)")
            break

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimsave(args.out, frames, fps=20)
    print(f"已保存 {len(frames)} 帧 -> {args.out}  (成功={done})")
    env.close()


def cmd_probe(args):
    """逐维施加动作，观察末端执行器怎么动 —— 建立对动作空间的直觉。"""
    env, task, ts = _make_env(args.suite, args.task_id)
    names = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

    print(f"任务: {task.language}\n")
    print(f"{'维度':<10}{'施加值':>8}{'末端位移 (m)':>34}{'夹爪变化':>12}")
    print("-" * 68)

    for dim in range(7):
        env.reset()
        obs = env.set_init_state(ts.get_task_init_states(args.task_id)[0])
        for _ in range(10):                                  # 先等落稳
            obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])

        p0 = np.array(obs["robot0_eef_pos"])
        g0 = np.array(obs["robot0_gripper_qpos"]).copy()

        val = 0.5 if dim < 6 else 1.0
        action = [0.0] * 6 + [-1.0]
        action[dim] = val
        for _ in range(20):                                  # 持续施加 1 秒
            obs, *_ = env.step(action)

        dp = np.array(obs["robot0_eef_pos"]) - p0
        dg = np.array(obs["robot0_gripper_qpos"]) - g0
        print(f"{names[dim]:<10}{val:>8.1f}"
              f"{f'[{dp[0]:+.4f}, {dp[1]:+.4f}, {dp[2]:+.4f}]':>34}"
              f"{np.abs(dg).sum():>12.4f}")

    print("\n注: 前 3 维直接推动末端位置；4-6 维改变姿态(位置变化小)；第 7 维只动夹爪。")
    env.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("list", help="列出所有 suite 和任务")
    p.add_argument("--limit", type=int, default=10)
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("inspect", help="打印观测结构与动作空间")
    p.add_argument("--suite", default="libero_spatial", choices=SUITES)
    p.add_argument("--task_id", type=int, default=0)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("render", help="随机动作录视频")
    p.add_argument("--suite", default="libero_spatial", choices=SUITES)
    p.add_argument("--task_id", type=int, default=0)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--out", default="results/libero_explore.mp4")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("probe", help="逐维测试动作作用")
    p.add_argument("--suite", default="libero_spatial", choices=SUITES)
    p.add_argument("--task_id", type=int, default=0)
    p.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
