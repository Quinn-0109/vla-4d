# LIBERO 使用指南

> 基于实际仓库代码核对（Lifelong-Robot-Learning/LIBERO），不是照抄 README。

## 1. LIBERO 是什么

一个**机器人操作的终身学习基准**。原论文关注前向/后向迁移，但 OpenVLA 只用它做
**监督微调 + 行为克隆评测**——我们跟 OpenVLA 保持一致。

核心设定（全部来自 `libero/libero/envs/env_wrapper.py` 的默认参数）：

| 项 | 值 | 备注 |
|---|---|---|
| 机械臂 | **Franka Panda** | ⚠️ 不是 WidowX，也不是 UR5/Dobot。立项书里写错的话要改 |
| 控制器 | `OSC_POSE` | 操作空间控制，输出末端位姿增量 |
| **控制频率** | **20 Hz** | 即 **dt = 0.05s**，算速度/加速度/jerk 的基准 |
| 相机 | `agentview`(第三人称) + `robot0_eye_in_hand`(腕部) | **两路都默认开启** |
| 默认分辨率 | 128×128 | OpenVLA 用 256×256（`get_libero_env(..., resolution=256)`） |
| 深度 | `camera_depths=False` | **可以打开**——后续做 4D 要用 |
| 仿真引擎 | MuJoCo (via robosuite 1.4.1) | 服务器上必须 `MUJOCO_GL=egl` |
| horizon | 1000 | 但 OpenVLA 按 suite 另设了 max_steps |

## 2. 五个任务套件

| Suite | 任务数 | 考察什么 | 每任务演示数 |
|---|---|---|---|
| `libero_spatial` | 10 | 相同物体、不同布局 → **空间关系理解** | 50 |
| `libero_object` | 10 | 相同布局、不同物体 → **物体类型理解** | 50 |
| `libero_goal` | 10 | 相同物体和布局、不同目标 → **任务语义理解** | 50 |
| `libero_10` (= LIBERO-Long) | 10 | 长程多步任务 | 50 |
| `libero_90` | 90 | 预训练用的大池子 | 50 |

**注意命名**：代码里叫 `libero_10`，论文里叫 **LIBERO-Long**，是同一个东西。

各 suite 的任务风格差别很大，看任务名最直观：

```
libero_spatial:  pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate
                 pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate
                 ↑ 十个任务全是"拿黑碗放盘子"，区别只在空间描述词

libero_object:   pick_up_the_alphabet_soup_and_place_it_in_the_basket
                 pick_up_the_ketchup_and_place_it_in_the_basket
                 ↑ 句式完全一样，区别只在物体名

libero_goal:     open_the_middle_drawer_of_the_cabinet
                 put_the_bowl_on_the_stove
                 ↑ 动作类型都不同

libero_10:       KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
                 LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket
                 ↑ 带场景前缀，一句话里两个子任务
```

> 理解这个设计，就知道为什么 OpenVLA 在 Spatial 上 84.7% 而 Long 只有 53.7%——
> 前三个 suite 是单步抓放，Long 要连续完成两个子目标，误差会累积。

## 3. 核心 API

```python
from libero.libero import benchmark

# 1. 拿到 suite
task_suite = benchmark.get_benchmark_dict()["libero_spatial"]()
print(task_suite.n_tasks)          # 10

# 2. 拿到某个任务
task = task_suite.get_task(0)
#   Task 是 NamedTuple: name / language / problem / problem_folder
#                       / bddl_file / init_states_file
print(task.language)               # 送给 VLA 的自然语言指令

# 3. 拿到固定初始状态（评测可复现的关键）
init_states = task_suite.get_task_init_states(0)   # shape: (50, state_dim)

# 4. 建环境
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path
import os

bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0)      # ⚠️ 必须 seed：即使用固定初始状态，seed 仍会影响物体位置

# 5. 跑一个 episode
env.reset()
obs = env.set_init_state(init_states[episode_idx])   # 用第 idx 个初始状态
for t in range(max_steps):
    action = policy(obs)                              # 7 维
    obs, reward, done, info = env.step(action)
    if done:                                          # done == 任务成功
        break
```

### 观测字典的关键键名

```python
obs["agentview_image"]        # 第三人称 RGB，(H, W, 3)  ⚠️ 上下颠倒，要转 180°
obs["robot0_eye_in_hand_image"]  # 腕部 RGB
obs["robot0_eef_pos"]         # 末端执行器位置 (3,)
obs["robot0_eef_quat"]        # 末端执行器四元数 (4,)
obs["robot0_gripper_qpos"]    # 夹爪关节 (2,)
```

### 动作空间：7 维

```
[Δx, Δy, Δz,  Δroll, Δpitch, Δyaw,  gripper]
 └─ 平移增量 ─┘  └──── 旋转增量 ────┘   └ 开合 ┘
```
夹爪：**-1 = 张开，+1 = 闭合**（OpenVLA 的 dataloader 训练时翻转过符号，
执行前要用 `invert_gripper_action` 翻回来）。

## 4. ⚠️ 三个必踩的坑

**坑 1：图像上下颠倒。**
`get_libero_image()` 里有一行 `img = img[::-1, ::-1]`，把图转 180°。OpenVLA 论文
Appendix E 明确说训练和测试都要转。**忘了转，成功率会掉到接近 0**，而且报错不明显。

**坑 2：前 10 步必须空转。**
仿真器 reset 后物体是"掉"到桌面的，需要时间落稳。所以要先用 dummy action
`[0,0,0,0,0,0,-1]` 走 `num_steps_wait=10` 步。不等就开始预测，模型看到的是物体在半空中。

**坑 3：服务器上必须设 `MUJOCO_GL=egl`。**
无显示器时 MuJoCo 默认渲染后端会失败。缺 `libegl1` 的话还要
`apt install -y libegl1 libgl1`。

## 5. 💡 纯评测不需要下载数据集

这一条能给你省 10GB+ 下载和一堆时间。

LIBERO 的演示数据集（human teleoperation demos）要单独下载：
```bash
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

**但如果只是用 OpenVLA 官方的 4 个微调 checkpoint 做评测，这个数据集完全用不到。**

理由（已在仓库里核实）：
- 评测只需要 **初始状态** 和 **环境定义**
- `init_files/` (13 MB) 和 `bddl_files/` (580 KB) **都随仓库自带**，`pip install -e .` 就有了
- 演示数据只在你要**自己训练/微调**时才需要

> 什么时候才要下：想自己 LoRA 微调时。而且那时要下的也不是原版 LIBERO 数据集，
> 是 OpenVLA 重新渲染标注过的 **`openvla/modified_libero_rlds`**（HuggingFace，约 10 GB）。
> 原版数据 OpenVLA 是不能直接用的（见 `docs/01` 第 6 节的五项改造）。

## 6. 评测协议（跟 OpenVLA 对齐）

| 项 | 值 |
|---|---|
| 每 suite trial 数 | 500（10 任务 × 50 episode） |
| 随机种子 | 3 个，取平均 → 每个统计量 1500 次 rollout |
| 初始状态 | 用原版 LIBERO 提供的固定配置，**不改测试环境** |
| 成功判据 | 环境返回 `done == True` |
| 每 suite 独立策略 | 是——4 个 checkpoint 分别对应 4 个 suite，不能混用 |

各 suite 的 max_steps（OpenVLA 设的，与最长训练 demo 对应）：

| Suite | max_steps | 最长 demo |
|---|---|---|
| libero_spatial | 220 | 193 |
| libero_object | 280 | 254 |
| libero_goal | 300 | 270 |
| libero_10 | 520 | 505 |

## 7. 上手建议

先花十分钟跑通这段，比读十页文档管用：

```python
# 不加载任何模型，纯粹熟悉环境
import os; os.environ["MUJOCO_GL"] = "egl"
import numpy as np, imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

ts = benchmark.get_benchmark_dict()["libero_spatial"]()
task = ts.get_task(0)
print("指令:", task.language)

bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(0); env.reset()
obs = env.set_init_state(ts.get_task_init_states(0)[0])

frames = []
for t in range(60):
    # 随机动作，只为看环境怎么响应
    a = np.concatenate([np.random.uniform(-0.3, 0.3, 6), [-1]])
    obs, r, done, info = env.step(a.tolist())
    frames.append(obs["agentview_image"][::-1, ::-1])   # 记得转 180°
imageio.mimsave("libero_demo.mp4", frames, fps=20)
print("观测键:", sorted(obs.keys()))
```

看完这段视频，你就理解了：任务长什么样、20Hz 是多快、随机动作有多离谱、
以及为什么需要前 10 步等待。
