# VLA-4D

大创项目：复现 OpenVLA，并沿 4D 时空方向做改进。

## 目录

```
docs/     论文精读笔记与研究方案
setup/    服务器环境搭建
scripts/  评测启动与轨迹记录
src/      指标计算与可视化
papers/   参考论文 PDF (在 main 分支根目录)
```

## 快速开始

### 1. 环境（在租用的 GPU 服务器上）

```bash
git clone https://github.com/Quinn-0109/vla-4d.git && cd vla-4d
bash setup/setup_server.sh
```

脚本会装好 conda 环境、OpenVLA、LIBERO，并锁定官方复现版本
（Python 3.10.13 / PyTorch 2.2.0 / transformers 4.40.1 / flash-attn 2.5.5），
最后跑自检确认 LIBERO 能启动。

### 2. 先熟悉 LIBERO（不加载模型，几分钟）

```bash
python scripts/explore_libero.py list                 # 列出 5 个 suite 的全部任务
python scripts/explore_libero.py inspect              # 观测字典结构 + 动作空间含义
python scripts/explore_libero.py probe                # 逐维施加动作，看末端怎么响应
python scripts/explore_libero.py render --task_id 0   # 随机动作录一段视频
```

配合 `docs/03-LIBERO使用指南.md` 看。**纯评测不需要下载 LIBERO 数据集**
（初始状态和环境定义都随仓库自带，省 10 GB+），理由见指南第 5 节。

### 3. 冒烟测试（先确认跑得通，别急着看数字）

```bash
conda activate openvla
export MUJOCO_GL=egl
export OPENVLA_ROOT=$HOME/vla-work/openvla
bash scripts/run_eval.sh smoke
```

### 4. 完整评测

```bash
bash scripts/run_eval.sh full        # 4 suite × 50 trials，单种子
bash scripts/run_eval.sh full 3      # 3 个种子，对齐论文协议
```

### 5. 出图表

```bash
python src/analysis/analyze.py --traj_dir results/trajectories
```

## 硬件要求

| 用途 | 显存 | 说明 |
|---|---|---|
| 评测（推理） | **≥16 GB** | 7B bf16 占 15 GB；RTX 4090 约 6 Hz |
| LoRA 微调 | **≥27 GB** | 官方下限；batch_size=16 需 ~72 GB |
| 全量微调 | 8× A100 | 每任务 5–15 小时 |

磁盘建议 **200 GB+**（4 个 checkpoint 各约 16 GB）。

**耗时预估**：按 6 Hz 估算，四个 suite 各 500 rollouts 单种子约 **25 小时**
（LIBERO-Long 最慢，单 episode 最多 520 步）。

## 对标基线

OpenVLA 论文 Appendix E Table 12：

| Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|
| 84.7 ± 0.9 | 88.4 ± 0.8 | 79.2 ± 1.0 | 53.7 ± 1.3 | 76.5 ± 0.6 |

## 我们额外记录的东西

官方评测脚本只输出 MP4 回放和成功/失败日志。`scripts/run_libero_eval_traj.py`
在保持评测逻辑一致的前提下，额外落盘每一步的动作、末端执行器位姿与夹爪状态，
由此可以算出官方没给的指标：

- 归一化 jerk、SPARC（轨迹平滑度）
- 速度方向反转次数、空转步数占比（抖动与停顿）
- 路径效率、动作抖动、夹爪翻转次数
- 完成时间 CT（对齐 VLA-4D 论文的指标）

## 文档

| 文件 | 内容 |
|---|---|
| `docs/01-openvla-精读笔记.md` | 架构、动作离散化、训练配置、Appendix E 复现协议 |
| `docs/02-vla-4d-精读笔记.md` | 4D 视觉/动作表征、两阶段训练、全部消融、复现风险 |
| `docs/03-LIBERO使用指南.md` | 任务套件、API、观测/动作空间、三个必踩的坑 |
| `docs/04-研究思路-4D自适应token池化.md` | 改进方向的设计草案与 novelty 分析 |

## 技术路线

**底座选定 OpenVLA**（而非 Qwen2.5-VL）。理由：保住在 970k 条真实机器人轨迹上
预训练出的动作能力，且官方提供 4 个 LIBERO 微调 checkpoint 可直接对标。
代价是后续若要加 4D RoPE，需要改 Llama-2 的 1D RoPE，改动较深。

当前阶段目标：**读透论文 → 跑通代码 → 学会 LIBERO → 拿到自己的实验数据与可视化。**
