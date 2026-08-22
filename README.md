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
bash setup/setup_server.sh          # 自动探测数据盘；也可 DATA_DIR=/root/autodl-tmp 指定
source ~/.bashrc
```

脚本会：装 conda 环境 / OpenVLA / LIBERO，锁定官方复现版本，**把 conda 和
HuggingFace 缓存都放到数据盘**（系统盘通常只有 30 GB，不重定向必爆），
最后自检 LIBERO 能否启动。

**💡 省钱：先用「无卡模式」装环境和下模型。** 脚本检测不到 GPU 时不会退出，
照常安装。装环境 + 下 checkpoint 要 5–8 小时，在无卡模式下做几乎不花钱，
装完再切 GPU 模式跑评测。

```bash
# 无卡模式下：
bash setup/setup_server.sh
bash scripts/download_checkpoints.sh libero_spatial   # 约 15 GB

# 切回 GPU 模式后先自检：
bash setup/check_gpu.sh
```

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

### ⚠️ 架构门槛：必须 Ampere (sm80) 及以上

`experiments/robot/openvla_utils.py:45` **写死了** `attn_implementation="flash_attention_2"`，
配合 `torch_dtype=torch.bfloat16` —— 两者都要求 Ampere 及以上架构。**这不是可选项。**

| | GPU | 算力 |
|---|---|---|
| ❌ 跑不了 | V100 / T4 / RTX 2080Ti | 7.0 / 7.5 |
| ✅ 可用 | A100(8.0)、A10·A5000·A6000·RTX 3090(8.6)、RTX 4090·L20·L40S(8.9)、H100(9.0) | ≥8.0 |

### 显存

| 用途 | 显存 | 说明 |
|---|---|---|
| 评测（bf16） | **≥16 GB** | 模型占 15 GB；RTX 4090 约 6 Hz |
| 评测（4bit 量化） | **≥8 GB** | 论文实测 4bit 精度与 bf16 **持平**；⚠️ 别用 8bit，更慢且更差 |
| LoRA 微调 | **≥27 GB** | 官方下限；batch_size=16 需 ~72 GB |
| 全量微调 | 8× A100 | 每任务 5–15 小时 |

### 磁盘

| 项目 | 大小 |
|---|---|
| conda + PyTorch + CUDA 库 | ~18 GB |
| 单个 checkpoint (7B bf16) | **~15 GB** |
| 4 个 checkpoint 全下 | **~60 GB** |
| LIBERO + assets | ~1 GB |

**只跑一个 suite 约需 35 GB，四个全跑约 80 GB。** 数据盘 50 GB 也能做——
一次只下一个，跑完用 `download_checkpoints.sh --rm <suite>` 删掉再下下一个。

⚠️ HuggingFace 默认缓存在 `~/.cache`（系统盘）。setup 脚本会自动把 `HF_HOME`
重定向到数据盘；手动装的话务必自己设，否则 30 GB 系统盘必然爆。

> **LIBERO 是同步仿真器**：`env.step()` 会等动作算完才推进，所以推理快慢
> **完全不影响成功率**，只影响评测要跑多久。选卡是花钱买时间，不是买正确性。

**耗时预估**：按 6 Hz 估算，四个 suite 各 500 rollouts 单种子约 **25 小时**
（LIBERO-Long 最慢，单 episode 最多 520 步）。

## 复现进度

四个 suite 各 n=500，全部完成。

| Suite | 本次复现 | 论文 | 差异 | p |
|---|---|---|---|---|
| Spatial | 84.2% | 84.7 ± 0.9 | −0.5 | 0.788 ✅ |
| **Object** | **84.0%** | 88.4 ± 0.8 | **−4.4** | **0.016** ❌ |
| Goal | 76.4% | 79.2 ± 1.0 | −2.8 | 0.192 ✅ |
| Long | 52.0% | 53.7 ± 1.3 | −1.7 | 0.511 ✅ |
| **平均** | **74.1%** | 76.5 ± 0.6 | −2.35 | 0.034 |

三个 suite 复现良好；平均值的偏差几乎完全由 Object 一个 suite 驱动
（若 Object 达论文值，平均差异降至 −1.25，p=0.260 不再显著）。

**轨迹质量的跨 suite 趋势**（本项目独有数据，论文未提供）：

| Suite | 成功率 | 归一化 Jerk | 相对 Spatial |
|---|---|---|---|
| Spatial | 84.2% | 4 026 | 1.0× |
| Object | 84.0% | 7 297 | 1.8× |
| Goal | 76.4% | 15 758 | 3.9× |
| **Long** | **52.0%** | **75 374** | **18.7×** |

四个 suite 的成功率与 jerk 排序完全反向单调。详见 `docs/05-实验记录.md`。

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
| `docs/05-实验记录.md` | **实验记录**：复现结果、统计检验、轨迹质量分析、工程发现 |

## 技术路线

**底座选定 OpenVLA**（而非 Qwen2.5-VL）。理由：保住在 970k 条真实机器人轨迹上
预训练出的动作能力，且官方提供 4 个 LIBERO 微调 checkpoint 可直接对标。
代价是后续若要加 4D RoPE，需要改 Llama-2 的 1D RoPE，改动较深。

当前阶段目标：**读透论文 → 跑通代码 → 学会 LIBERO → 拿到自己的实验数据与可视化。**
