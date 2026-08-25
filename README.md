# VLA-4D

大创项目：复现 OpenVLA，并沿 4D 时空方向做改进。

**当前进度**：OpenVLA 四 suite 复现完成 → 单帧 token 消融完成（拿到核心发现）
→ 正在做多帧微调的硬门槛验证。详见 [进度](#进度) 与 `docs/06-后续计划.md`。

## 核心发现

> **位置编码的影响比 token 压缩本身大五倍。**

把 OpenVLA 的 256 个视觉 token 压到 96 个（2.7×）：

| 做法 | LIBERO-Spatial 成功率 |
|---|---|
| 不压缩（基线，n=200） | 82.5% |
| 压到 96，**不修正位置** | **18.0%** |
| 压到 96，**修正 `position_ids`** | **72.5%** |

同一个压缩率、同一个算子，只改位置编码就差 **54 个百分点**；
而压缩本身的代价是 **−10.6 个百分点**（四 suite 合并，p=1.6e-03）。

原因在 `modeling_prismatic.py:381`：视觉块夹在 BOS 与语言 token 之间，
且 `position_ids=None` 让 Llama 按 `arange` 生成位置。压短视觉块，
**整个语言指令与动作解码位置都被前移**。把每个合并 token 的位置还原成
其成分 patch 的质心后，掉点几乎全部消失。

这直接支撑本项目的论点：**位置编码必须来自真实时空坐标，而非序列下标**——
也就是要做的 4D RoPE。完整实验与统计见 `docs/05-实验记录.md` §7。

## 目录

**新读者从 [`docs/00-索引.md`](docs/00-索引.md) 开始** —— 那里有按角色的阅读路径、
脚本地图（10 个脚本分别什么时候用），以及踩过的坑清单。

```
docs/     论文精读笔记、研究方案、实验记录、后续计划
papers/   参考论文 PDF（附索引说明各自作用）
setup/    服务器环境搭建与依赖锁定
scripts/  评测、消融、显存实测、微调
src/      指标计算、统计检验、可视化、token 压缩算子
```

## 快速开始

### 1. 环境（在租用的 GPU 服务器上）

```bash
git clone https://github.com/Quinn-0109/vla-4d.git && cd vla-4d
bash setup/setup_server.sh          # 自动探测数据盘；也可 DATA_DIR=/root/autodl-tmp 指定
source ~/.bashrc
```

脚本会装 conda 环境 / OpenVLA / LIBERO，按 `setup/constraints.txt` 锁定版本，
**把 conda 与 HuggingFace 缓存都放到数据盘**（系统盘通常只有 30 GB，不重定向必爆），
最后自检 LIBERO 能否启动。

**💡 省钱：先用「无卡模式」装环境和下模型。** 装环境 + 下 checkpoint 要 5–8 小时，
无卡模式下几乎不花钱，装完再切 GPU 模式。

```bash
bash setup/setup_server.sh
bash scripts/download_checkpoints.sh libero_spatial   # 约 15 GB
bash setup/check_gpu.sh                               # 切回 GPU 模式后自检
```

> ⚠️ **新开终端/tmux 窗口会掉回 base 环境**（conda init 的行为），
> 这个坑本项目踩过四次。建议把 `conda activate <环境路径>` 和
> `cd <仓库路径>` 加进 `~/.bashrc`。各启动脚本已加前置检查，
> 环境不对会直接拒绝而不是给出误导性的报错。

### 2. 熟悉 LIBERO（不加载模型，几分钟）

```bash
python scripts/explore_libero.py list      # 列出 5 个 suite 的全部任务
python scripts/explore_libero.py inspect   # 观测字典结构 + 动作空间含义
python scripts/explore_libero.py probe     # 逐维施加动作，看末端怎么响应
```

配合 `docs/03-LIBERO使用指南.md`。**纯评测不需要下载 LIBERO 数据集**
（初始状态与环境定义随仓库自带，省 10 GB+），理由见指南第 5 节。

### 3. 评测

```bash
bash scripts/run_eval.sh smoke              # 冒烟测试，先确认跑得通
bash scripts/run_eval.sh full               # 4 suite × 50 trials
python src/analysis/analyze.py --traj_dir results/trajectories
```

### 4. token 消融

```bash
bash scripts/run_token_ablation.sh cost     # 先看耗时与费用估算
bash scripts/run_token_ablation.sh budget   # 扫 token 预算找拐点
bash scripts/run_token_ablation.sh diag     # 分离「信息损失」与「位置错位」
bash scripts/run_token_ablation.sh fixpos   # 位置修正
python src/analysis/collect_ablation.py --suite all   # 跨 suite 合并检验
```

### 5. 多帧微调（阶段 B）

```bash
python scripts/probe_vram.py --sweep        # 显存实测，定 K 与 batch
bash scripts/prepare_finetune.sh check      # 体检: 环境/磁盘/依赖版本
bash scripts/prepare_finetune.sh data       # 下 RLDS 数据集（每 suite 约 1.9 GB）
python scripts/finetune_single.py --bench_only True   # 先测吞吐再决定步数
python scripts/merge_lora.py --adapter runs/<exp>/adapter/step10000 --out <出口>
```

## 硬件要求

### 架构门槛：Ampere (sm80) 及以上

官方 `openvla_utils.py:45` 默认 `attn_implementation="flash_attention_2"`，
配合 `torch_dtype=torch.bfloat16`。**若 flash-attn 装不上，
`setup_server.sh` 的 `SKIP_FLASH_ATTN=1` 会把它改成 sdpa**——
二者是同一注意力运算的不同 kernel，数值近乎一致，成功率不受影响
（论证见 `docs/05` §1 偏离①）。但 bf16 仍要求 Ampere 及以上。

| | GPU | 算力 |
|---|---|---|
| ❌ 跑不了 | V100 / T4 / RTX 2080Ti | 7.0 / 7.5 |
| ✅ 可用 | A100(8.0)、A10·A5000·A6000·RTX 3090(8.6)、RTX 4090·L20·L40S(8.9)、H100(9.0) | ≥8.0 |

### 显存（RTX 4090 24 GB 实测，非估算）

权重本身占 **15.13 GB**，余下约 8.6 GB 全部留给激活。
完整数据 `results/tables/vram_probe.json`，分析见 `docs/06` 阶段 B。

| 用途 | 峰值 | 备注 |
|---|---|---|
| 评测（bf16） | 15.4 GB | RTX 4090 约 6 Hz |
| 单帧 LoRA 微调 | **16.8 GB** | b=4 + 累积 4 + 梯度检查点 + 冻结视觉主干 |
| K=8 多帧微调（不压缩） | 21.2 GB | 必须开梯度检查点，否则 K 最多到 2 |
| K=8 多帧微调（压到 96/帧） | 20.1 GB | 与不压缩仅差 1.1 GB |
| K=16 | OOM | 需要 40 GB 以上的卡 |

**两个反直觉的实测结论**：

1. **梯度检查点才是主要杠杆，不是压缩。** K=8 时压缩与否只差 1.1 GB——
   检查点已消掉大半 LLM 激活。压缩换来的是速度，不是显存。
2. **token 更少可能更费显存。** K=2×256（512 token）要 20.3 GB，
   而 K=4×96（384 token）要 20.4 GB。因为压缩发生在视觉主干**之后**，
   K 帧各过一次 DINOv2+SigLIP 的激活与压缩无关。
   → 据此决定**多帧微调时冻结视觉主干**（不挂 LoRA）。

### 磁盘

| 项目 | 大小 |
|---|---|
| conda + PyTorch + CUDA 库 | ~18 GB |
| 单个 checkpoint (7B bf16) | ~15 GB |
| LIBERO RLDS 数据集 | **每 suite 约 1.9 GB**，四个 ~8 GB |
| LoRA adapter | 几百 MB（合并后才是 15 GB） |

⚠️ HuggingFace 默认缓存在 `~/.cache`（系统盘）。setup 脚本会重定向 `HF_HOME`
到数据盘；手动装务必自己设，否则 30 GB 系统盘必爆。

> **LIBERO 是同步仿真器**：`env.step()` 会等动作算完才推进，所以推理快慢
> **完全不影响成功率**，只影响评测要跑多久。选卡是花钱买时间，不是买正确性。

## 进度

### ✅ 阶段 0：OpenVLA 复现（四 suite 各 n=500）

| Suite | 本次复现 | 论文 | 差异 | p |
|---|---|---|---|---|
| Spatial | 84.2% | 84.7 ± 0.9 | −0.5 | 0.788 ✅ |
| **Object** | **84.0%** | 88.4 ± 0.8 | **−4.4** | **0.016** ❌ |
| Goal | 76.4% | 79.2 ± 1.0 | −2.8 | 0.192 ✅ |
| Long | 52.0% | 53.7 ± 1.3 | −1.7 | 0.511 ✅ |
| **平均** | **74.1%** | 76.5 ± 0.6 | −2.35 | 0.034 |

偏差几乎完全由 Object 一个 suite 驱动（若 Object 达论文值，平均差异降至
−1.25，p=0.260 不再显著）——**不存在系统性复现偏差**。

**另一项发现：LIBERO 评测是完全确定性的。** 换种子复跑逐位相同
（`env.seed(0)` 硬编码 + 初始状态读自文件 + 贪心解码）。
推论：论文的 ±0.9% 只可能来自 3 次独立微调，而非评测种子；
仅拿到一份 checkpoint 的复现者**在原理上无法触及训练种子方差**。见 `docs/05` §3.2。

### ✅ 阶段 0.5：轨迹质量分析（论文未提供的数据）

| Suite | 成功率 | 归一化 Jerk | 相对 Spatial |
|---|---|---|---|
| Spatial | 84.2% | 4 026 | 1.0× |
| Object | 84.0% | 7 297 | 1.8× |
| Goal | 76.4% | 15 758 | 3.9× |
| **Long** | **52.0%** | **75 374** | **18.7×** |

成功率与 jerk 的排序完全反向单调（6/6 对），且该预测在跑 Long 之前已登记。
另有一项**指标稳健性**发现：路径效率在四个 suite 间两次变号，
不可用于跨任务类型比较；只有 jerk、平均速度、空转占比是安全的。见 `docs/05` §4。

### ✅ 阶段 1：单帧 token 消融

见上文[核心发现](#核心发现)。附带产出：

- **冗余度量**：特征空间 top-10% 的 patch 承载 70–76% 的变化（`docs/04` §4.4）
- **信息底线**：4× 信息压缩几乎无损，8× 则崩溃——边界在 64 与 32 token 之间
- **一次撤回**：初次以 n=50 报告「2.7× 压缩免费（p=0.82）」，
  加量到 n=200 后被推翻。教训与过程记在 `docs/05` §7.5

### 🔄 阶段 2：多帧输入（进行中）

硬门槛：K=1 的 LoRA 微调能否复现到 ~84%。跑不到则说明训练配置有问题，
后续所有多帧对照都失去地基。配置与依据见 `docs/06` 阶段 B。

### ⏳ 阶段 3–6

> ⚠️ 阶段 3 开跑前有**六项前置工作**必须完成（`docs/06` C.0），
> 其中两项足以让主实验得出假阴性：**帧间隔从未定义**（K=8 取连续帧只有
> 0.4 秒"历史"）与**训练种子方差未估计**（每配置只训一次，而我们自己证明过
> 论文的误差棒只能来自训练种子）。另有一项推翻了原有论证：
> OpenVLA 在 LIBERO 上只用静态第三人称相机，而"4D 必要性"的论证依赖相机运动——
> 已决定加入腕部相机 + 离线缓存单目深度补上。

**前置控制（历史帧到底有没有用）** → 2D 池化 vs 4D 池化 → 4D RoPE → 消融与写作。

⚠️ 前置控制是后加的，起因是一个问题：*冻结视觉主干、只合并多帧特征送进 LLM，
视觉特征本身没变好，历史帧凭什么弥补单帧的判别缺陷？*

答案是**它不能**——历史帧补的是信息的可得性（只能由变化观测的状态、
遮挡物体的记忆、多步任务的阶段消歧），不是特征的判别力。但这暴露出
**我们从未验证过历史帧在 LIBERO 上是否有用**：若 K=8 与 K=1 打平，
后面比较两种池化方式就毫无意义。故补一组 K=1 vs K=8 的对照，
主战场放在 LIBERO-Long（多步、有阶段歧义，且我们自己的数据显示
其 jerk 是 Spatial 的 18.7 倍）。设计与三种结果的走向见 `docs/06` 阶段 C。

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
| `docs/00-索引.md` | **从这里开始**：阅读路径、脚本地图、踩坑清单 |
| `docs/01-openvla-精读笔记.md` | 架构、动作离散化、训练配置、Appendix E 复现协议 |
| `docs/02-vla-4d-精读笔记.md` | 4D 视觉/动作表征、两阶段训练、全部消融、复现风险 |
| `docs/03-LIBERO使用指南.md` | 任务套件、API、观测/动作空间、三个必踩的坑 |
| `docs/04-研究思路-4D自适应token池化.md` | 设计草案、novelty 核查、冗余度量、池化算子选型、4D RoPE 设计 |
| `docs/05-实验记录.md` | **实验记录**：复现、统计检验、轨迹质量、token 消融、工程发现 |
| `docs/06-后续计划.md` | **路线图**：显存实测结论、阶段 B–E 的配置与风险 |

## 技术路线

**底座选定 OpenVLA**（而非 Qwen2.5-VL）。理由：保住在 970k 条真实机器人轨迹上
预训练出的动作能力，且官方提供 4 个 LIBERO 微调 checkpoint 可直接对标。
代价是加 4D RoPE 要改 Llama-2 的 1D RoPE，改动较深——
不过阶段 1 的结果表明，**这个「代价」恰恰是价值所在**。
