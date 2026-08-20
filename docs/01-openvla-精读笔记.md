# OpenVLA 精读笔记

> Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, arXiv 2406.09246
> 本笔记基于 v2（78 页，含 Appendix E LIBERO 实验）

## 1. 一句话概括

把机器人动作预测**当成"视觉-语言"任务**：输入图像 + 语言指令，输出一串"动作 token"字符串。
底座是 Prismatic-7B VLM，在 Open X-Embodiment 的 97 万条真实机器人轨迹上微调而成。

## 2. 架构（三段式）

```
图像 ──> 视觉编码器 ──> 投影器 ──> LLM ──> 动作 token ──> 反离散化 ──> 7-DoF 动作
         (SigLIP+DINOv2)  (2层MLP)  (Llama 2 7B)
指令 ──────────────────────────────> ↑
```

| 组件 | 具体选型 | 关键点 |
|---|---|---|
| 视觉编码器 | **SigLIP + DINOv2 双编码器，特征沿通道拼接**，约 600M 参数 | DINOv2 补空间几何，SigLIP 补语义对齐。论文明确说这是比单 CLIP/SigLIP 强的原因 |
| 投影器 | 2 层 MLP | 把视觉特征映射到语言 embedding 空间 |
| LLM | Llama 2 7B | 来自 Prismatic-7B VLM |
| 输入分辨率 | **224×224** | 试过 384×384，**性能无差异但训练慢 3 倍**，故选 224 |

**为什么选 Prismatic**：作者对比过 IDEFICS-1、LLaVA。LLaVA 在多物体语言 grounding 上比 IDEFICS-1 高 35% 绝对成功率；Prismatic 又比 LLaVA 再高约 10%，归因于 SigLIP-DINOv2 融合带来的空间推理能力。

## 3. 动作离散化（复现最容易踩坑的地方）

这是 OpenVLA 的核心机制，必须吃透：

1. 7 维动作**每一维独立离散化成 256 个 bin**
2. bin 宽度按训练数据的 **第 1 与第 99 分位数**均匀切分
   - 关键：用分位数而非 min-max。RT-2 用 min-max，会被离群动作把区间撑大、削弱有效精度
3. Llama tokenizer 只预留 100 个 special token，不够 256 个 → **直接覆写词表中最少使用的 256 个 token**（即最后 256 个）
4. 训练目标：标准 next-token prediction，**交叉熵只在动作 token 上计算**

对应代码：`prismatic/vla/action_tokenizer.py`

## 4. 训练配置

| 项 | 值 |
|---|---|
| 数据 | Open X-Embodiment，970k 轨迹；筛选条件=至少一个第三人称相机 + 单臂末端执行器控制；混合权重沿用 Octo |
| 算力 | **64× A100，14 天，共 21,500 A100-hours** |
| Batch size | 2048 |
| 学习率 | **固定 2e-5，无 warmup**（扫过多个数量级） |
| Epochs | **27 轮** |
| 视觉编码器 | **必须一起微调**（冻结会显著掉点，与 VLM 常规做法相反） |

> 注：DROID 数据集以 10% 权重加入过，但 action token accuracy 始终上不去，**最后 1/3 训练把它移除了**。

**几条作者明确写出的经验**：
- VLA 训练需要**远多于 LLM/VLM 的 epoch 数**，真机性能会一直涨到 action token accuracy 超过 95%
- 冻结视觉编码器 = 性能差。假设是预训练视觉骨干抓不住精细空间细节

## 5. 微调与推理（和租服务器直接相关）

**LoRA 微调**
- r=32 为推荐默认值，**rank 对性能影响可忽略**
- 只训 **1.4% 参数**，性能追平全量微调
- 单张 A100 上 **10–15 小时**完成一个任务的微调（比全量微调省 8 倍算力）
- 全量微调需要 8× A100 跑 5–15 小时
- 消融结论：只调最后一层 / 冻结视觉编码器 → 都很差；sandwich 微调（视觉编码器+embedding+最后一层）较好；**LoRA 最优**

**推理开销**
- **bfloat16 下 15GB 显存**，16GB 卡即可serve
- **RTX 4090 上约 6 Hz**（未做 compile / 投机解码等加速）
- 4-bit 量化：显存降一半以上，**性能与 bf16 持平**
- 官方提供 remote inference server，机器人端不需要本地大卡

## 6. Appendix E：LIBERO 实验（我们要复现的部分）

### 数据集改造 —— 五处修改，缺一不可

原始 LIBERO 数据**不能直接用**，作者做了 5 项处理：

1. **重新渲染到 256×256**。原始数据是 128×128，直接上采样画质太差；作者用 demo 里存的动作重新 step 仿真环境并保存渲染图
2. **过滤 no-op 动作** —— 平移和旋转分量接近零且不改变夹爪状态的动作。**这步至关重要**：OpenVLA 这类高表达力单步策略会学会模仿 no-op，导致评测时**在某些状态永久卡死**
3. **第三人称图像训练和测试时都旋转 180°**（作者硬件上 LIBERO 返回的图是倒的）
4. **回放并剔除失败 demo**：移除 Spatial 68/500、Object 46/500、Goal 72/500、**Long 121/500**
5. **只用第三人称相机，不用腕部相机** —— 为了公平对比，因为 OpenVLA 只吃第三人称图像

> 测试环境**未改动**，用的是原始 LIBERO 提供的初始配置。

### 评测协议

- 4 个 suite，每个 10 任务 × 50 条人类遥操作 demo
- 每 suite **500 次 trial，3 个随机种子** → 每个统计量 1500 次 rollout
- 每个 suite **独立训练一个策略**（不是一个策略打四个 suite）

### 结果（Table 12）

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---|---|---|---|---|
| Diffusion Policy (from scratch) | 78.3 ± 1.1 | **92.5 ± 0.7** | 68.3 ± 1.2 | 50.5 ± 1.3 | 72.4 ± 0.7 |
| Octo (fine-tuned) | 78.9 ± 1.0 | 85.7 ± 0.9 | **84.6 ± 0.9** | 51.1 ± 1.3 | 75.1 ± 0.6 |
| **OpenVLA (fine-tuned)** | **84.7 ± 0.9** | 88.4 ± 0.8 | 79.2 ± 1.0 | **53.7 ± 1.3** | **76.5 ± 0.6** |

**作者自己的解读（诚实且重要）**：OpenVLA 在 LIBERO 上的领先幅度**明显小于真机实验**。归因为 OpenVLA 纯用真实数据预训练、没有仿真数据，存在 sim-real domain gap。Octo 同样是真实数据预训练，也只比从零训的 Diffusion Policy 略好。

## 7. 论文承认的局限（正好是 VLA-4D 的切入点）

1. **只支持单帧图像输入** —— 明确说扩展到多图/本体感知/观测历史是重要future work
2. **推理吞吐不足** —— 6Hz 撑不起 ALOHA 那种 50Hz 高频控制；提到 action chunking 是可能解法
3. **可靠性不够** —— 各任务通常 <90% 成功率
4. 在 5.2 节还有一句关键自陈：
   > "For narrower but highly dexterous tasks, **Diffusion Policy still shows smoother and more precise trajectories**; incorporating **action chunking and temporal smoothing** ... may help OpenVLA attain the same level of dexterity."

> **这段是我们报告里最有价值的引用**：OpenVLA 作者自己承认轨迹不够平滑、缺时间维度处理。这就是 4D 路线的直接依据，不需要我们硬编叙事。
