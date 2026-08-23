"""
视觉 token 压缩算子 —— 用于冻结模型下的 token 预算消融。

介入点: prismatic/extern/hf/modeling_prismatic.py:366
    patch_features = self.vision_backbone(pixel_values)     # (B, 256, D)
    projected     = self.projector(patch_features)          # (B, 256, llm_dim)
下游 attention mask 按 shape[1] 动态构建(372-373 行)，故只需压缩
vision_backbone 的输出，整条链路自动适配，无需改动模型代码。

⚠️ 本消融的适用范围: OpenVLA 只吃单帧，没有历史序列，因此这里无法使用
   docs/04 设计的"时间变化门控"选择准则。测的是"单帧本身能压多狠"。
   这是**下界** —— 历史帧相对当前帧的冗余只会更高。
"""

from __future__ import annotations

import torch


# ---------------------------------------------------------------- 选择算子
def keep_random(x: torch.Tensor, k: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """随机保留 k 个 token。最弱的基线。索引排序后取，保持空间顺序。"""
    b, n, _ = x.shape
    idx = torch.stack([torch.randperm(n, generator=gen, device=x.device)[:k] for _ in range(b)])
    idx = idx.sort(dim=1).values
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def keep_uniform(x: torch.Tensor, k: int) -> torch.Tensor:
    """在 16x16 网格上均匀取样。对应"不看内容的规则下采样"。"""
    n = x.shape[1]
    idx = torch.linspace(0, n - 1, k, device=x.device).round().long()
    return x[:, idx, :]


def keep_norm(x: torch.Tensor, k: int) -> torch.Tensor:
    """按特征 L2 范数取 top-k。常用的无监督显著性代理。"""
    score = x.norm(dim=-1)                                   # (B, N)
    idx = score.topk(k, dim=1).indices.sort(dim=1).values     # 排序以保持空间顺序
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def avg_pool_grid(x: torch.Tensor, k: int, grid: int = 16) -> torch.Tensor:
    """
    在 patch 网格上做平均池化。k 需为完全平方数(否则向下取最近的)。
    这是最朴素的做法，也是 ETA-VLA 那类"整块平均"的空间版对照。
    """
    b, n, d = x.shape
    side = max(int(k ** 0.5), 1)
    x2 = x.float().transpose(1, 2).reshape(b, d, grid, grid)
    pooled = torch.nn.functional.adaptive_avg_pool2d(x2, (side, side))
    return pooled.reshape(b, d, side * side).transpose(1, 2)


def tome_merge(x: torch.Tensor, k: int, return_pos: bool = False):
    """
    ToMe 式二部图软匹配合并，迭代直到 token 数降到 k。

    每轮: 按位置奇偶把 token 分成 A/B 两组，为 A 中每个 token 找 B 中最相似者，
    合并相似度最高的 r 对。自相似的背景会被合并，独特的前景被保留。

    选它作主算子的理由(见 docs/04 4.3): 合并后的 token 有明确定义的位置——
    各成分的加权质心——这是后续施加 4D RoPE 的前提。均值/最大池化没有这个性质,
    注意力池化的输出 token 更是没有确定位置。

    两处必须做对的细节:
    1. **按 size 加权**。要合并 5 轮才能压到 12 个 token，每轮都取无权平均的话，
       一个由 32 个原始 patch 合成的 token 和一个单独的 patch 会被等权平均，
       合出来的东西不再代表任何区域。这里维护每个 token 的成分数 size，
       特征和位置都按 size 加权，等价于始终对原始 patch 取平均。
    2. **按位置还原顺序**。ToMe 原版长在 ViT 内部，token 顺序无所谓；这里合并后的
       token 要送进 Llama，Llama 按序列下标施加 RoPE。若按"保留的A + 合并的B"
       拼接输出，空间顺序就乱了，等于给冻结模型喂了训练时没见过的排列。
       故输出前按质心位置重排。

    return_pos=True 时额外返回质心位置 (B, k) —— 4D RoPE 阶段要用。
    """
    b, n, d = x.shape
    device = x.device
    # bf16 的有效位只有 7 bit，余弦相似度的排序会被舍入噪声打乱，内部升到 fp32
    feat = x.float()
    size = torch.ones(b, n, 1, device=device)
    pos = torch.arange(n, device=device, dtype=torch.float32).expand(b, n).unsqueeze(-1).clone()

    while feat.shape[1] > k:
        n_cur = feat.shape[1]
        r = min(n_cur - k, n_cur // 2)          # 每轮最多合并一半

        fa, fb = feat[:, ::2], feat[:, 1::2]
        sa, sb = size[:, ::2], size[:, 1::2]
        pa, pb = pos[:, ::2], pos[:, 1::2]

        sim = torch.nn.functional.normalize(fa, dim=-1) @ \
              torch.nn.functional.normalize(fb, dim=-1).transpose(1, 2)
        best, best_idx = sim.max(dim=-1)                     # A 中每个 token 的最佳匹配
        order = best.argsort(dim=-1, descending=True)
        merge_src, keep_src = order[:, :r], order[:, r:]

        def take(t, idx):
            return torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, t.shape[-1]))

        dst = torch.gather(best_idx, 1, merge_src)           # (B, r) 目标 B 下标
        s_src = take(sa, merge_src)                          # (B, r, 1)
        f_src = take(fa, merge_src) * s_src                  # 加权和，而非均值
        p_src = take(pa, merge_src) * s_src

        di_f = dst.unsqueeze(-1).expand(-1, -1, d)
        di_1 = dst.unsqueeze(-1)
        new_s = sb.scatter_add(1, di_1, s_src)
        new_f = (fb * sb).scatter_add(1, di_f, f_src) / new_s
        new_p = (pb * sb).scatter_add(1, di_1, p_src) / new_s

        feat = torch.cat([take(fa, keep_src), new_f], dim=1)
        size = torch.cat([take(sa, keep_src), new_s], dim=1)
        pos = torch.cat([take(pa, keep_src), new_p], dim=1)

    # 按质心位置还原空间顺序，否则送进 Llama 的是训练时没见过的排列
    order = pos.squeeze(-1).argsort(dim=1)
    feat = torch.gather(feat, 1, order.unsqueeze(-1).expand(-1, -1, d))
    if return_pos:
        return feat, torch.gather(pos.squeeze(-1), 1, order)
    return feat


METHODS = {
    "random": keep_random,
    "uniform": keep_uniform,
    "norm": keep_norm,
    "avgpool": avg_pool_grid,
    "tome": tome_merge,
}


# ---------------------------------------------------------------- 挂载
def patch_vision_backbone(model, keep: int, method: str = "tome") -> None:
    """把压缩算子挂到 model.vision_backbone 的 forward 上（原地修改）。"""
    if method not in METHODS:
        raise ValueError(f"未知方法 {method}，可选: {list(METHODS)}")
    fn = METHODS[method]
    orig = model.vision_backbone.forward

    reported = []

    def wrapped(pixel_values, *a, **kw):
        feats = orig(pixel_values, *a, **kw)
        if keep >= feats.shape[1]:
            return feats
        with torch.no_grad():
            out = fn(feats, keep).to(feats.dtype)
        if not reported:                       # 只在第一次前向时报一次实际 token 数
            reported.append(out.shape[1])
            print(f"[token 消融] {method}: {feats.shape[1]} -> {out.shape[1]} "
                  f"(保留 {out.shape[1] / feats.shape[1] * 100:.1f}%)")
        return out

    model.vision_backbone.forward = wrapped
    print(f"[token 消融] {method}: 目标保留 {keep} 个视觉 token；"
          f"实际数量首次前向时打印(avgpool 只能落在完全平方数上)")


# ---------------------------------------------------------------- 自检
if __name__ == "__main__":
    torch.manual_seed(0)
    X = torch.randn(2, 256, 64)

    print("--- 形状 ---")
    for k in (256, 128, 64, 38, 24, 12):
        cols = []
        for name, fn in METHODS.items():
            cols.append(f"{name}={fn(X, k).shape[1]}")
        print(f"k={k:4d}  " + "  ".join(cols))
    print("(avgpool 只能落在完全平方数上，与 k 不等属预期)")

    print("\n--- tome 合并语义 ---")
    # 造 4 组各 64 个几乎相同的 token: 正确的合并应该恢复出这 4 个簇
    base = torch.randn(1, 4, 32)
    Y = base.repeat_interleave(64, dim=1) + 0.01 * torch.randn(1, 256, 32)
    out, pos = tome_merge(Y, 4, return_pos=True)
    err = (out[0] - base[0]).abs().max().item()
    print(f"压到 4 个 token 后与簇中心的最大偏差: {err:.4f} (应 << 1)")
    print(f"质心位置: {[round(v, 1) for v in pos[0].tolist()]} (理想 31.5/95.5/159.5/223.5)")

    print("\n--- 位置单调性(送进 Llama 前必须成立) ---")
    for k in (128, 64, 24):
        _, p = tome_merge(X, k, return_pos=True)
        mono = bool((p[:, 1:] >= p[:, :-1]).all())
        print(f"k={k:4d}  质心位置单调递增: {mono}")

    print("\n--- size 加权检验 ---")
    # 一半 token 全同、一半互异: 全同的那半应被压成极少数几个，且值接近原值
    Z = torch.cat([torch.zeros(1, 128, 16), torch.randn(1, 128, 16)], dim=1)
    out = tome_merge(Z, 16)
    n_zero = int((out[0].abs().sum(-1) < 1e-3).sum())
    print(f"压到 16: 其中接近零向量的有 {n_zero} 个 (无权平均会把零向量污染掉，应 >= 1)")
