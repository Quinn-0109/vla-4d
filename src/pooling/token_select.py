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


def tome_merge(x: torch.Tensor, k: int, return_pos: bool = False,
               return_assign: bool = False):
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
    return_assign=True 时额外返回 (B, N) 的归属表，第 p 个原始 patch 落到哪个
    合并后 token —— 供 expand_to_full 做"只降信息、不降长度"的对照实验。
    """
    b, n, d = x.shape
    device = x.device
    # bf16 的有效位只有 7 bit，余弦相似度的排序会被舍入噪声打乱，内部升到 fp32
    feat = x.float()
    size = torch.ones(b, n, 1, device=device)
    pos = torch.arange(n, device=device, dtype=torch.float32).expand(b, n).unsqueeze(-1).clone()
    # 原始 patch -> 当前 token 的归属，每轮跟着重新编号
    assign = torch.arange(n, device=device).expand(b, n).clone() if return_assign else None

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

        if assign is not None:
            # 新顺序是 [保留的A(按 keep_src 次序), 全部B(按原次序)]，据此改写归属
            n_keep = keep_src.shape[1]
            nb = fb.shape[1]
            newidx = torch.empty(b, n_cur, dtype=torch.long, device=device)
            newidx[:, 1::2] = n_keep + torch.arange(nb, device=device)
            ar = torch.arange(n_keep, device=device).expand(b, -1)
            newidx.scatter_(1, keep_src * 2, ar)             # a[j] 位于 cur 的 2j
            newidx.scatter_(1, merge_src * 2, n_keep + dst)
            assign = torch.gather(newidx, 1, assign)

    # 按质心位置还原空间顺序，否则送进 Llama 的是训练时没见过的排列
    order = pos.squeeze(-1).argsort(dim=1)
    feat = torch.gather(feat, 1, order.unsqueeze(-1).expand(-1, -1, d))
    out = [feat]
    if return_pos:
        out.append(torch.gather(pos.squeeze(-1), 1, order))
    if return_assign:
        inv = torch.empty_like(order)                        # order 的逆置换
        inv.scatter_(1, order, torch.arange(order.shape[1], device=device).expand(b, -1))
        out.append(torch.gather(inv, 1, assign))
    return out[0] if len(out) == 1 else tuple(out)


def expand_to_full(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    **对照实验用**: 只降信息量，不降序列长度。

    先用 tome 把 256 个 token 合成 k 个，再把每个合并 token 的值广播回它
    所有成分 patch 的原始位置 —— 输出仍是 (B, 256, D)，但里面只有 k 个不同的值。

    为什么需要它: 压缩同时改变了两件事——信息量和序列位置。OpenVLA 训练时
    视觉部分恒为 256 个 token，Llama 按序列下标施加 RoPE；压到 128 之后，
    原本第 200 号 patch 坐到了第 100 号位置，这是分布外的。
    掉的点里有多少是信息没了、有多少只是位置错位，直接压是分不开的。

    这个算子把位置这一路固定住:
      - 若成功率基本不掉 -> 信息确实冗余，问题出在长度/位置 -> 方案成立，
        但必须让压缩层参与训练，且位置编码要能在压缩后保持意义(即 4D RoPE)
      - 若同样崩掉       -> 模型真的需要这些细节，免训练压缩这条路走不通
    """
    merged, assign = tome_merge(x, k, return_assign=True)
    d = x.shape[-1]
    return torch.gather(merged, 1, assign.unsqueeze(-1).expand(-1, -1, d))


def shuffle_all(x: torch.Tensor, k: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """
    **诊断用**: 保留全部 256 个 token，只把顺序打乱。k 参数被忽略。

    测的是模型对绝对位置有多敏感——这是"压缩掉点是否主要来自位置错位"
    这一问的上界参照。若打乱顺序本身就让成功率归零，说明位置极其关键，
    压缩引起的位置偏移足以解释大部分掉点。
    """
    b, n, d = x.shape
    idx = torch.stack([torch.randperm(n, generator=gen, device=x.device) for _ in range(b)])
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d))


METHODS = {
    "random": keep_random,
    "uniform": keep_uniform,
    "norm": keep_norm,
    "avgpool": avg_pool_grid,
    "tome": tome_merge,
    # 以下两个是对照/诊断算子，不是候选方案
    "expand": expand_to_full,
    "shuffle": shuffle_all,
}

# 这两个算子输出仍是 256 个 token，因此不能被"keep >= N 就跳过"的短路挡掉
FULL_LENGTH = {"expand", "shuffle"}


# ---------------------------------------------------------------- 位置修正
# 关键发现(docs/05 7.2): OpenVLA 把视觉块夹在 BOS 和语言 token 之间
#     multimodal = cat([emb[:, :1], visual(256), emb[:, 1:]])   # :381
# 且 language_model(..., position_ids=None)，Llama 按 arange(seq_len) 生成位置。
# 压到 k 个 token 后，语言指令和动作解码位置整体前移了 (256-k) 位。
# 实测这一位移解释了几乎全部掉点: k=64 时保持长度(expand) 只掉 6 个百分点(n.s.)，
# 真的缩短序列则掉到 6%。
#
# 下面把位置显式补回去: 每个合并 token 拿回自己质心的原始位置，语言块仍从 257 开始。
# 注意 transformers 4.40 里 position_ids 驱动 RoPE、cache_position 驱动 causal mask，
# 两者可以分开——我们只改前者，注意力的可见范围仍按真实序列走，是对的。

N_ORIG = 256          # OpenVLA 的视觉 token 数
LANG_START = N_ORIG + 1   # BOS 占 0，视觉占 1..256，语言从 257 开始


def centroids_to_positions(pos: torch.Tensor) -> torch.Tensor:
    """
    质心(小数) -> 整数 position_id。+1 给 BOS 让位。

    取整会让相邻质心撞到同一个位置(比如 12.4 和 12.6)，那样两个 token 拿到
    完全相同的 RoPE 旋转，模型再也分不开它们——训练时没有这种情形。
    pos 已按质心升序，用 cummax 技巧强制严格递增，代价是个别 token 的位置
    偏移 1，远小于共享位置的损害。
    """
    p = pos.round().long() + 1
    idx = torch.arange(p.shape[1], device=p.device)
    p = torch.cummax(p - idx, dim=1).values + idx      # 强制严格递增
    return p.clamp(1, N_ORIG)


def _patch_language_positions(model, state: dict) -> None:
    """包住 language_model.forward，在 prefill 和逐步解码时都喂正确的 position_ids。"""
    lm = model.language_model
    orig = lm.forward

    def wrapped(*a, **kw):
        pos = state.get("visual_pos")
        if pos is not None and kw.get("inputs_embeds") is not None and kw.get("position_ids") is None:
            # prefill: [BOS] + [k 个视觉] + [L 个语言]
            emb = kw["inputs_embeds"]
            k = pos.shape[1]
            n_lang = emb.shape[1] - 1 - k
            dev = emb.device
            pid = torch.cat([
                torch.zeros(1, 1, dtype=torch.long, device=dev),
                pos.to(dev),
                torch.arange(LANG_START, LANG_START + n_lang, device=dev).unsqueeze(0),
            ], dim=1)
            kw["position_ids"] = pid
            # 解码步要接着语言块往下走，而不是接着(更短的)缓存长度
            state["next_pos"] = LANG_START + n_lang
        elif (state.get("next_pos") is not None
              and kw.get("past_key_values") is not None
              and kw.get("position_ids") is None
              and kw.get("input_ids") is not None):
            dev = kw["input_ids"].device
            kw["position_ids"] = torch.tensor([[state["next_pos"]]], device=dev)
            state["next_pos"] += 1
        return orig(*a, **kw)

    lm.forward = wrapped


# ---------------------------------------------------------------- 挂载
def patch_vision_backbone(model, keep: int, method: str = "tome",
                          fix_positions: bool = False) -> None:
    """
    把压缩算子挂到 model.vision_backbone 的 forward 上（原地修改）。

    fix_positions=True 时额外修正 position_ids，让压缩后的 token 保持原始
    绝对位置、语言块不被前移（只支持 tome，因为只有它给得出质心位置）。
    """
    if method not in METHODS:
        raise ValueError(f"未知方法 {method}，可选: {list(METHODS)}")
    if fix_positions and method != "tome":
        raise ValueError("fix_positions 只支持 tome —— 其余算子给不出合并后 token 的位置")
    fn = METHODS[method]
    orig = model.vision_backbone.forward

    state: dict = {}
    reported = []

    def wrapped(pixel_values, *a, **kw):
        feats = orig(pixel_values, *a, **kw)
        if keep >= feats.shape[1] and method not in FULL_LENGTH:
            return feats
        with torch.no_grad():
            if fix_positions:
                out, pos = tome_merge(feats, keep, return_pos=True)
                state["visual_pos"] = centroids_to_positions(pos)
                out = out.to(feats.dtype)
            else:
                out = fn(feats, keep).to(feats.dtype)
        if not reported:                       # 只在第一次前向时报一次实际 token 数
            reported.append(out.shape[1])
            print(f"[token 消融] {method}: {feats.shape[1]} -> {out.shape[1]} "
                  f"(保留 {out.shape[1] / feats.shape[1] * 100:.1f}%)"
                  + ("，已修正 position_ids" if fix_positions else ""))
        return out

    model.vision_backbone.forward = wrapped
    if fix_positions:
        _patch_language_positions(model, state)
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

    print("\n--- expand 对照算子 ---")
    out = expand_to_full(X, 64)
    uniq = len(torch.unique(out[0], dim=0))
    print(f"输出形状 {tuple(out.shape)} (应为 (2, 256, 64))，不同值的个数 {uniq} (应为 64)")
    # 4 簇数据: 广播回去后每个位置都应等于自己那一簇的中心
    base = torch.randn(1, 4, 32)
    Y = base.repeat_interleave(64, dim=1) + 0.01 * torch.randn(1, 256, 32)
    e = expand_to_full(Y, 4)
    ref = base.repeat_interleave(64, dim=1)
    print(f"4 簇数据广播回 256 位置后与簇中心的最大偏差: {(e - ref).abs().max():.4f} (应 << 1)")

    print("\n--- 质心 -> position_id ---")
    _, pc = tome_merge(X, 64, return_pos=True)
    pid = centroids_to_positions(pc)
    strict = bool((pid[:, 1:] > pid[:, :-1]).all())
    print(f"严格递增(无位置碰撞): {strict}；范围 [{int(pid.min())}, {int(pid.max())}] "
          f"(应落在 [1, 256] 内)")
    collide = int((pc.round().long()[:, 1:] == pc.round().long()[:, :-1]).sum())
    print(f"若不做去碰撞处理，会有 {collide} 处相邻 token 共享同一位置")

    print("\n--- size 加权检验 ---")
    # 一半 token 全同、一半互异: 全同的那半应被压成极少数几个，且值接近原值
    Z = torch.cat([torch.zeros(1, 128, 16), torch.randn(1, 128, 16)], dim=1)
    out = tome_merge(Z, 16)
    n_zero = int((out[0].abs().sum(-1) < 1e-3).sum())
    print(f"压到 16: 其中接近零向量的有 {n_zero} 个 (无权平均会把零向量污染掉，应 >= 1)")
