"""
4D RoPE —— **扩展**而非替换 Llama-2 的 1D RoPE（docs/06 §2.3）。

直觉做法是照 Qwen2-VL 的 M-RoPE 把 head_dim 全拆成 4 组分别编码 (t,x,y,z)。
**不采用**：那样文本 token 的位置语义也被改了，对一个只做 LoRA 微调的
预训练模型风险过高，而且要回答"文本坐标与视觉度量坐标怎么共存"这个麻烦问题。

采用的方案 —— 按 rotary pair 划分通道：

    ~70% pairs   全部 token   原样 1D RoPE，完全不动
    ~30% pairs   仅视觉 token  4 轴 (t,x,y,z)
                 文本 token   **单位旋转（角度恒为 0）**

三个好处：① 预训练通路原封不动；② 消融把 4D 通道置恒等即天然退化成基线，
③ 绕开文本/视觉坐标共存的问题。

**与 coord_pool 的关系就是 §0 那个框**：`coord_bin_pool` 返回的 `coord`
是箱内坐标均值（质心），这里直接拿它当 RoPE 的输入，中间没有任何坐标系转换。
G3/G4/M2 的差别只在传进来的 `coord` 是 (t,h,w) 还是 (t,x,y,z)。
"""

from __future__ import annotations

import torch

# Llama-2 7B: hidden 4096 / 32 heads = 128
HEAD_DIM = 128
ROPE_BASE = 10000.0

# 4D 通道占比，以及四轴之间怎么分（docs/04 §5.4 坑 3）。
# t 的取值范围只有 K 个离散时刻，远小于 x,y,z 的连续范围，所以拿小头。
RESERVE_FRAC = 0.30
AXIS_SPLIT = (1, 3, 3, 3)          # t : x : y : z


def _inv_freq(n_pairs: int, base: float = ROPE_BASE, offset: int = 0,
              total: int | None = None, device=None) -> torch.Tensor:
    """
    标准 RoPE 频率：`1 / base^(2i/d)`。`i` 越小频率越高。

    `offset`/`total` 让我们能取出**原始频率谱的某一段**——保留下来的 1D 通道
    必须继续用它在预训练里对应的那个频率，不能重新编号，否则等于换了一套位置编码。
    """
    total = total or n_pairs
    i = torch.arange(offset, offset + n_pairs, device=device, dtype=torch.float32)
    return 1.0 / (base ** (2.0 * i / (2.0 * total)))


def channel_plan(head_dim: int = HEAD_DIM, n_axes: int = 4,
                 reserve_frac: float = RESERVE_FRAC,
                 axis_split: tuple[int, ...] = AXIS_SPLIT) -> dict:
    """
    决定哪些 rotary pair 归多维坐标、每个轴各拿几个。

    **从高频端拨**（pair 下标小的一端）：高频通道在 2048 长的序列上早已绕过多圈，
    携带的可用长程信号最少，拿走它们对预训练通路的扰动最小。
    ⚠️ 这条是推理不是实测，`docs/06` §2.3 细节 1 列为待消融项——
    低频端、两端各取一半都要试。

    ⚠️ **`n_axes` 存在的唯一理由是公平性。** G3 的坐标是 (t,h,w) 三轴，
    G4 是 (t,x,y,z) 四轴。若让 G3 空着 z 那批通道不转，**G4 就比 G3 多转一批
    通道**，"G4 赢是因为它转的通道多"立刻成为无法排除的替代解释。
    所以 3 轴时把 z 的份额按比例匀回 t/h/w：**两臂参与旋转的 pair 总数严格相等**，
    只是分给了不同数量的轴。

    余数按**最大余数法**分配——早先图省事全丢给第二个轴，19 个 pair 分成
    `t:1 x:12 y:3 z:3`，x 拿了一大半。这种错误不会报错，只会让 x 轴的
    位置分辨率远高于 y/z，而那是没有任何依据的各向异性。

    返回 pair 下标：`d4[a]` 是第 a 个轴占的 pair 列表，`d1` 是保留给 1D 的。
    """
    n_pairs = head_dim // 2
    n_4d = int(round(n_pairs * reserve_frac))
    split = list(axis_split[:n_axes])
    tot = sum(split)
    exact = [n_4d * s / tot for s in split]
    counts = [int(e) for e in exact]
    rem = n_4d - sum(counts)
    order = sorted(range(n_axes), key=lambda i: (-(exact[i] - counts[i]), i))
    for i in order[:rem]:
        counts[i] += 1
    d4, cur = [], 0
    for c in counts:
        d4.append(list(range(cur, cur + c)))
        cur += c
    return {"n_pairs": n_pairs, "n_4d": n_4d, "d4": d4,
            "d1": list(range(cur, n_pairs))}


def build_rope(
    pos1d: torch.Tensor,
    coord: torch.Tensor,
    is_visual: torch.Tensor,
    head_dim: int = HEAD_DIM,
    plan: dict | None = None,
    base: float = ROPE_BASE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    造出可直接喂给 Llama attention 的 `cos` / `sin`，形状 (B, T, head_dim)。

    pos1d      (B, T)      每个 token 的 1D 序列位置。视觉 token 用**成分 patch 的
                           平均原始下标**（`seq_centroid`）——这正是阶段 1 那个
                           值 +54 个点的位置修正的推广（`docs/05` §7.4）。
    coord      (B, T, C)   C=4。视觉 token 的 4D 坐标（已归一化，见 `normalize`）；
                           文本行的取值被忽略。
    is_visual  (B, T)      bool。False 的行在 4D 通道上取**单位旋转**。

    输出按 `rotate_half` 约定排布（前半 / 后半各一份），与 transformers 的
    `apply_rotary_pos_emb` 一致，因此接入时不必改 attention 的数学。
    """
    plan = plan or channel_plan(head_dim, coord.shape[-1])
    if len(plan["d4"]) != coord.shape[-1]:
        raise ValueError(
            f"plan 有 {len(plan['d4'])} 个轴，coord 有 {coord.shape[-1]} 个分量。"
            f"G3(3轴) 与 G4(4轴) 必须各用 channel_plan(n_axes=...) 生成自己的 plan——"
            f"两者参与旋转的 pair 总数相等，这是公平性要求，见 channel_plan。")
    b, t = pos1d.shape
    device = pos1d.device
    n_pairs = plan["n_pairs"]
    ang = torch.zeros(b, t, n_pairs, device=device, dtype=torch.float32)

    # 1D 通道：全部 token 一视同仁，频率沿用它在原始谱里的那一段
    d1 = torch.tensor(plan["d1"], device=device, dtype=torch.long)
    if d1.numel():
        f = _inv_freq(d1.numel(), base, offset=int(d1[0]), total=n_pairs, device=device)
        ang[:, :, d1] = pos1d.unsqueeze(-1).float() * f

    # 4D 通道：只有视觉 token 转，文本 token 角度留 0 = 单位旋转
    vis = is_visual.unsqueeze(-1).float()
    for a, pairs in enumerate(plan["d4"]):
        if not pairs:
            continue
        idx = torch.tensor(pairs, device=device, dtype=torch.long)
        f = _inv_freq(idx.numel(), base, offset=int(idx[0]), total=n_pairs, device=device)
        ang[:, :, idx] = coord[..., a].unsqueeze(-1).float() * f * vis

    ang = torch.cat([ang, ang], dim=-1)              # rotate_half 约定
    return ang.cos(), ang.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """(B, T, head_dim) 上施加旋转。与 transformers 的实现同式。"""
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------- 坐标准备
def normalize(coord: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
              k: int, grid: int = 16) -> torch.Tensor:
    """
    把池化输出的质心映到 RoPE 吃得下的范围：**空间轴 → [0, grid)，t → [0, k)**。

    为什么是 `grid`（=16）而不是别的（`docs/04` §5.4 坑 2）：米制坐标直接塞进 RoPE
    会频率混叠——RoPE 本为整数位置设计，而 LIBERO 桌面工作空间约 1 m。
    映到 [0,16) 是让空间频率量级**对齐模型预训练时见过的 16×16 网格**。
    这个尺度是超参，**G4 若打平，第一个排查它**。

    ⚠️ G3 传进来的 (t,h,w) 本来就在 [0,16)，这一步对它是恒等的——
    两臂共用同一个函数，尺度上没有谁被偏袒。
    """
    span = (hi - lo).clamp(min=1e-8).to(coord.device)
    q = (coord - lo.to(coord.device)) / span                       # → [0,1)
    scale = torch.full((coord.shape[-1],), float(grid), device=coord.device)
    scale[0] = float(k)                                            # t 轴单独给 [0,k)
    return q * scale


def seq_centroid(assign: torch.Tensor, n_slots: int) -> torch.Tensor:
    """
    每个输出槽的 **1D 序列位置 = 其成分 patch 原始下标的均值**。

    这是阶段 1 那个发现的直接推广（`docs/05` §7.4）：压缩后若让 token 按新下标
    重新编号，语言块整体前移，**掉 54 个点**；把每个合并 token 的位置还原成
    成分质心，掉点几乎全消。多帧下同理，只是下标域从 256 变成 K×256。
    """
    b, t = assign.shape
    out = assign.new_zeros(b, n_slots, dtype=torch.float32)
    for i in range(b):
        keep = assign[i] >= 0
        idx = assign[i][keep]
        src = torch.arange(t, device=assign.device, dtype=torch.float32)[keep]
        cnt = torch.zeros(n_slots, device=assign.device).index_add_(
            0, idx, torch.ones_like(src))
        out[i] = torch.zeros(n_slots, device=assign.device).index_add_(
            0, idx, src) / cnt.clamp(min=1)
    return out


def assemble(
    vis_coord: torch.Tensor,
    vis_seq: torch.Tensor,
    vis_mask: torch.Tensor,
    n_text: int,
    k: int,
    grid: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    拼出整条序列的 (pos1d, coord, is_visual)。布局照 OpenVLA：

        [BOS] [视觉 N 个] [文本 n_text 个]        modeling_prismatic.py:381

    ⚠️ **文本起始位置固定为 `k*grid*grid + 1`，与存活的视觉 token 数无关。**
    这是本模块最不能改的一行。若文本起点随池化率浮动，G3/G4/M2 之间比较的
    就不再是坐标系，而是位置扰动——而位置扰动值 54 个点，比要测的效应大一个量级
    （`docs/05` §7.4）。三臂的文本块因此落在完全相同的位置上。
    """
    b, n = vis_coord.shape[:2]
    c = vis_coord.shape[-1]
    device = vis_coord.device
    text_start = float(k * grid * grid + 1)

    pos1d = torch.cat([
        torch.zeros(b, 1, device=device),                          # BOS
        vis_seq + 1.0,                                             # 视觉：质心下标
        torch.arange(n_text, device=device, dtype=torch.float32)
        .expand(b, -1) + text_start,                               # 文本：固定起点
    ], dim=1)

    coord = torch.cat([
        torch.zeros(b, 1, c, device=device),
        vis_coord,
        torch.zeros(b, n_text, c, device=device),
    ], dim=1)

    is_visual = torch.cat([
        torch.zeros(b, 1, dtype=torch.bool, device=device),
        vis_mask,                                                  # padding 槽不算视觉
        torch.zeros(b, n_text, dtype=torch.bool, device=device),
    ], dim=1)
    return pos1d, coord, is_visual


# ---------------------------------------------------------------- 自检
if __name__ == "__main__" and __package__ in (None, ""):
    # 直接 `python src/.../x.py` 跑自检时把 src/ 放上 sys.path。
    # 各脚本用的都是 `from common.x import ...` 这种以 src/ 为根的写法
    # （scripts/*.py 里那句 sys.path.insert(..., "src") 就是干这个的），
    # 少了这几行就只有 rope4d 一个模块得换个跑法，是纯粹的绊脚石。
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    from pooling.coord_pool import (GRID, N_T_DEFAULT, coord_bin_pool,
                                        grid_coords, grid_extent, metric_coords,
                                        metric_extent)

    torch.manual_seed(0)
    K, N, D, NTXT, B = 8, 256, 64, 12, 2
    T = K * GRID * GRID
    plan = channel_plan(n_axes=4)          # G4 / M2 的池化坐标是 4 轴
    plan3 = channel_plan(n_axes=3)         # G3 的是 3 轴
    print(f"通道划分（head_dim={HEAD_DIM}，{plan['n_pairs']} 个 rotary pair）")
    print(f"  4 轴 (G4/M2): {[len(x) for x in plan['d4']]} 转 {plan['n_4d']}，"
          f"1D {len(plan['d1'])}")
    print(f"  3 轴 (G3)   : {[len(x) for x in plan3['d4']]} 转 {plan3['n_4d']}，"
          f"1D {len(plan3['d1'])}")
    assert plan["n_4d"] == plan3["n_4d"], "两臂参与旋转的 pair 数必须相等"
    print("  ✓ 两臂参与旋转的 pair 数相等 —— 堵掉「G4 转的通道更多」这条替代解释\n")

    feat = torch.randn(B, T, D)
    gc = grid_coords(K).unsqueeze(0).repeat(B, 1, 1)
    glo, ghi = grid_extent(K)
    depth = 1.2 + 0.2 * torch.sin(torch.linspace(0, 9, T)).reshape(1, K, -1).repeat(B, 1, 1)
    # 合成相机；真机参数由 scripts/dump_camera.py 导出并用仿真器真值验证
    from common.camera import Camera as _Cam
    _cam = _Cam(fovy=45.0, height=224, width=224,
                pos=torch.zeros(3, dtype=torch.float64),
                rot=torch.eye(3, dtype=torch.float64), flipped=True)
    mc = metric_coords(depth, K, _cam)
    bbox = torch.stack([mc[..., 1:].reshape(-1, 3).min(dim=0).values,
                        mc[..., 1:].reshape(-1, 3).max(dim=0).values])
    mlo, mhi = metric_extent(K, bbox)

    g4 = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=N_T_DEFAULT)
    vc = normalize(g4.coord, mlo, mhi, K)
    vs = seq_centroid(g4.assign, N)
    pos1d, coord, isv = assemble(vc, vs, g4.mask, NTXT, K)
    cos, sin = build_rope(pos1d, coord, isv, plan=plan)
    print(f"序列 {pos1d.shape[1]} = 1 BOS + {N} 视觉 + {NTXT} 文本\n")

    print("[1] 4D 通道置恒等后，必须逐位退化成基线 1D RoPE（消融的干净退路）")
    zero_plan = {"n_pairs": plan["n_pairs"], "n_4d": 0, "d4": [[], [], [], []],
                 "d1": list(range(plan["n_pairs"]))}
    c0, s0 = build_rope(pos1d, coord, isv, plan=zero_plan)
    f = _inv_freq(plan["n_pairs"], total=plan["n_pairs"])
    a = pos1d.unsqueeze(-1).float() * f
    a = torch.cat([a, a], dim=-1)
    assert torch.allclose(c0, a.cos(), atol=1e-6) and torch.allclose(s0, a.sin(), atol=1e-6)
    print("    ✓ 与教科书 1D RoPE 逐位一致")

    print("\n[2] 文本 token 在 4D 通道上必须是单位旋转（预训练通路不受影响）")
    d4all = sum(plan["d4"], [])
    tx = slice(1 + N, None)
    assert torch.allclose(cos[:, tx][..., d4all], torch.ones(1)) and \
           torch.allclose(sin[:, tx][..., d4all], torch.zeros(1))
    print("    ✓ cos=1, sin=0")

    print("\n[3] ⭐ q·k 只依赖坐标之差 —— §1.3② 全部压在这条上")
    q, kk = torch.randn(1, 1, HEAD_DIM), torch.randn(1, 1, HEAD_DIM)
    p1 = torch.tensor([[[2.0, 3.0, 4.0, 5.0]]])
    p2 = torch.tensor([[[6.0, 1.0, 9.0, 2.0]]])
    one = torch.ones(1, 1, dtype=torch.bool)
    z1 = torch.zeros(1, 1)

    def logit(pa, pb, shift):
        ca, sa = build_rope(z1, pa + shift, one, plan=plan)
        cb, sb = build_rope(z1, pb + shift, one, plan=plan)
        return (apply_rope(q, ca, sa) * apply_rope(kk, cb, sb)).sum()

    base_l = logit(p1, p2, 0.0)
    for sh in (1.0, -3.5, 100.0):
        d = abs(logit(p1, p2, sh) - base_l).item()
        assert d < 1e-3, f"平移 {sh} 后 logit 变了 {d}"
    print(f"    ✓ 整体平移 (+1, −3.5, +100) 后 logit 不变（|Δ| < 1e-3）")
    print("    → 度量坐标下这个不变量就是「与绝对位置无关、只看相对几何」，")
    print("      也是为什么用 RoPE 而不是 VLA-4D 的 Fourier 加性编码（docs/04 §5.3）")

    print("\n[4] ⚠️ 文本位置必须与存活视觉 token 数无关（位置扰动值 54 个点）")
    outs = []
    for n_keep in (120, 180, 240):
        e = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=N_T_DEFAULT, enforce_n=n_keep)
        p, cd, iv = assemble(normalize(e.coord, mlo, mhi, K),
                             seq_centroid(e.assign, n_keep), e.mask, NTXT, K)
        outs.append(p[:, -NTXT:])
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[1], outs[2])
    print(f"    ✓ 存活 120 / 180 / 240 三种情况下，文本块位置逐位相同"
          f"（起点固定 {K*GRID*GRID+1}）")

    print("\n[5] G3 与 G4 只差传进来的 coord，走的是同一条代码路径")
    g3 = coord_bin_pool(feat, gc, N, glo, ghi, n_t=N_T_DEFAULT)
    p3, c3, i3 = assemble(normalize(g3.coord, glo, ghi, K),
                          seq_centroid(g3.assign, N), g3.mask, NTXT, K)
    c3c, c3s = build_rope(p3, c3, i3, plan=plan3)
    assert c3c.shape == cos.shape and not torch.allclose(c3c, cos)
    print("    ✓ 形状一致、数值不同 —— 正是「只换坐标张量」要的效果")

    print("\n[6] M2 错配臂：池化侧与 G4 相同，PE 侧改吃 (t,h,w) 质心")
    m2 = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=N_T_DEFAULT)
    assert torch.equal(m2.feat, g4.feat), "M2 的池化侧必须与 G4 逐位相同"
    # PE 侧：用同一批箱，但坐标取 (t,h,w) 的箱内均值 —— 同一套平均逻辑（§3.0.4 ①）
    m2_grid = coord_bin_pool(gc, mc, N, mlo, mhi, n_t=N_T_DEFAULT).feat[..., :3]
    pm, cm, im = assemble(normalize(m2_grid, glo, ghi, K),
                          seq_centroid(m2.assign, N), m2.mask, NTXT, K)
    cmc, _ = build_rope(pm, cm, im, plan=plan3)   # M2 的 PE 侧是 (t,h,w) 三轴
    assert not torch.allclose(cmc, cos), "M2 的 PE 必须与 G4 不同，否则错配臂无效"
    print("    ✓ 池化侧逐位相同、PE 侧不同 —— 强命题的对照成立")
    print("      注：M2 的 (h̄,w̄) 由『把 grid 坐标当特征、走同一次分箱』求出，")
    print("      与 G4 的度量质心共用同一份平均逻辑，没有第二套实现")
    print("      M2 的 PE 侧是三轴，因此与 G3 用同一个 plan3 —— 转的 pair 数与 G4 相等")

    print("\n全部通过。")
    print("""
⚠️ 尚未验证、必须在真模型上做的两件事：
  ① 接入 LlamaAttention 后端到端跑通（这里只验了 cos/sin 的数学性质）
  ② 「从高频端拨通道」是推理不是实测，要与低频端、两端各半做消融
     （docs/06 §2.3 细节 1）
""")
