"""
坐标空间分箱池化 —— 主线 Phase 1 的池化算子（docs/06 §2.2）。

**这一份代码同时是 G2 / G3 / G4 / M2 四组的池化侧。**
四组的差别只有两处，都在调用方：

    G2  coords=(t,h,w)      group_axes=(0,)   ← t 轴不参与合并 = 帧独立
    G3  coords=(t,h,w)      group_axes=()
    G4  coords=(t,x,y,z)    group_axes=()
    M2  coords=(t,x,y,z)    group_axes=()     ← 池化侧与 G4 逐位相同，
                                                只有 PE 侧另取 (t,h,w) 质心

于是「G3 vs G4 只差传进去的坐标张量」这句话在代码层面是字面成立的，
连"两组实现不一样"这种质疑都堵死。

为什么 Phase 1 不用 ToMe（docs/06 §2.2）：ToMe 的相似度里要带几何项
`sim = cos(Eᵢ,Eⱼ) − λ‖pᵢ−pⱼ‖`，而 **λ 在度量空间（米）与网格空间（格）
量纲不同、最优值必然不同**——各自调优等于给 G4 开小灶，核心对照会被一句话问废。
硬分箱守住的规矩是**"没有在两臂之间取值不同的自由参数"**：空间轴的箱数由预算
按确定性规则求出，时间轴的箱数 `n_t` 是显式的**无量纲整数、两臂取同一个值**
（为什么必须显式给定，见 `_resolve_bins` —— 放任自由细化会让主实验静默失去功效）。

⚠️ 已知代价：分箱均值池化正是 docs/04 §4.3 判为"只作 baseline"的那类算子。
   若粗到两组都被地板效应压住，差值会被压向 0 造成假阴性。
   故判读非对称：G4 > G3 是强结论；G4 ≈ G3 必须升级到 ToMe 再测一次。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# LIBERO 的 patch 网格（224/14 = 16）
GRID = 16

# t 轴默认箱数。K=8 下取 2 ⇒ 每个时间箱覆盖 4 帧，静止表面能真正塌成一个 token。
# 这个值要作为消融整组扫描（{1,2,4,8}），G3/G4/M2 始终取同一个值。
N_T_DEFAULT = 2


@dataclass
class PoolOut:
    """池化结果。`feat`/`coord` 的第 1 维恒为 budget，未占满的槽由 `mask` 标出。"""

    feat: torch.Tensor      # (B, N, D)  箱内特征均值
    coord: torch.Tensor     # (B, N, C)  箱内坐标均值 = 质心，供位置编码使用
    mask: torch.Tensor      # (B, N)     bool，True = 真实 token，False = padding
    size: torch.Tensor      # (B, N)     每个箱里的原始 patch 数
    assign: torch.Tensor    # (B, T)     每个原始 patch 落到哪个输出槽，-1 = 被丢弃
    n_used: torch.Tensor    # (B,)       实际用掉的槽数

    def __len__(self) -> int:
        return self.feat.shape[1]


# ---------------------------------------------------------------- 坐标构造
def grid_coords(k: int, grid: int = GRID, device=None) -> torch.Tensor:
    """(t, h, w) 图像网格坐标，(k*grid*grid, 3)。G2/G3 用，也是 M2 的 PE 侧坐标。"""
    t = torch.arange(k, device=device, dtype=torch.float32)
    h = torch.arange(grid, device=device, dtype=torch.float32)
    w = torch.arange(grid, device=device, dtype=torch.float32)
    tt, hh, ww = torch.meshgrid(t, h, w, indexing="ij")
    return torch.stack([tt, hh, ww], dim=-1).reshape(-1, 3)


def metric_coords(depth: torch.Tensor, k: int, camera, grid: int = GRID) -> torch.Tensor:
    """
    (t, x, y, z) **世界系**度量坐标，(B, k*grid*grid, 4)。G4/M2 的池化侧坐标。

    `depth` 是 patch 级深度 (B, k, grid*grid)，由仿真器回放离线缓存
    （docs/06 §2.4：只存 256 个值/帧，不是全分辨率深度图）。
    `camera` 是 `common.camera.Camera`，由 `scripts/dump_camera.py` 导出并
    **用仿真器真值验证过**的常量参数。

    ⚠️ `camera` 是**必填**的。早先这里是个占位实现，返回 `(t, h, w, z)`——
    x/y 还是图像网格坐标。那样的"G4"只是一个带深度通道的 G3，
    **而它不会报任何错**：分箱照跑、训练照收敛、对照表照填。
    所以宁可让调用方多传一个参数，也不给一个能静默降级的默认值。
    """
    if camera is None:
        raise ValueError(
            "metric_coords 需要相机参数才能做反投影。"
            "先跑 scripts/dump_camera.py 导出，再 Camera.from_json(...) 传进来。"
            "缺了它就只能得到图像网格坐标，那是 G3 不是 G4。")
    b = depth.shape[0]
    d = depth.reshape(b, k, grid * grid)
    xyz = camera.patch_xyz(d).to(depth.dtype)                        # (B, k, 256, 3)
    t = torch.arange(k, device=depth.device, dtype=depth.dtype)
    t = t.view(1, k, 1, 1).expand(b, k, grid * grid, 1)
    return torch.cat([t, xyz], dim=-1).reshape(b, -1, 4)             # (B, T, 4) = (t,x,y,z)


# ---------------------------------------------------------------- 分箱规则
def _bin_counts(idx: torch.Tensor) -> int:
    """给定每个 patch 的整数箱下标 (T, C)，数有多少个非空箱。"""
    return torch.unique(idx, dim=0).shape[0]


def _resolve_bins(
    q: torch.Tensor,
    budget: int,
    group_axes: tuple[int, ...],
    n_group: tuple[int, ...],
    n_t: int | None = None,
) -> list[int]:
    """
    确定每个轴分几个箱。**这是本模块唯一的"超参"，而它由规则定死、不可调。**

    规则：**贪心细化当前最粗的轴**，每次把某个轴的箱数 +1，只要非空箱数仍 ≤ budget；
    并列时取轴下标小者。没有任何一个轴还能细化时停止。

    为什么用这条规则而不是"各轴均分"（docs/06 §3.0.5 ③）：
    均分只能取整数立方根，G3 三轴在 budget=256 下只能到 6×6×6=216，白白浪费 40 个槽；
    贪心细化能走到 7×6×6=252。而它同样是**确定性的、无自由参数的**，
    并且对 G3 与 G4 用的是同一条规则，没有各自调优的余地。

    ⚠️ ~~它同时防住 t 轴退化~~ —— **这句话是错的，实测推翻**。见下面 `n_t` 的说明：
       贪心细化确实不会把 t 轴一次推满，但会推到 6–7 个箱，**跨帧合并照样几乎不发生**。
       所以 t 轴的箱数必须由 `n_t` 显式钉死，不能交给贪心。

    `group_axes` 里的轴不参与细化，直接按 `n_group` 给的全分辨率分箱——
    G2 用它把 t 轴钉死成"每帧一箱"，从而实现帧独立池化。

    ⚠️⚠️ **`n_t` 必须显式给定，别用自由细化。** 这是实测撞出来的：
    K=8 时 t 轴只有 8 个取值，纯贪心会把它推到 6–7 个箱，于是

        G3 [7,6,6]    跨帧 token 仅 36/252，最大跨度 2 帧
        G4 [6,6,5,5]  跨帧 token 仅 23/251，最大跨度 2 帧

    **两组基本都在做逐帧的空间池化**，后果是三连击：
    ① "静止表面在多帧可见 → 塌成一个带 size 的 token"这个机制根本没发生；
    ② G3 几乎不跨帧 → G3 ≈ G2，粒度对照空转；
    ③ docs/06 §1.2① 的"同一 (h,w) 不同帧可能是不同表面"这个**错误合并机制
       触发不了** → G4 vs G3 也就没什么可赢的，主实验近乎没有功效。
    而这**不会报错**，只会安静地跑出假阴性。

    **`n_t` 不是 λ 那类混淆项。** λ 的问题在于两个坐标系量纲不同（米 vs 格），
    没法取同一个值，各自调优就是给 G4 开小灶。`n_t` 是**无量纲的整数**，
    G3/G4/M2 取**同一个值**，且要整组扫描并同时报告曲线。
    真正要守住的规矩是"没有在两臂之间取值不同的自由参数"，这条仍然成立。

    取值含义：n_t = K 等于不跨帧合并（那就是 G2）；n_t = 1 是完全时间塌缩；
    1 < n_t < K 是部分。默认 `N_T_DEFAULT`。
    """
    c = q.shape[1]
    g = [1] * c
    if n_t is not None:
        g[0] = max(1, n_t)
    for ax, n in zip(group_axes, n_group):
        g[ax] = n

    def occupied(gs: list[int]) -> int:
        # 归一化坐标 q ∈ [0,1)，第 ax 轴分 gs[ax] 个箱
        scale = torch.tensor(gs, device=q.device, dtype=q.dtype)
        idx = torch.clamp((q * scale).floor().long(), min=torch.zeros_like(
            torch.tensor(gs, device=q.device)), max=torch.tensor(
            [x - 1 for x in gs], device=q.device))
        return _bin_counts(idx)

    if occupied(g) > budget:                 # 连最粗的划分都超预算，直接返回
        return g

    fixed = set(group_axes) | ({0} if n_t is not None else set())
    free = [ax for ax in range(c) if ax not in fixed]
    cur = occupied(g)
    while free:
        coarsest = min(g[ax] for ax in free)      # 只细化当前最粗的轴，并列取下标小者
        advanced = False
        for ax in free:
            if g[ax] != coarsest:
                continue
            trial = list(g)
            trial[ax] += 1
            n = occupied(trial)
            # ⚠️ 必须**真的增加**非空箱数才接受。只判 `<= budget` 会死循环：
            # 箱数超过数据本身的分辨率后（比如 h 轴已经 16 个箱、原始就只有 16 行），
            # 再细化非空箱数不变，贪心会永远认为"还能细"。自测直接挂住。
            if n <= budget and n > cur:
                g, cur, advanced = trial, n, True
                break
        if not advanced:
            break
    return g


# ---------------------------------------------------------------- 主算子
def coord_bin_pool(
    feat: torch.Tensor,
    coord: torch.Tensor,
    budget: int,
    lo: torch.Tensor | None = None,
    hi: torch.Tensor | None = None,
    group_axes: tuple[int, ...] = (),
    n_group: tuple[int, ...] = (),
    n_t: int | None = None,
    enforce_n: int | None = None,
) -> PoolOut:
    """
    把 (B, T, D) 的 token 按坐标分箱、箱内平均，压到 budget 个。

    feat   (B, T, D)
    coord  (B, T, C)   C=3 → (t,h,w)；C=4 → (t,x,y,z)
    lo/hi  (C,)        坐标归一化范围。**必须由调用方显式给定且跨样本固定**——
                       若按每个样本自己的 min/max 归一化，同一个物理位置在不同样本里
                       会落到不同的箱，位置编码就失去了跨样本的一致含义。
    n_t                t 轴分几个箱。**必须显式给定**，理由见 `_resolve_bins`——
                       放任自由细化会让跨帧合并几乎不发生，主实验静默失去功效。
                       G3/G4/M2 必须取**同一个值**（它无量纲，可以）。
    enforce_n          若给定，只保留 patch 数最多的 enforce_n 个箱。
                       用来把各组的**有效** token 数拉到严格相等——分箱是离散的，
                       G2/G3/G4 各自能用满的槽数未必一样（见 __main__ 自测的打印），
                       而"token 预算严格相等"是主线对照的前提。

    返回的 `coord` 是**箱内坐标的算术均值**，即质心。三个分量走同一套平均逻辑
    （docs/06 §3.0.5 ①）——M2 的 (t̄,h̄,w̄) 与 G4 的 (t̄,x̄,ȳ,z̄) 由同一份代码算出，
    这样两组的池化侧才是逐位相同的。

    ⚠️ 运动物体的箱里，平均出的 (h̄,w̄) 可能落在既不是起点也不是终点的位置。
       **这是预期行为，不是 bug**——它正是"错配"要暴露的退化。别"顺手修好"。
    """
    b, t, d = feat.shape
    c = coord.shape[-1]
    device = feat.device
    if lo is None or hi is None:
        raise ValueError(
            "lo/hi 必须显式给定：按样本自适应归一化会让同一物理位置在不同样本里"
            "落到不同的箱，位置编码失去跨样本一致性。用 grid_extent()/metric_extent()。"
        )
    span = (hi - lo).clamp(min=1e-8).to(device)
    q = ((coord - lo.to(device)) / span).clamp(0.0, 1.0 - 1e-6)      # (B, T, C) → [0,1)

    n_out = enforce_n or budget
    out_f = feat.new_zeros(b, n_out, d)
    out_c = coord.new_zeros(b, n_out, c)
    out_m = torch.zeros(b, n_out, dtype=torch.bool, device=device)
    out_s = torch.zeros(b, n_out, dtype=torch.long, device=device)
    out_a = torch.full((b, t), -1, dtype=torch.long, device=device)
    used = torch.zeros(b, dtype=torch.long, device=device)

    for i in range(b):
        with torch.no_grad():
            g = _resolve_bins(q[i], budget, group_axes, n_group, n_t)
            gs = torch.tensor(g, device=device, dtype=q.dtype)
            gmax = torch.tensor([x - 1 for x in g], device=device)
            idx = (q[i] * gs).floor().long().clamp(min=torch.zeros_like(gmax), max=gmax)
            keys, inv = torch.unique(idx, dim=0, return_inverse=True)   # (M, C), (T,)
            m = keys.shape[0]
            cnt = torch.zeros(m, device=device).index_add_(
                0, inv, torch.ones(t, device=device))

            # 槽位不足时保留 patch 数最多的箱；并列按箱下标，保证确定性
            if m > n_out:
                order = torch.argsort(
                    cnt * (m + 1) - torch.arange(m, device=device, dtype=cnt.dtype),
                    descending=True)[:n_out]
                sel = torch.zeros(m, dtype=torch.bool, device=device)
                sel[order] = True
            else:
                sel = torch.ones(m, dtype=torch.bool, device=device)

        # 均值池化。scatter 走 fp32：bf16 的 7 位有效位在累加几十个 patch 时会丢精度
        acc_f = feat.new_zeros(m, d, dtype=torch.float32).index_add_(
            0, inv, feat[i].float())
        acc_c = coord.new_zeros(m, c, dtype=torch.float32).index_add_(
            0, inv, coord[i].float())
        mean_f = acc_f / cnt.unsqueeze(-1)
        mean_c = acc_c / cnt.unsqueeze(-1)

        # 输出按 (t̄, 其余分量) 字典序排。乱序等于给模型喂训练时没见过的排列，
        # 测出来的掉点会是排列扰动而非信息损失（docs/04 §4.6 的坑）
        kept = sel.nonzero(as_tuple=True)[0]
        with torch.no_grad():
            key = torch.zeros(kept.shape[0], device=device, dtype=torch.float64)
            for ax in range(c):
                key = key * 1e4 + mean_c[kept, ax].double()
            kept = kept[key.argsort()]

        k = kept.shape[0]
        out_f[i, :k] = mean_f[kept].to(out_f.dtype)
        out_c[i, :k] = mean_c[kept].to(out_c.dtype)
        out_m[i, :k] = True
        out_s[i, :k] = cnt[kept].long()
        used[i] = k

        slot = torch.full((m,), -1, dtype=torch.long, device=device)
        slot[kept] = torch.arange(k, device=device)
        out_a[i] = slot[inv]

    return PoolOut(out_f, out_c, out_m, out_s, out_a, used)


# ---------------------------------------------------------------- 归一化范围
def grid_extent(k: int, grid: int = GRID, device=None):
    """(t,h,w) 的 lo/hi。t ∈ [0,k)，h/w ∈ [0,grid)。"""
    lo = torch.zeros(3, device=device)
    hi = torch.tensor([float(k), float(grid), float(grid)], device=device)
    return lo, hi


def metric_extent(k: int, bbox: torch.Tensor, device=None):
    """
    (t,x,y,z) 的 lo/hi。`bbox` 是 (2, 3) 的工作空间包围盒，**跨样本固定的常量**。

    docs/04 §5.4 坑 2：米制坐标直接塞进 RoPE 会频率混叠，要先按工作空间归一化。
    这里只负责分箱侧的归一化；RoPE 侧把归一化坐标再映到 [0,16)，
    与模型预训练时见过的 16×16 网格频率量级对齐。
    """
    lo = torch.cat([torch.zeros(1, device=device), bbox[0].to(device)])
    hi = torch.cat([torch.tensor([float(k)], device=device), bbox[1].to(device)])
    return lo, hi


# ---------------------------------------------------------------- 退化断言
def assert_cross_frame_merge(
    out: PoolOut, coord: torch.Tensor, t_axis: int = 0, min_frac: float = 0.25
) -> float:
    """
    ⚠️ **开跑前必须过的闸**（docs/06 §3.0.5 ③）。

    检查输出里确实存在**跨越多个原始帧**的 token。若一个都没有，说明分箱在 t 轴上
    每帧一箱，G3 已经退化成 G2 —— 而这**不会报任何错**，只会得到"G3 ≈ G2"，
    再被误读成"跨帧联合合并没有价值"。一次 12 小时的训练就这么白跑了。

    ⚠️ **阈值必须是比例，不能只查 `> 0`。** 初版写的是 `n_cross > 0`，
    而实测下自由细化给出 36/252 = 14% —— **断言过了，机制却没发生**。
    差一点就这么跑出一个假阴性。默认要求跨帧 token 占比 ≥ 25%。

    返回跨帧 token 的占比；低于 min_frac 时抛 AssertionError。
    """
    n_cross = n_tok = 0
    for i in range(out.feat.shape[0]):
        a = out.assign[i]
        keep = a >= 0
        # (slot, t) 去重后按 slot 计数 = 每个 slot 覆盖了几个不同的原始帧。
        # 压成一维键再 unique —— 二维 unique(dim=0) 要按行排序，慢一到两个数量级
        tv = coord[i, :, t_axis][keep].long()
        base = int(tv.max()) + 1
        slots = torch.unique(a[keep] * base + tv).div(base, rounding_mode="floor")
        span = torch.bincount(slots)
        n_tok += int((span > 0).sum())
        n_cross += int((span > 1).sum())
    frac = n_cross / max(n_tok, 1)
    assert frac >= min_frac, (
        f"跨帧 token 只占 {frac:.1%}（要求 ≥ {min_frac:.0%}）—— 分箱在 t 轴过细，"
        f"跨帧合并几乎没发生，G3 实际上退化成了 G2。把 n_t 调小；"
        f"n_t=K 等于完全不跨帧。详见 _resolve_bins 的说明。"
    )
    return frac


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
    torch.manual_seed(0)
    K, N, D = 8, 256, 64
    T = K * GRID * GRID

    feat = torch.randn(2, T, D)
    gc = grid_coords(K).unsqueeze(0).repeat(2, 1, 1)
    glo, ghi = grid_extent(K)

    # ⚠️ 合成深度，只为跑通接口。它把 z 写成展平下标的正弦，(x,y,z) 落在一个
    #    退化的二维流形上，与真实场景的占用结构完全不同。
    #    **下面 G4 的槽位利用率与跨帧占比都不可当作预测**——
    #    真值深度缓存好（docs/06 §4.2）之后必须重测一遍。
    depth = 1.2 + 0.2 * torch.sin(torch.linspace(0, 9, T)).reshape(1, K, -1).repeat(2, 1, 1)
    # 自检用一台合成相机（45° 视场、原点、看向 −z）。真机上的参数由
    # scripts/dump_camera.py 导出并用仿真器真值验证，从 json 读。
    from common.camera import Camera as _Cam
    cam = _Cam(fovy=45.0, height=224, width=224,
               pos=torch.zeros(3, dtype=torch.float64),
               rot=torch.eye(3, dtype=torch.float64), flipped=True)
    mc = metric_coords(depth, K, cam)
    # 包围盒同样由数据本身定，免得自检里出现一个凭空的常量。
    # 真机上这个 bbox 来自 dump_camera.py 的 p1–p99 统计。
    lo3 = mc[..., 1:].reshape(-1, 3).min(dim=0).values
    hi3 = mc[..., 1:].reshape(-1, 3).max(dim=0).values
    bbox = torch.stack([lo3, hi3])
    mlo, mhi = metric_extent(K, bbox)

    print(f"输入 {T} token（K={K} × {GRID}×{GRID}），预算 N={N}\n")

    NT = N_T_DEFAULT
    print(f"t 轴箱数 n_t={NT}（G3/G4/M2 取同一个值）\n")
    g2 = coord_bin_pool(feat, gc, N, glo, ghi, group_axes=(0,), n_group=(K,))
    g3 = coord_bin_pool(feat, gc, N, glo, ghi, n_t=NT)
    g4 = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=NT)
    m2 = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=NT)   # 池化侧与 G4 完全相同

    for name, o in [("G2 帧独立", g2), ("G3 网格", g3), ("G4 度量", g4)]:
        print(f"  {name:10s} 用掉 {int(o.n_used[0]):3d}/{N} 槽   "
              f"箱内 patch 数 {int(o.size[0][o.mask[0]].min())}–{int(o.size[0][o.mask[0]].max())}")

    print("\n[1] M2 的池化侧必须与 G4 逐位相同（只有 PE 侧换坐标）")
    assert torch.equal(m2.feat, g4.feat) and torch.equal(m2.assign, g4.assign)
    print("    ✓ feat / assign 逐位相同")

    print("\n[2] G3 必须真的跨帧合并，否则它就是 G2")
    frac = assert_cross_frame_merge(g3, gc)
    print(f"    ✓ G3 跨帧 token 占比 {frac:.1%}")
    try:
        assert_cross_frame_merge(g2, gc)
        raise SystemExit("    ✗ G2 不该有跨帧 token —— 帧独立池化失效了")
    except AssertionError:
        print("    ✓ G2 没有跨帧 token（帧独立，符合预期）")

    print("\n[2b] ⚠️ n_t 不给定（放任贪心细化）时，断言必须拦下来")
    loose = coord_bin_pool(feat, gc, N, glo, ghi)          # n_t=None
    try:
        assert_cross_frame_merge(loose, gc)
        raise SystemExit("    ✗ 没拦住 —— 断言阈值太松，会放过假阴性")
    except AssertionError as e:
        print(f"    ✓ 拦下了：{str(e).splitlines()[0][:60]}…")
    print("    这正是实测撞出来的坑：自由细化给 [7,6,6]，跨帧仅 14%，"
          "旧断言（>0）能过，机制却没发生")

    print("\n[3] 每个 patch 恰好落进一个箱，不重不漏")
    for name, o in [("G2", g2), ("G3", g3), ("G4", g4)]:
        assert (o.assign[0] >= 0).all(), f"{name} 有 patch 被丢弃"
        assert int(o.size[0].sum()) == T, f"{name} size 之和 {int(o.size[0].sum())} ≠ {T}"
    print("    ✓ 三组的 size 之和都等于输入 token 数")

    print("\n[4] 确定性：同输入重跑必须逐位相同")
    assert torch.equal(coord_bin_pool(feat, mc, N, mlo, mhi, n_t=NT).feat, g4.feat)
    print("    ✓ 无初始化敏感、无随机性")

    print("\n[5] 输出按质心字典序排列（送进 Llama 的顺序必须稳定）")
    for name, o in [("G3", g3), ("G4", g4)]:
        cm = o.coord[0][o.mask[0]]
        key = torch.zeros(cm.shape[0], dtype=torch.float64)
        for ax in range(cm.shape[-1]):
            key = key * 1e4 + cm[:, ax].double()
        assert torch.equal(key.argsort(), torch.arange(cm.shape[0])), f"{name} 顺序不单调"
    print("    ✓ 有序")

    print("\n[6] enforce_n 把各组有效 token 数拉到严格相等")
    common = int(min(g2.n_used.min(), g3.n_used.min(), g4.n_used.min()))
    e2 = coord_bin_pool(feat, gc, N, glo, ghi, group_axes=(0,), n_group=(K,), enforce_n=common)
    e3 = coord_bin_pool(feat, gc, N, glo, ghi, n_t=NT, enforce_n=common)
    e4 = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=NT, enforce_n=common)
    assert e2.mask.sum() == e3.mask.sum() == e4.mask.sum()
    print(f"    ✓ 三组统一到 {common} 个有效 token（{int(e2.mask[0].sum())} per sample）")
    print("    ⚠️ 主线跑之前用它把预算拉平 —— 分箱是离散的，"
          "各组能用满的槽数天然不同，而『预算严格相等』是核心对照的前提")

    print("\n[7] n_t 扫描 —— 这条曲线要作为消融整组报告，G3/G4 同值")
    print("      n_t   G3 跨帧占比   G4 跨帧占比   （n_t=K 即完全不跨帧 = G2）")
    for nt in (1, 2, 4, 8):
        a = coord_bin_pool(feat, gc, N, glo, ghi, n_t=nt)
        c_ = coord_bin_pool(feat, mc, N, mlo, mhi, n_t=nt)
        fa = assert_cross_frame_merge(a, gc, min_frac=0.0)
        fc = assert_cross_frame_merge(c_, mc, min_frac=0.0)
        print(f"      {nt:<5d} {fa:>10.1%}   {fc:>11.1%}")

    print("""
⚠️ 两处只在真实深度下才有意义，别拿上面的数当结论：

  ① G4 的槽位利用率明显低于 G3（161 vs 242）。这是合成深度退化造成的，
     但**机制是真的**：G3 的网格箱数与数据无关，G4 的体素占用取决于场景，
     所以 enforce_n 拉平后的公共预算会被 G4 拖低。真实深度下要重测，
     若仍明显偏低，说明 G4 的度量分箱在浪费预算，要调 metric_extent 的包围盒。

  ② G4 的跨帧占比随 n_t 上升掉得比 G3 快。同样先归因于合成深度
     （z 跨帧恒定，(x,y,z) 几乎没有时间结构），真值深度下重看。

✅ metric_coords 已换成真正的反投影（common/camera.py），x/y 是世界系坐标而非
图像网格。相机参数须由 scripts/dump_camera.py 导出——那个脚本会用仿真器
自己的 site 世界坐标做锚点验证，因为坐标系约定错了不会崩，只会得到一个
镜像或偏移的世界。
""")
    print("全部通过。")
