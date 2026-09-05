"""
模型侧接线 —— 把 K 帧输入、坐标池化、4D RoPE 挂进 OpenVLA。

**七组对照全部由这一份代码产生**，差别只在 `WireConfig.arm`：

    G0  K=1，不池化，1D RoPE                        （单帧基线，等于原模型）
    G1  K=8，不池化（2048 token），1D RoPE          （上限参考）
    G2  K=8，帧独立池化到 N，PE 用 (t,h,w)
    G3  K=8，跨帧池化，池化坐标 (t,h,w)，PE 也是 (t,h,w)      ← 一致
    G4  K=8，跨帧池化，池化坐标 (t,x,y,z)，PE 也是 (t,x,y,z)  ← 一致，本方案
    M2  K=8，跨帧池化，池化坐标 (t,x,y,z)，PE 却用 (t,h,w)    ← 错配臂
    M3  K=8，跨帧池化，池化坐标 (t,h,w)，PE 却用 (t,x,y,z)    ← 错配臂（M2 的镜像）

四个挂载点（transformers 4.40.1，见 `_patch_rope` 的版本约定）：

    vision_backbone.forward   (B, K*6, H, W) → (B, K*256, D_vis)
    projector.forward         投影后池化到 N 个 token，并算出 PE 要用的坐标
    每层 attention.rotary_emb 换成我们的 (cos, sin)
    language_model.forward    补上 position_ids —— 官方传的是 None

⚠️ **`set_batch()` 必须在每次 forward 前调用**，把这一批的深度、补帧掩码、
   task_id 交进来（它们不在 `pixel_values` 里，也进不了 `forward` 的签名）。
   忘了调 = 拿上一批的深度算这一批的坐标，**不会报错**。所以 state 是一次性的：
   消费掉就作废，下一次 forward 拿不到就抛异常。

`python src/pooling/wire.py` 跑 9 项自检（桩模型，不加载 7B）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

if __name__ == "__main__" and __package__ in (None, ""):
    # 直接 `python src/pooling/wire.py` 跑自检时把 src/ 放上 sys.path。
    # ⚠️ 必须在下面的 `from pooling...` **之前**，其它模块的跨模块 import 都在
    #    函数体里、放文件末尾也行，这个文件是顶层 import，放晚了就来不及。
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

from pooling.coord_pool import (GRID, N_T_DEFAULT, coord_bin_pool, grid_coords,
                                grid_extent, metric_coords, metric_extent)
from pooling.rope4d import assemble, build_rope, channel_plan, normalize, seq_centroid

N_PATCH = GRID * GRID          # 256
ARMS = ("G0", "G1", "G2", "G3", "G4", "M2", "M3")


@dataclass
class WireConfig:
    arm: str
    K: int = 8
    budget: int = N_PATCH               # docs/06 §3.0：N=256，所有组同预算
    n_t: int = N_T_DEFAULT
    head_dim: int = 128
    enforce_n: Optional[int] = None     # 跨臂拉平后的公共预算
    bbox: Optional[torch.Tensor] = None  # (2,3) 工作空间包围盒，G4/M2 必需

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"arm 必须是 {ARMS} 之一，收到 {self.arm!r}")
        if self.arm == "G0" and self.K != 1:
            raise ValueError("G0 是单帧基线，K 必须为 1")
        if self.needs_depth and self.bbox is None:
            raise ValueError(
                f"{self.arm} 用度量坐标，必须给 bbox（scripts/dump_camera.py 的产物）。"
                "缺了它 metric_extent 无从归一化。")

    @property
    def pools(self) -> bool:
        return self.arm in ("G2", "G3", "G4", "M2", "M3")

    @property
    def metric(self) -> bool:
        """**池化侧**是否用度量坐标。"""
        return self.arm in ("G4", "M2")

    @property
    def pe_metric(self) -> bool:
        """**PE 侧**是否用度量坐标。与 `metric` 分开，2×2 的四格就是这两个的组合：

            池化\PE      (t,h,w)      (t,x,y,z)
            (t,h,w)        G3            M3        ← M3 ≈「SpatialVLA 的 PE 装在池化模型上」
            (t,x,y,z)      M2            G4

        **没有 M3，标题句可以被整个重新解释成「就是 Ego3D PE 有用」**，
        而那个解释有一篇发表论文和 +27.3 点撑着（docs/06 §1.3）。
        """
        return self.arm in ("G4", "M3")

    @property
    def needs_depth(self) -> bool:
        """哪一侧用度量坐标都要深度与包围盒。"""
        return self.metric or self.pe_metric

    @property
    def pe_axes(self) -> int:
        """PE 侧坐标的轴数。错配就体现在这里：M2 池化 4 轴 / PE 3 轴，M3 反过来。"""
        return 4 if self.pe_metric else 3

    def pool_extent(self, device=None):
        """**池化侧**的量程，跟着 `metric` 走。"""
        if self.metric:
            return metric_extent(self.K, self.bbox.to(device), device=device)
        return grid_extent(self.K, device=device)

    def pe_extent(self, device=None):
        """
        **PE 侧**的量程，跟着 `pe_metric` 走 —— 与 `pool_extent` 是两个东西。

        ⚠️ 这里曾经写成「`arm == "G4"` 才用度量量程」，于是 M3（网格池化 +
        度量 PE）拿 3 轴的网格量程去归一化 4 轴的度量质心。**幸好它当场报维度
        不匹配**：这是本轮少见的不静默的错，冒烟测试五分钟就抓到了。真正危险的
        是它的镜像——M2 若拿度量量程去归一化 3 轴网格质心，广播恰好合法，
        坐标会被静静地压扁。所以量程只准从这两个函数取，别在调用点各判一次。
        """
        if self.pe_metric:
            return metric_extent(self.K, self.bbox.to(device), device=device)
        return grid_extent(self.K, device=device)


@dataclass
class _Batch:
    """一次 forward 所需的、进不了 `forward` 签名的东西。"""
    depth: Optional[torch.Tensor] = None        # (B, K, 256) patch 级深度，米
    frame_pad_mask: Optional[torch.Tensor] = None   # (B, K) bool，True = 真实帧
    cameras: list = field(default_factory=list)     # 每个样本一个 Camera


class _State:
    """
    一次性的批次状态。**消费掉就作废。**

    这不是洁癖：depth / camera 不在 pixel_values 里，忘了 set_batch 就会拿
    上一批的深度算这一批的坐标，数值上完全合理、loss 照降，只是全错。
    """

    def __init__(self):
        self.batch: Optional[_Batch] = None
        self.uses_left: int = 0         # 还允许被消费几次，见 set_batch(n_uses=)
        self.vision_feats = None        # 评测用的预算特征，见 set_vision_feats
        self.rope: Optional[tuple] = None
        self.rope_calls: int = 0        # 我们的 rotary_emb 被真正调用了几次
        self.cfg = None                 # wire() 时存下，assert_arm_wiring 要按臂判
        self.pe_axes_seen: int = 0      # 上一次 forward 里 PE 坐标真的是几轴
        self.orig: dict = {}            # 原始实现，unwire 时还回去
        # 32 层用的是同一条序列，cos/sin 只算一次（build_rope 不便宜）
        self.rope_cache = None
        self.rope_cache_key = None

    def take(self) -> _Batch:
        if self.batch is None or self.uses_left <= 0:
            raise RuntimeError(
                "forward 前没有调用 set_batch()。深度与相机不在 pixel_values 里，"
                "沿用上一批的值不会报错、loss 照降，但坐标全是错的 —— 所以这里硬抛。"
                "\n（`use_cache=False` 的自回归生成会**逐步重跑整条前向**，"
                "投影器于是被调用 1+max_new_tokens 次 —— 那种场合要 "
                "`set_batch(..., n_uses=...)` 把次数说清楚，而不是把它改成常驻。）")
        b = self.batch
        self.uses_left -= 1
        if self.uses_left == 0:
            self.batch = None
        self.rope_cache = self.rope_cache_key = None      # 新一批，缓存作废
        return b


# ---------------------------------------------------------------- 各挂载点
def _patch_vision(model, k: int, state: Optional["_State"] = None) -> None:
    """(B, K*6, H, W) → (B, K*256, D)。与 `scripts/probe_vram.py` 同一套约定。"""
    orig = model.vision_backbone.forward

    def wrapped(pixel_values, *a, **kw):
        # ⭐ 评测时同一帧会在连续 K 个时刻各出现一次，而**视觉主干是冻结的**
        #    （LoRA 不含视觉），同一帧的特征逐位相同。所以评测侧每帧只算一次、
        #    缓存复用，这里直接收下算好的。见 set_vision_feats 与
        #    run_eval_kframe.py 的 --verify_vision_cache（逐位对拍）。
        #    ⚠️ 只在评测用：训练要走增广，同一帧每次的像素都不同，不能缓存。
        if state is not None and state.vision_feats is not None:
            f, state.vision_feats = state.vision_feats, None
            return f
        if k == 1:
            return orig(pixel_values, *a, **kw)
        return torch.cat([orig(c, *a, **kw)
                          for c in torch.split(pixel_values, 6, dim=1)], dim=1)

    model.vision_backbone.forward = wrapped


def _pool_and_coords(emb: torch.Tensor, cfg: WireConfig, bt: _Batch):
    """
    投影后的 (B, K*256, D) → 池化后的 (B, N, D) + PE 侧要用的东西。

    返回 (emb_out, pos1d, coord_pe, is_visual_len)。
    """
    b, _, _ = emb.shape
    k, dev = cfg.K, emb.device
    valid = None
    if bt.frame_pad_mask is not None:
        # (B,K) → (B, K*256)：补帧的 256 个 patch 全部屏蔽（docs/06 §4.1 纪律 2）
        valid = bt.frame_pad_mask.to(dev).repeat_interleave(N_PATCH, dim=1)

    gc = grid_coords(k, device=dev).unsqueeze(0).expand(b, -1, -1)
    if not cfg.pools:
        # G0 / G1：不池化。PE 侧仍要坐标（G1 用 (t,h,w)），位置用原始下标
        pos1d = torch.arange(k * N_PATCH, device=dev).float().expand(b, -1)
        return emb, pos1d, gc, None

    # 度量坐标：**池化侧或 PE 侧任一需要就得算**（M3 只有 PE 侧要）
    pc_metric = None
    if cfg.needs_depth:
        if not bt.cameras:
            raise RuntimeError(f"{cfg.arm} 需要每个样本的相机，set_batch 里没给。")
        # ⚠️ **坐标必须留在 fp32**，不能跟着 emb 转成 bf16。
        #    bf16 只有 8 位尾数，在 ±2 m 量程上相邻可表示值差约 1 cm ——
        #    正好把 G4 要分辨的尺度磨掉，而且**不报任何错**：
        #    分箱照跑、训练照收敛，只是度量坐标退化成了厘米级的量化网格。
        #    特征走 bf16，坐标走 fp32，两者本来就不必同精度。
        pc_metric = torch.cat(
            [metric_coords(bt.depth[i:i + 1].to(dev), k, bt.cameras[i])
             for i in range(b)], dim=0).float()

    pc = pc_metric if cfg.metric else gc
    lo, hi = cfg.pool_extent(dev)

    kw = dict(group_axes=(0,), n_group=(k,)) if cfg.arm == "G2" else dict(n_t=cfg.n_t)
    out = coord_bin_pool(emb, pc, cfg.budget, lo, hi,
                         enforce_n=cfg.enforce_n, valid=valid, **kw)

    # PE 侧坐标 —— 2×2 的四格在这里分开。**两个错配臂都是"另取一套坐标算质心"**，
    # 走的是同一份 `_grid_centroid`（它对任意坐标张量通用），
    # 所以 M2 与 M3 的实现是彼此镜像的，不存在"哪一臂被特殊照顾"。
    n_slots = cfg.enforce_n or cfg.budget
    if cfg.metric and not cfg.pe_metric:        # M2：度量池化 + 网格 PE
        coord_pe = _grid_centroid(out.assign, gc, n_slots)
    elif cfg.pe_metric and not cfg.metric:      # M3：网格池化 + 度量 PE
        coord_pe = _grid_centroid(out.assign, pc_metric, n_slots)
    else:                                        # G3 / G4：两侧一致，直接用池化质心
        coord_pe = out.coord

    pos1d = seq_centroid(out.assign, cfg.enforce_n or cfg.budget)
    return out.feat, pos1d, coord_pe, out.mask


def _grid_centroid(assign: torch.Tensor, gc: torch.Tensor, n_slots: int) -> torch.Tensor:
    """每个输出槽里成分 patch 的 (t,h,w) 算术均值。M2 的 PE 侧坐标。"""
    b, t = assign.shape
    out = gc.new_zeros(b, n_slots, gc.shape[-1])
    for i in range(b):
        m = assign[i] >= 0
        idx = assign[i][m]
        cnt = torch.zeros(n_slots, device=gc.device).index_add_(
            0, idx, torch.ones(int(m.sum()), device=gc.device)).clamp(min=1)
        acc = gc.new_zeros(n_slots, gc.shape[-1]).index_add_(0, idx, gc[i][m])
        out[i] = acc / cnt.unsqueeze(-1)
    return out


def _patch_projector(model, cfg: WireConfig, state: _State) -> None:
    orig = model.projector.forward

    def wrapped(img_patches, *a, **kw):
        emb = orig(img_patches, *a, **kw)                 # (B, K*256, D_llm)
        bt = state.take()
        emb, pos1d, coord_pe, _ = _pool_and_coords(emb, cfg, bt)

        # ⚠️ 这里**只准备视觉部分**，整条序列留到 `_Rope.forward` 再拼。
        #    投影器看不到 `input_ids`，文本有多长它不知道；早先在这里写死
        #    `n_text=0`，于是 cos 只有 1+2048 长而 q 是 1+2048+19，
        #    直接在 apply_rotary_pos_emb 里维度对不上。**幸好它会报错** ——
        #    这是本轮少见的一个不静默的错。
        lo, hi = cfg.pe_extent(emb.device)         # ⚠️ PE 侧量程，不是池化侧
        if coord_pe.shape[-1] != lo.numel():
            raise RuntimeError(
                f"{cfg.arm}：PE 坐标 {coord_pe.shape[-1]} 轴，量程 {lo.numel()} 轴。"
                "两者必须同时由 pe_metric 决定。")
        state.pe_axes_seen = int(coord_pe.shape[-1])
        state.rope = (normalize(coord_pe.float(), lo, hi, cfg.K), pos1d)
        return emb

    model.projector.forward = wrapped


def _patch_rope(model, cfg: WireConfig, state: _State) -> None:
    """
    换掉每层 attention 的 `rotary_emb`。

    ⚠️ **版本约定：transformers 4.40.1**（`setup/constraints.txt` 锁的版本）。
    该版本里 `LlamaAttention.forward` 调 `self.rotary_emb(value_states, position_ids)`
    并拿到 `(cos, sin)`，形状 (B, T, head_dim)——正是 `rope4d.build_rope` 的输出形状。
    **4.43 以后旋转嵌入上移到了 `LlamaModel`，这个挂点会失效**（不会报错，
    只是我们的 cos/sin 永远用不上），所以升级 transformers 时必须重跑
    `tests/` 里那条"坐标平移不变"的端到端检验。
    """
    plan = channel_plan(cfg.head_dim, cfg.pe_axes)

    class _Rope(torch.nn.Module):
        def forward(self, x, position_ids=None, seq_len=None):
            if state.rope is None:
                raise RuntimeError("rotary_emb 被调用时 state.rope 还没建好")
            state.rope_calls += 1
            c_norm, pos1d = state.rope
            n_vis = c_norm.shape[1]

            # ⚠️ **带 KV cache 的自回归解码每步只喂 1 个 token**，那时
            #    x.shape[-2] == 1，按它推 n_text 会得到负数。真实位置在
            #    `position_ids` 里（transformers 4.40 就是这么传的），
            #    所以：按最大位置建整条序列的 cos/sin，再按 position_ids 取。
            #    prefill 时 position_ids = arange(T)，取出来与整条一致。
            if position_ids is not None:
                need = int(position_ids.max().item()) + 1
            else:
                need = x.shape[-2]
            n_text = need - 1 - n_vis
            if n_text < 0:
                raise RuntimeError(
                    f"序列长 {need} 比 1+视觉 {1 + n_vis} 还短 —— 池化输出与"
                    "实际喂进 LLM 的视觉块对不上，先查 projector 那一步。")
            if state.rope_cache_key != need:
                p1, c4, isvis = assemble(
                    c_norm, pos1d, torch.ones_like(pos1d, dtype=torch.bool),
                    n_text=n_text, k=cfg.K)
                state.rope_cache = build_rope(p1, c4, isvis, cfg.head_dim, plan)
                state.rope_cache_key = need
            cos, sin = state.rope_cache                      # (B, need, hd)
            if position_ids is not None and position_ids.shape[-1] != cos.shape[1]:
                idx = position_ids.to(cos.device).long()
                if idx.shape[0] != cos.shape[0]:
                    idx = idx.expand(cos.shape[0], -1)
                g = idx.unsqueeze(-1).expand(-1, -1, cos.shape[-1])
                cos, sin = cos.gather(1, g), sin.gather(1, g)
            return cos.to(x.dtype), sin.to(x.dtype)

    layers = model.language_model.model.layers
    n = 0
    for layer in layers:
        # 挂载前先确认这个属性真的存在。4.43+ 把旋转嵌入上移到了 LlamaModel，
        # 那时 self_attn 上没有 rotary_emb，`setattr` 会**安静地新建一个没人用的属性**。
        if not hasattr(layer.self_attn, "rotary_emb"):
            raise RuntimeError(
                "layer.self_attn 上没有 rotary_emb。本模块按 transformers 4.40.1 写，"
                "4.43+ 旋转嵌入上移到了 LlamaModel.rotary_emb —— 挂点变了，"
                "直接 setattr 会新建一个没人调用的属性，4D RoPE 静默失效。"
                "升级 transformers 就必须改这里，并重跑 assert_rope_active。")
        layer.self_attn.rotary_emb = _Rope()
        n += 1
    state.n_layers = n


def wire(model, cfg: WireConfig) -> _State:
    """挂上四个点，返回 state。训练循环每步调 `set_batch(state, ...)`。"""
    state = _State()
    state.cfg = cfg
    state.orig = {
        "vision": model.vision_backbone.forward,
        "projector": model.projector.forward,
        "rotary": [l.self_attn.rotary_emb
                   for l in model.language_model.model.layers],
    }
    _patch_vision(model, cfg.K, state)
    if cfg.arm != "G0":
        _patch_projector(model, cfg, state)
        _patch_rope(model, cfg, state)
    return state


def unwire(model, state: _State) -> None:
    """
    还原成原始模型。**一个进程里连测多组时必须调**，否则包装层会套娃：
    第二次 wire 拿到的 "orig" 已经是第一次的包装，K 帧切分会做两遍，
    视觉 token 数变成 K² 倍 —— 那会直接 OOM，算是这一类里少见的会报错的。
    """
    if not state.orig:
        return
    model.vision_backbone.forward = state.orig["vision"]
    model.projector.forward = state.orig["projector"]
    for layer, rot in zip(model.language_model.model.layers, state.orig["rotary"]):
        layer.self_attn.rotary_emb = rot
    state.orig = {}


def assert_rope_active(state: _State) -> None:
    """
    ⚠️ **第一次 forward 之后必须调一次。**

    挂点错了不会崩：Llama 照样用它自己的 1D RoPE 跑完，loss 正常下降，
    只是 4D RoPE 从头到尾没参与——G4 退化成"坐标只用于池化"，
    标题句那半边（位置编码也用同一套坐标）根本没被检验。
    这类静默失效本项目已经吃过三次（假 OOM、判据错、分箱平台），不再靠人记得。
    """
    if state.rope_calls == 0:
        raise RuntimeError(
            "跑完一次 forward，我们的 rotary_emb 一次都没被调用 —— 4D RoPE 没生效。"
            "多半是 transformers 版本变了（本模块按 4.40.1 写），或 attention 实现"
            "走了别的分支。**不要带着这个状态去训练**：loss 会正常下降，结论全错。")
    exp = getattr(state, "n_layers", 0)
    if exp and state.rope_calls < exp:
        raise RuntimeError(
            f"只有 {state.rope_calls}/{exp} 层用上了 4D RoPE —— 部分层走了别的路径。")


def assert_arm_wiring(state: _State, arm: str) -> None:
    """
    按臂检查接线 —— **第一次 forward 之后必须调一次。**

    G0 是单帧基线，`wire()` 有意不给它挂 RoPE，所以它的判据是**反的**：
    `rope_calls` 必须是 0。早先训练循环无条件调 `assert_rope_active`，
    G0 一开跑就被自己的闸拦下 —— 判据没有按臂分开。

    两个方向都要查：G0 若被挂上了 4D RoPE，它就不再是"等于原模型"的基线，
    而那同样不会报错。

    ⚠️ **PE 轴数也在这里查。** 错配臂与它的同池化伙伴（M3↔G3、M2↔G4）在日志里
    长得一模一样：token 数一样、rope_calls 一样、序列长一样（序列长 = 1+N+文本，
    只随批里的语言指令抖动，**与臂无关** —— 别拿它当判据，我拿它当过一次）。
    真正的区别只有 PE 坐标是 3 轴还是 4 轴，而 `pe_metric` 若判错，
    M3 会安静地退化成 G3：loss 正常、acc 正常、跑满 30k 步，然后 2×2 里
    有一格是假的。所以这条要在第一次 forward 之后当场对拍。
    """
    if arm == "G0":
        if state.rope_calls:
            raise RuntimeError(
                f"G0 是单帧基线，本该完全不接 RoPE，却被调用了 {state.rope_calls} 次 —— "
                "它已经不是'等于原模型'的对照了。查 wire() 里的分支。")
        return
    assert_rope_active(state)
    cfg = state.cfg
    if cfg is not None and cfg.pools:
        if state.pe_axes_seen != cfg.pe_axes:
            raise RuntimeError(
                f"{arm} 的 PE 坐标实际是 {state.pe_axes_seen} 轴，应为 {cfg.pe_axes} 轴。"
                f"{arm} 已经退化成它的同池化伙伴，而这不会有任何别的症状。")


def set_vision_feats(state: _State, feats) -> None:
    """
    交出**已经算好**的视觉特征 (B, K*256, D)，下一次 vision_backbone.forward
    直接返回它、不再跑主干。消费一次即作废。

    ⚠️ **只用于评测。** 训练开着 image_aug，同一帧每次的像素都不同，缓存就是错的。
    """
    state.vision_feats = feats


def frame_feats(state: _State, model, px6) -> torch.Tensor:
    """
    单帧 (1, 6, H, W) → (1, 256, D)，走**未经包装的原始主干**。
    评测侧用它逐帧填缓存。
    """
    return state.orig["vision"](px6)


def set_batch(state: _State, depth=None, frame_pad_mask=None, cameras=(),
              n_uses: int = 1) -> None:
    """
    交出这一批的深度 / 补帧掩码 / 相机。**每次 forward 前都要调。**

    `n_uses` = 这一批允许被投影器消费几次，缺省 1（训练与带 KV cache 的生成
    都只在 prefill 那一次跑投影器）。⚠️ 只有 `use_cache=False` 的自回归生成
    例外：它每步都重跑整条前向，要传 `n_uses = 1 + max_new_tokens`。
    说成一个**具体次数**而不是"常驻"，是为了让"忘了 set_batch"继续报错 ——
    那才是这个机制存在的理由。
    """
    assert n_uses >= 1, n_uses
    state.batch = _Batch(depth=depth, frame_pad_mask=frame_pad_mask,
                         cameras=list(cameras))
    state.uses_left = n_uses


# ---------------------------------------------------------------- 自检
def _selftest() -> None:
    from common.camera import Camera

    K, B, D = 8, 2, 32
    dev = "cpu"
    cam = Camera(fovy=45.0, height=224, width=224,
                 pos=torch.zeros(3, dtype=torch.float64),
                 rot=torch.eye(3, dtype=torch.float64), flipped=True)
    depth = (1.0 + 0.3 * torch.rand(B, K, N_PATCH)).double()
    xyz = torch.cat([cam.patch_xyz(depth[i]).reshape(-1, 3) for i in range(B)])
    bbox = torch.stack([xyz.min(0).values, xyz.max(0).values]).float()
    emb = torch.randn(B, K * N_PATCH, D)

    def run(arm, pad=None):
        # ⚠️ 用 needs_depth 判，不要写死臂名 —— 加 M3 时正是这里先炸的，
        #    而 __post_init__ 的那条校验把它拦下了（该拦）。
        _probe = WireConfig(arm=arm, K=1 if arm == "G0" else K,
                            bbox=bbox)          # 先给 bbox 只为读属性
        cfg = WireConfig(arm=arm, K=K,
                         bbox=bbox if _probe.needs_depth else None)
        bt = _Batch(depth=depth, frame_pad_mask=pad, cameras=[cam] * B)
        return _pool_and_coords(emb, cfg, bt)

    e1, _, _, _ = run("G1")
    assert e1.shape[1] == K * N_PATCH
    for arm in ("G2", "G3", "G4", "M2", "M3"):
        e, p, c, _ = run(arm)
        assert e.shape == (B, N_PATCH, D), (arm, e.shape)
        assert p.shape == (B, N_PATCH) and c.shape[-1] in (3, 4)
    print("✅ 1/9 token 数：G1 保持 2048，五个池化臂都压到 256")

    _, _, c4, _ = run("G4")
    _, _, cm, _ = run("M2")
    assert c4.shape[-1] == 4 and cm.shape[-1] == 3
    print("✅ 2/9 M2 的错配：池化侧与 G4 同为度量坐标，PE 侧是 3 轴 (t,h,w)")

    e4, _, _, _ = run("G4")
    e2, _, _, _ = run("M2")
    assert torch.equal(e4, e2), "M2 的池化侧必须与 G4 逐位相同"
    print("✅ 3/9 M2 与 G4 的池化输出逐位相同（只有 PE 侧不同）")

    # ---- M3：M2 的镜像。2×2 的四格靠这两条不变量钉死 ----
    #      池化\PE     (t,h,w)   (t,x,y,z)
    #      (t,h,w)       G3        M3      ← 与 G3 同池化
    #      (t,x,y,z)     M2        G4      ← M2 与 G4 同池化
    _, _, c3, _ = run("G3")
    _, _, cm3, _ = run("M3")
    assert c3.shape[-1] == 3 and cm3.shape[-1] == 4, (c3.shape, cm3.shape)
    print("✅ 3b/9 M3 的错配（M2 的镜像）：池化侧与 G3 同为网格坐标，"
          "PE 侧是 4 轴 (t,x,y,z)")

    e3, _, _, _ = run("G3")
    em3, _, _, _ = run("M3")
    assert torch.equal(e3, em3), "M3 的池化侧必须与 G3 逐位相同"
    print("✅ 3c/9 M3 与 G3 的池化输出逐位相同（只有 PE 侧不同）")
    print("      → 两个错配臂走同一份 _grid_centroid（它对任意坐标张量通用），"
          "实现彼此镜像，\n         不存在\"哪一臂被特殊照顾\"")

    # ---- 缺的正是这条不变量：PE 坐标的轴数必须与 PE 量程的轴数一致 ----
    #      M3 曾经拿池化侧（3 轴网格）的量程去归一化 4 轴度量质心，当场维度不匹配；
    #      它的镜像（M2 拿 4 轴量程压 3 轴坐标）却会广播成功、静静地算错。
    #      所以这里逐臂对拍，并且**真的调一次 normalize**，不只比形状。
    for arm in ("G0", "G1", "G2", "G3", "G4", "M2", "M3"):
        kk = 1 if arm == "G0" else K
        _probe = WireConfig(arm=arm, K=kk, bbox=bbox)
        cfg_a = WireConfig(arm=arm, K=kk,
                           bbox=bbox if _probe.needs_depth else None)
        bt_a = _Batch(depth=depth[:, :kk], frame_pad_mask=None, cameras=[cam] * B)
        _, _, c_a, _ = _pool_and_coords(emb[:, :kk * N_PATCH], cfg_a, bt_a)
        lo_a, hi_a = cfg_a.pe_extent(c_a.device)
        assert c_a.shape[-1] == lo_a.numel() == cfg_a.pe_axes, (
            arm, c_a.shape, lo_a.numel(), cfg_a.pe_axes)
        q = normalize(c_a.float(), lo_a, hi_a, cfg_a.K)
        assert q.shape == c_a.shape and torch.isfinite(q).all(), arm
    print("✅ 3d/9 七臂逐一对拍：PE 坐标轴数 == pe_extent 轴数 == pe_axes，"
          "且 normalize 真跑得通")

    pad = torch.ones(B, K, dtype=torch.bool)
    pad[:, :3] = False                       # 前 3 帧是补帧
    cfg = WireConfig(arm="G3", K=K)
    bt = _Batch(depth=depth, frame_pad_mask=pad)
    out = coord_bin_pool(
        emb, grid_coords(K).unsqueeze(0).expand(B, -1, -1), cfg.budget,
        *grid_extent(K), n_t=cfg.n_t,
        valid=pad.repeat_interleave(N_PATCH, dim=1))
    assert (out.assign[:, :3 * N_PATCH] == -1).all()
    assert int(out.size[0].sum()) == int(pad[0].sum()) * N_PATCH
    print("✅ 4/9 补帧被屏蔽：assign 全 -1，且不占任何箱")

    for n_keep in (120, 200, 256):
        p1 = torch.arange(n_keep).float().unsqueeze(0)
        c = torch.rand(1, n_keep, 4)
        # 布局是 [BOS] [视觉 n_keep 个] [文本]，所以第一个文本 token 在下标 1+n_keep
        pos, _, isv = assemble(c, p1, torch.ones(1, n_keep, dtype=torch.bool),
                               n_text=5, k=K)
        first_text = float(pos[0, 1 + n_keep])
        assert first_text == K * N_PATCH + 1, (n_keep, first_text)
        assert not bool(isv[0, 1 + n_keep]) and bool(isv[0, 1])
    print(f"✅ 5/9 文本起点恒为 K*256+1 = {K * N_PATCH + 1}，与存活 token 数无关"
          "（120/200/256 三种存活数都试过）")

    st = _State()
    try:
        st.take()
        raise SystemExit("    ✗ 没 set_batch 竟然过了")
    except RuntimeError:
        pass
    set_batch(st, depth=depth, frame_pad_mask=pad, cameras=[cam] * B)
    st.take()
    try:
        st.take()
        raise SystemExit("    ✗ state 被消费两次竟然过了")
    except RuntimeError:
        pass
    # use_cache=False 的生成每步重跑整条前向，投影器要被调 1+max_new_tokens 次
    set_batch(st, depth=depth, frame_pad_mask=pad, cameras=[cam] * B, n_uses=3)
    for _ in range(3):
        st.take()
    try:
        st.take()
        raise SystemExit("    ✗ 超出 n_uses 竟然过了")
    except RuntimeError:
        pass
    print("✅ 6/9 state 是一次性的：没 set_batch、重复消费、超出 n_uses 都硬抛")

    st2 = _State()
    st2.n_layers = 32
    for calls, should_raise in ((0, True), (16, True), (32, False)):
        st2.rope_calls = calls
        try:
            assert_rope_active(st2)
            assert not should_raise, f"{calls} 次调用竟然过了"
        except RuntimeError:
            assert should_raise
    print("✅ 7/9 assert_rope_active：0 次和部分层都拦下，32/32 才放行")

    # 判据必须按臂分开：G0 的正确状态就是 0 次，用同一条断言会把它拦下
    st3 = _State()
    assert_arm_wiring(st3, "G0")            # 0 次，应当放行
    st3.rope_calls = 5
    try:
        assert_arm_wiring(st3, "G0")
        raise SystemExit("    ✗ G0 被挂上 RoPE 竟然过了")
    except RuntimeError:
        pass
    print("   ✅ 7b/9 按臂分派：G0 要求 rope_calls==0，挂上了反而拦下")

    # 错配臂在日志里与它的同池化伙伴长得完全一样（token 数、rope_calls、
    # 序列长都相同 —— 序列长只随语言指令抖动，**与臂无关**）。唯一的区别是
    # PE 轴数，所以判据只能是它。这条要能拦下"M3 安静地退化成 G3"。
    for arm, wrong in (("M3", 3), ("G4", 3), ("M2", 4), ("G3", 4)):
        st4 = _State()
        st4.cfg = WireConfig(arm=arm, K=K,
                             bbox=bbox if arm in ("G4", "M2", "M3") else None)
        st4.rope_calls = st4.n_layers = 32
        st4.pe_axes_seen = st4.cfg.pe_axes
        assert_arm_wiring(st4, arm)          # 正确轴数，放行
        st4.pe_axes_seen = wrong
        try:
            assert_arm_wiring(st4, arm)
            raise SystemExit(f"    ✗ {arm} 的 PE 退化成 {wrong} 轴竟然过了")
        except RuntimeError:
            pass
    print("   ✅ 7c/9 PE 轴数按臂对拍：M3/G4 必须 4 轴、G3/M2 必须 3 轴，"
          "退化成同池化伙伴当场拦下")

    # 回归：cos/sin 的长度必须跟着**真实序列长**走，不能在投影器里写死。
    # 初版把 n_text 写死成 0，真模型上 q 是 2068 而 cos 是 2049，
    # 直接在 apply_rotary_pos_emb 里维度对不上。
    class _Attn:
        pass

    class _Layer:
        def __init__(self):
            self.self_attn = _Attn()
            self.self_attn.rotary_emb = torch.nn.Identity()

    class _FakeLM:
        def __init__(self):
            self.model = type("M", (), {"layers": [_Layer() for _ in range(4)]})()

    class _Fake:
        def __init__(self):
            self.vision_backbone = type("V", (), {"forward": staticmethod(lambda x: x)})()
            self.projector = type("P", (), {"forward": staticmethod(lambda x: x)})()
            self.language_model = _FakeLM()

    fake, cfg = _Fake(), WireConfig(arm="G3", K=K)
    st = wire(fake, cfg)
    n_vis = 256
    st.rope = (torch.rand(1, n_vis, 3) * 16, torch.arange(n_vis).float().unsqueeze(0))
    rope = fake.language_model.model.layers[0].self_attn.rotary_emb
    for n_text in (7, 19, 40):
        total = 1 + n_vis + n_text
        cos, sin = rope(torch.zeros(1, 4, total, cfg.head_dim))
        assert cos.shape[1] == total, (n_text, cos.shape)
    assert st.rope_calls == 3
    unwire(fake, st)
    assert isinstance(fake.language_model.model.layers[0].self_attn.rotary_emb,
                      torch.nn.Identity), "unwire 没还原"
    print("✅ 8/9 cos/sin 跟随真实序列长（7/19/40 三种文本长度），unwire 能还原")

    # 回归：带 KV cache 的自回归解码每步只喂 1 个 token，x.shape[-2]==1，
    # 按它推 n_text 会得到负数。真实位置在 position_ids 里。
    f2, c2 = _Fake(), WireConfig(arm="G3", K=K)
    st2 = wire(f2, c2)
    st2.rope = (torch.rand(1, n_vis, 3) * 16, torch.arange(n_vis).float().unsqueeze(0))
    rp = f2.language_model.model.layers[0].self_attn.rotary_emb
    T = 1 + n_vis + 20
    full, _ = rp(torch.zeros(1, 4, T, c2.head_dim), position_ids=torch.arange(T).unsqueeze(0))
    dec, _ = rp(torch.zeros(1, 4, 1, c2.head_dim), position_ids=torch.tensor([[T - 1]]))
    assert full.shape[1] == T and dec.shape[1] == 1
    assert torch.allclose(dec[0, 0], full[0, T - 1]), "解码位必须与 prefill 同位置一致"
    rp(torch.zeros(1, 4, 1, c2.head_dim), position_ids=torch.tensor([[T]]))  # 新生成位
    unwire(f2, st2)
    print("✅ 9/9 KV cache 解码：prefill / 解码位 / 新生成位三者一致（评测走 generate）")

    print("\n⚠️ 尚未验证、必须在真模型上做的两件事："
          "\n  ① 端到端挂上 7B 后，**第一次 forward 之后调 `assert_rope_active`** —— "
          "\n     挂点错了不会崩，只是 4D RoPE 全程没参与"
          "\n  ② 端到端跑通一次前向，确认 loss 有限、序列长度符合预期")


if __name__ == "__main__":
    _selftest()
