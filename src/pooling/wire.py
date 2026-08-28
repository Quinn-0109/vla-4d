"""
模型侧接线 —— 把 K 帧输入、坐标池化、4D RoPE 挂进 OpenVLA。

**六组对照全部由这一份代码产生**，差别只在 `WireConfig.arm`：

    G0  K=1，不池化，1D RoPE                        （单帧基线，等于原模型）
    G1  K=8，不池化（2048 token），1D RoPE          （上限参考）
    G2  K=8，帧独立池化到 N，PE 用 (t,h,w)
    G3  K=8，跨帧池化，池化坐标 (t,h,w)，PE 也是 (t,h,w)      ← 一致
    G4  K=8，跨帧池化，池化坐标 (t,x,y,z)，PE 也是 (t,x,y,z)  ← 一致，本方案
    M2  K=8，跨帧池化，池化坐标 (t,x,y,z)，PE 却用 (t,h,w)    ← 错配臂

四个挂载点（transformers 4.40.1，见 `_patch_rope` 的版本约定）：

    vision_backbone.forward   (B, K*6, H, W) → (B, K*256, D_vis)
    projector.forward         投影后池化到 N 个 token，并算出 PE 要用的坐标
    每层 attention.rotary_emb 换成我们的 (cos, sin)
    language_model.forward    补上 position_ids —— 官方传的是 None

⚠️ **`set_batch()` 必须在每次 forward 前调用**，把这一批的深度、补帧掩码、
   task_id 交进来（它们不在 `pixel_values` 里，也进不了 `forward` 的签名）。
   忘了调 = 拿上一批的深度算这一批的坐标，**不会报错**。所以 state 是一次性的：
   消费掉就作废，下一次 forward 拿不到就抛异常。

`python src/pooling/wire.py` 跑 8 项自检（桩模型，不加载 7B）。
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
ARMS = ("G0", "G1", "G2", "G3", "G4", "M2")


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
        if self.arm in ("G4", "M2") and self.bbox is None:
            raise ValueError(
                f"{self.arm} 用度量坐标，必须给 bbox（scripts/dump_camera.py 的产物）。"
                "缺了它 metric_extent 无从归一化。")

    @property
    def pools(self) -> bool:
        return self.arm in ("G2", "G3", "G4", "M2")

    @property
    def metric(self) -> bool:
        """池化侧是否用度量坐标。"""
        return self.arm in ("G4", "M2")

    @property
    def pe_axes(self) -> int:
        """PE 侧坐标的轴数。M2 的错配就在这里：池化 4 轴、PE 3 轴。"""
        return 4 if self.arm == "G4" else 3


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
        self.rope: Optional[tuple] = None
        self.rope_calls: int = 0        # 我们的 rotary_emb 被真正调用了几次
        self.orig: dict = {}            # 原始实现，unwire 时还回去
        # 32 层用的是同一条序列，cos/sin 只算一次（build_rope 不便宜）
        self.rope_cache = None
        self.rope_cache_key = None

    def take(self) -> _Batch:
        if self.batch is None:
            raise RuntimeError(
                "forward 前没有调用 set_batch()。深度与相机不在 pixel_values 里，"
                "沿用上一批的值不会报错、loss 照降，但坐标全是错的 —— 所以这里硬抛。")
        b, self.batch = self.batch, None
        self.rope_cache = self.rope_cache_key = None      # 新一批，缓存作废
        return b


# ---------------------------------------------------------------- 各挂载点
def _patch_vision(model, k: int) -> None:
    """(B, K*6, H, W) → (B, K*256, D)。与 `scripts/probe_vram.py` 同一套约定。"""
    orig = model.vision_backbone.forward

    def wrapped(pixel_values, *a, **kw):
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

    if cfg.metric:
        if not bt.cameras:
            raise RuntimeError(f"{cfg.arm} 需要每个样本的相机，set_batch 里没给。")
        # ⚠️ **坐标必须留在 fp32**，不能跟着 emb 转成 bf16。
        #    bf16 只有 8 位尾数，在 ±2 m 量程上相邻可表示值差约 1 cm ——
        #    正好把 G4 要分辨的尺度磨掉，而且**不报任何错**：
        #    分箱照跑、训练照收敛，只是度量坐标退化成了厘米级的量化网格。
        #    特征走 bf16，坐标走 fp32，两者本来就不必同精度。
        pc = torch.cat([metric_coords(bt.depth[i:i + 1].to(dev), k, bt.cameras[i])
                        for i in range(b)], dim=0).float()
        lo, hi = metric_extent(k, cfg.bbox.to(dev), device=dev)
    else:
        pc, (lo, hi) = gc, grid_extent(k, device=dev)

    kw = dict(group_axes=(0,), n_group=(k,)) if cfg.arm == "G2" else dict(n_t=cfg.n_t)
    out = coord_bin_pool(emb, pc, cfg.budget, lo, hi,
                         enforce_n=cfg.enforce_n, valid=valid, **kw)

    # PE 侧坐标：G3/G4 就用池化出来的质心；**M2 的错配在这里** ——
    # 池化侧是度量质心，PE 侧另取箱内 patch 的 (t,h,w) 算术均值（docs/06 §3.0.5 ①）
    if cfg.arm == "M2":
        coord_pe = _grid_centroid(out.assign, gc, cfg.enforce_n or cfg.budget)
    elif cfg.metric:
        coord_pe = out.coord
    else:
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
        lo, hi = (metric_extent(cfg.K, cfg.bbox.to(emb.device), device=emb.device)
                  if cfg.arm == "G4"
                  else grid_extent(cfg.K, device=emb.device))
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
            # 真实序列长只有到这里才知道（x 是 value_states: (B, heads, T, hd)）。
            # 布局 [BOS] [视觉 n_vis] [文本]，所以 n_text = T − 1 − n_vis。
            total = x.shape[-2]
            n_vis = c_norm.shape[1]
            n_text = total - 1 - n_vis
            if n_text < 0:
                raise RuntimeError(
                    f"序列长 {total} 比 1+视觉 {1 + n_vis} 还短 —— 池化输出与"
                    "实际喂进 LLM 的视觉块对不上，先查 projector 那一步。")
            key = (total, n_text)
            if state.rope_cache_key != key:
                p1, c4, isvis = assemble(
                    c_norm, pos1d, torch.ones_like(pos1d, dtype=torch.bool),
                    n_text=n_text, k=cfg.K)
                state.rope_cache = build_rope(p1, c4, isvis, cfg.head_dim, plan)
                state.rope_cache_key = key
            cos, sin = state.rope_cache
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
    state.orig = {
        "vision": model.vision_backbone.forward,
        "projector": model.projector.forward,
        "rotary": [l.self_attn.rotary_emb
                   for l in model.language_model.model.layers],
    }
    _patch_vision(model, cfg.K)
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


def set_batch(state: _State, depth=None, frame_pad_mask=None, cameras=()) -> None:
    state.batch = _Batch(depth=depth, frame_pad_mask=frame_pad_mask,
                         cameras=list(cameras))


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
        cfg = WireConfig(arm=arm, K=K, bbox=bbox if arm in ("G4", "M2") else None)
        bt = _Batch(depth=depth, frame_pad_mask=pad, cameras=[cam] * B)
        return _pool_and_coords(emb, cfg, bt)

    e1, _, _, _ = run("G1")
    assert e1.shape[1] == K * N_PATCH
    for arm in ("G2", "G3", "G4", "M2"):
        e, p, c, _ = run(arm)
        assert e.shape == (B, N_PATCH, D), (arm, e.shape)
        assert p.shape == (B, N_PATCH) and c.shape[-1] in (3, 4)
    print("✅ 1/8 token 数：G1 保持 2048，四个池化臂都压到 256")

    _, _, c4, _ = run("G4")
    _, _, cm, _ = run("M2")
    assert c4.shape[-1] == 4 and cm.shape[-1] == 3
    print("✅ 2/8 M2 的错配：池化侧与 G4 同为度量坐标，PE 侧是 3 轴 (t,h,w)")

    e4, _, _, _ = run("G4")
    e2, _, _, _ = run("M2")
    assert torch.equal(e4, e2), "M2 的池化侧必须与 G4 逐位相同"
    print("✅ 3/8 M2 与 G4 的池化输出逐位相同（只有 PE 侧不同）")

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
    print("✅ 4/8 补帧被屏蔽：assign 全 -1，且不占任何箱")

    for n_keep in (120, 200, 256):
        p1 = torch.arange(n_keep).float().unsqueeze(0)
        c = torch.rand(1, n_keep, 4)
        # 布局是 [BOS] [视觉 n_keep 个] [文本]，所以第一个文本 token 在下标 1+n_keep
        pos, _, isv = assemble(c, p1, torch.ones(1, n_keep, dtype=torch.bool),
                               n_text=5, k=K)
        first_text = float(pos[0, 1 + n_keep])
        assert first_text == K * N_PATCH + 1, (n_keep, first_text)
        assert not bool(isv[0, 1 + n_keep]) and bool(isv[0, 1])
    print(f"✅ 5/8 文本起点恒为 K*256+1 = {K * N_PATCH + 1}，与存活 token 数无关"
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
    print("✅ 6/8 state 是一次性的：没 set_batch 或重复消费都硬抛")

    st2 = _State()
    st2.n_layers = 32
    for calls, should_raise in ((0, True), (16, True), (32, False)):
        st2.rope_calls = calls
        try:
            assert_rope_active(st2)
            assert not should_raise, f"{calls} 次调用竟然过了"
        except RuntimeError:
            assert should_raise
    print("✅ 7/8 assert_rope_active：0 次和部分层都拦下，32/32 才放行")

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
    print("✅ 8/8 cos/sin 跟随真实序列长（7/19/40 三种文本长度），unwire 能还原")

    print("\n⚠️ 尚未验证、必须在真模型上做的两件事："
          "\n  ① 端到端挂上 7B 后，**第一次 forward 之后调 `assert_rope_active`** —— "
          "\n     挂点错了不会崩，只是 4D RoPE 全程没参与"
          "\n  ② 端到端跑通一次前向，确认 loss 有限、序列长度符合预期")


if __name__ == "__main__":
    _selftest()
