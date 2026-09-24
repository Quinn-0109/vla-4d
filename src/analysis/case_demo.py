"""固定案例的最小演示（阶段 D2）：把已录制的轨迹与 token 来源画成逐帧画面。

每个方法一行面板，按环境步对齐，并排展示：

1. 当前机器人视角；
2. 窗口里的 K 个历史帧（旧 → 新），补帧标 PAD；
3. 池化前后 token 数，以及各帧分到的 token 份额（按 patch 计数的分数归属）；
4. 输出 token 网格：每格按主要来源帧着色，颜色越淡表示该 token 被越多帧混合；
5. 最新帧所在 token 数与纯度、当前动作与最终成败。

画面标注 “recorded rollout, offline playback”：这是对已记录帧与动作的离线播放，
不是重新运行的闭环。只依赖 Pillow；写 MP4 时另需 imageio 与 ffmpeg，否则写 GIF。
画面文字一律用 ASCII，避免依赖中文字体。
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# 旧 → 新 的有序配色（viridis 取 8 点），最新帧是最亮的黄色
FRAME_COLORS = ["#440154", "#46327e", "#365c8d", "#277f8e",
                "#1fa187", "#4ac16d", "#a0da39", "#fde725"]
PAD_COLOR = "#9e9e9e"
UNUSED = (232, 232, 232)
BG = (255, 255, 255)
INK = (33, 33, 33)
MUTED = (110, 110, 110)
GOOD = (27, 120, 55)
BAD = (178, 34, 34)

PANEL_W, PANEL_H = 1184, 272        # 8 的倍数：H.264 编码不必缩放
VIEW = 224
THUMB = 68
CELL = 12


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "Arial.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf",
                 "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def frame_color(index: int, k: int, is_real: bool = True) -> tuple:
    if not is_real:
        return _rgb(PAD_COLOR)
    if k == 1:
        return _rgb(FRAME_COLORS[-1])
    pos = round(index * (len(FRAME_COLORS) - 1) / (k - 1))
    return _rgb(FRAME_COLORS[pos])


def _blend(color: tuple, strength: float) -> tuple:
    """strength=1 → 原色；0 → 白。用来表示混合程度。"""
    s = max(0.0, min(1.0, strength))
    return tuple(round(c * s + 255 * (1 - s)) for c in color)


class EpisodeView:
    """一个方法、一局的逐步数据，外加关键帧的小缓存。"""

    def __init__(self, run: dict, task: int, episode: int, cache_size: int = 64):
        self.run, self.arm, self.dir = run, run["arm"], run["dir"]
        self.k = int((run["meta"].get("config") or {}).get("K") or 0)
        traj = sorted((r for r in run["trajectory"]
                       if r["task_id"] == task and r["episode"] == episode),
                      key=lambda r: r["env_step"])
        if not traj:
            raise ValueError(f"{self.arm}: {run['dir'].name} 里没有 task {task} ep {episode}")
        alloc = {a["env_step"]: a for a in run["allocation"]
                 if a["task_id"] == task and a["episode"] == episode}
        self.steps = traj
        self.alloc = alloc
        self.by_step = {r["env_step"]: r for r in traj}
        self.k = self.k or len(traj[0]["frame_pad_mask"])
        self.success = int(traj[-1]["success"])
        self.reference = traj[-1].get("g3_reference_success")
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    def __len__(self):
        return len(self.steps)

    def image(self, step: int):
        row = self.by_step.get(step)
        if row is None or not row.get("frame"):
            return None
        if step in self._cache:
            self._cache.move_to_end(step)
            return self._cache[step]
        path = self.dir / row["frame"]
        img = Image.open(path).convert("RGB") if path.is_file() else None
        self._cache[step] = img
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return img


def _placeholder(size: int, text: str, font) -> Image.Image:
    im = Image.new("RGB", (size, size), (200, 200, 200))
    d = ImageDraw.Draw(im)
    d.text((6, size // 2 - 8), text, fill=MUTED, font=font)
    return im


def render_panel(view: EpisodeView, step_index: int, fonts: dict) -> Image.Image:
    """一个方法在第 step_index 个动作步（超出则停在最后一步）的面板。"""
    ended = step_index >= len(view)
    idx = min(step_index, len(view) - 1)
    row = view.steps[idx]
    step = row["env_step"]
    a = view.alloc.get(step)
    k = len(row["frame_pad_mask"])
    im = Image.new("RGB", (PANEL_W, PANEL_H), BG)
    d = ImageDraw.Draw(im)
    f_s, f_m, f_b = fonts["s"], fonts["m"], fonts["b"]

    # ---- 标题
    outcome = "SUCCESS" if view.success else "FAIL"
    ref = {1: "SUCCESS", 0: "FAIL"}.get(view.reference, "n/a")
    d.text((10, 6), view.arm, fill=INK, font=f_b)
    d.text((58, 9), f"step {step:>3d} / {len(view) - 1}   final: {outcome}   "
                    f"(formal b8 reference: {ref})   recorded rollout, offline playback",
           fill=MUTED, font=f_s)
    if ended or (row["success"] and idx == len(view) - 1):
        tag = f"DONE: {outcome} at step {view.steps[-1]['env_step']}"
        d.text((PANEL_W - 250, 6), tag, fill=GOOD if view.success else BAD, font=f_m)

    top = 32
    # ---- 1 当前视角
    cur = view.image(step) or _placeholder(VIEW, "no frame", f_s)
    im.paste(cur.resize((VIEW, VIEW)), (10, top))
    d.text((10, top + VIEW + 2), "current view", fill=MUTED, font=f_s)

    # ---- 2 历史帧（旧 → 新），两行
    hx, hy = 250, top
    cols = 4 if k > 1 else 1
    for j, (src, real) in enumerate(zip(row["source_steps"], row["frame_pad_mask"])):
        cx = hx + (j % cols) * (THUMB + 8)
        cy = hy + (j // cols) * (THUMB + 18)
        thumb = view.image(src)
        thumb = thumb.resize((THUMB, THUMB)) if thumb else _placeholder(THUMB, "-", f_s)
        if not real:
            thumb = Image.blend(thumb, Image.new("RGB", thumb.size, (255, 255, 255)), 0.6)
        im.paste(thumb, (cx, cy))
        d.rectangle([cx - 1, cy - 1, cx + THUMB, cy + THUMB],
                    outline=frame_color(j, k, real), width=3)
        label = "PAD" if not real else ("now" if src == step else f"-{step - src}")
        d.text((cx + 2, cy + THUMB + 2), label, fill=INK if real else MUTED, font=f_s)
    hist_bottom = hy + ((k - 1) // cols + 1) * (THUMB + 18)

    # ---- 3 token 数与各帧份额条
    bx, by, bw = hx, hist_bottom + 4, cols * (THUMB + 8) - 8
    if a is not None:
        used = a["slots_used"]
        d.text((bx, by), f"tokens: {k} x 256 = {k * 256} -> {used}"
                         + ("  (no pooling)" if k == 1 else ""), fill=INK, font=f_s)
        x = bx
        for f in a["per_frame"]:
            w = round(bw * f["slot_weight_sum"] / used) if used else 0
            if w > 0:
                d.rectangle([x, by + 16, x + w - 1, by + 30],
                            fill=frame_color(f["frame_index"], k, f["is_real"]))
                x += w
        d.rectangle([bx, by + 16, bx + bw - 1, by + 30], outline=MUTED)
        d.text((bx, by + 32), "token share per frame (old -> new)", fill=MUTED, font=f_s)

    # ---- 4 输出 token 网格
    gx, gy = 580, top
    budget = 256
    side = 16
    colors = [UNUSED] * budget
    if a is not None:
        for sl in a["slots"]:
            if sl["slot"] >= budget:
                continue
            # 同权时取较新的帧：跨帧均分的 token 不会被画成“看起来来自最旧帧”
            main = max(sl["sources"], key=lambda s_: (s_["weight"], s_["frame_index"]))
            colors[sl["slot"]] = _blend(frame_color(main["frame_index"], k, main["is_real"]),
                                        main["weight"])
    for i, c in enumerate(colors):
        r_, c_ = divmod(i, side)
        x0, y0 = gx + c_ * CELL, gy + r_ * CELL
        d.rectangle([x0, y0, x0 + CELL - 2, y0 + CELL - 2], fill=c)
    d.text((gx, gy + side * CELL + 2), "output tokens: colour = main source frame",
           fill=MUTED, font=f_s)
    d.text((gx, gy + side * CELL + 16), "(ties -> newer); paler = more mixed", fill=MUTED, font=f_s)

    # ---- 5 数值与动作
    tx, ty = 790, top
    lines = []
    if a is not None:
        latest_ws = [sl["latest_frame_weight"] for sl in a["slots"] if sl["latest_frame_weight"] > 0]
        purity = sum(latest_ws) / len(latest_ws) if latest_ws else 0.0
        lat = a["per_frame"][a["latest_real_frame"]]
        lines += [
            (f"real history frames: {a['real_frames']} / {a['frames']}", INK),
            (f"latest frame in {len(latest_ws)} tokens", INK),
            (f"latest-frame purity: {purity:.2f}", INK),
            (f"latest-frame share: {lat['slot_weight_sum']:.1f} tokens", INK),
            (f"latest-only tokens: {a['latest_frame_exclusive_slots']}", INK),
            (f"mixed tokens: {100 * a['mixed_frame_slot_fraction']:.0f}%", INK),
            (f"time span: mean {a['mean_step_span']:.0f}, max {a['max_step_span']} steps", INK),
        ]
    act = row["raw_action"]
    grip = row["executed_action"][6] if len(row["executed_action"]) > 6 else None
    lines += [
        ("", INK),
        (f"move    x {act[0]:+.3f}   y {act[1]:+.3f}   z {act[2]:+.3f}", INK),
        (f"rotate  rx {act[3]:+.3f}  ry {act[4]:+.3f}  rz {act[5]:+.3f}", INK),
        (f"gripper {'close' if grip is not None and grip > 0 else 'open'}", INK),
    ]
    for i, (text, color) in enumerate(lines):
        d.text((tx, ty + i * 17), text, fill=color, font=f_m if i < 1 else f_s)

    # ---- 图例（帧颜色）
    lx, ly = tx, PANEL_H - 26
    d.text((lx, ly), "old", fill=MUTED, font=f_s)
    for j in range(k):
        d.rectangle([lx + 30 + j * 16, ly + 2, lx + 42 + j * 16, ly + 14], fill=frame_color(j, k))
    d.text((lx + 36 + k * 16, ly), "new", fill=MUTED, font=f_s)
    d.line([0, PANEL_H - 1, PANEL_W, PANEL_H - 1], fill=(220, 220, 220), width=1)
    return im


def compose(views: list[EpisodeView], step_index: int, fonts: dict) -> Image.Image:
    panels = [render_panel(v, step_index, fonts) for v in views]
    out = Image.new("RGB", (PANEL_W, PANEL_H * len(panels)), BG)
    for i, p in enumerate(panels):
        out.paste(p, (0, i * PANEL_H))
    return out


def default_fonts() -> dict:
    return {"s": _font(12), "m": _font(14), "b": _font(20)}


def render_video(views: list[EpisodeView], out_path: Path, fps: int = 10, every: int = 1,
                 fmt: str = "auto") -> Path:
    """逐步写视频；各方法按环境步对齐，先结束的停在最后一帧。返回实际写出的路径。"""
    fonts = default_fonts()
    n = max(len(v) for v in views)
    indices = list(range(0, n, max(1, every)))
    if indices[-1] != n - 1:
        indices.append(n - 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "auto":
        try:
            import imageio  # noqa: F401
            import imageio_ffmpeg  # noqa: F401
            fmt = "mp4"
        except ImportError:
            fmt = "gif"
    if fmt == "mp4":
        import imageio
        import numpy as np
        path = out_path.with_suffix(".mp4")
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=7,
                                macro_block_size=8) as w:
            for i in indices:
                w.append_data(np.asarray(compose(views, i, fonts)))
            for _ in range(fps):                    # 结尾停一秒
                w.append_data(np.asarray(compose(views, n - 1, fonts)))
        return path
    path = out_path.with_suffix(".gif")
    frames = [compose(views, i, fonts).convert("P", palette=Image.ADAPTIVE) for i in indices]
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=round(1000 / fps), loop=0)
    return path


def default_sheet_steps(views: list[EpisodeView]) -> list[int]:
    """开局、历史未满、满历史、结束四个时刻（去重）。"""
    n = max(len(v) for v in views)
    k = max(v.k for v in views)
    picks = [0, min(40, n - 1), min((k - 1) * 16 + 8, n - 1), n - 1]
    return sorted(set(picks))


def render_sheet(views: list[EpisodeView], out_path: Path, steps=None, scale: float = 0.6) -> Path:
    """静态拼图：若干时刻 × 各方法，给报告用。"""
    fonts = default_fonts()
    steps = steps if steps else default_sheet_steps(views)
    tiles = [compose(views, s, fonts) for s in steps]
    w, h = tiles[0].size
    sw, sh = round(w * scale), round(h * scale)
    sheet = Image.new("RGB", (sw, sh * len(tiles)), BG)
    for i, t in enumerate(tiles):
        sheet.paste(t.resize((sw, sh), Image.LANCZOS), (0, i * sh))
    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
