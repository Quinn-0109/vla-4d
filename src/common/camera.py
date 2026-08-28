"""
相机反投影 —— 把 patch 深度变成**世界系**的 (x, y, z)。

G4 与 M2 的池化坐标 `(t, x, y, z)` 全靠这一步。在它补上之前，
`coord_pool.metric_coords` 返回的是 `(t, h, w, z)`——x/y 仍是图像网格坐标，
**那样的"G4"只是一个带深度通道的 G3**，而且不会报任何错。

四个必须写死、错了都不会崩的约定：

1. **MuJoCo 相机看向自己的 −z**，+x 向右、+y 向**上**；而图像行号 v 向**下**增长。
   所以 y 要取反。写成 `+` 得到的是上下颠倒的世界坐标，深度对、横向也对，
   只有纵向整体镜像——分箱照跑，G4 学到的是一个镜像世界。
2. **缓存的深度已经被 180° 翻转过**（`depth_diag.replay_depth` 里 `[::-1, ::-1]`，
   与 `get_libero_image` 对 RGB 的翻转对齐）。反投影用的像素坐标必须先**翻回去**，
   否则射线方向整个反了。默认 `flipped=True` 就是为了不让人忘。
3. **内参按 224 算**，不是渲染时的 256。深度在 `to_patch_depth` 里已经 resize 到 224，
   而 fovy 与分辨率无关，所以直接用 224 建内参即可，不要再乘缩放系数（会乘两次）。
4. **patch 中心不是 patch 左上角**。16×16 个 patch、每个 14 像素，
   中心在 `r*14 + 6.5`。差半个 patch 在度量空间里是几毫米到几厘米，
   刚好落在"分箱分不分得开"的量级上。

`python src/common/camera.py` 跑 6 项自检（纯 torch，不需要仿真器）。
真机上还要用 `scripts/dump_camera.py` 与 robosuite 自己的
`transform_from_pixels_to_world` 对齐——**自洽不等于对**，
本项目已经栽过一次"子进程知道答案、父进程把它丢了"。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

GRID = 16
PATCH = 14
IMG = GRID * PATCH          # 224


@dataclass
class Camera:
    """一台固定相机的完整参数。LIBERO 的 agentview 在所有任务里应当一致。"""
    fovy: float                       # 垂直视场角，度
    height: int
    width: int
    pos: torch.Tensor                 # (3,) 相机在世界系的位置
    rot: torch.Tensor                 # (3,3) 相机系 -> 世界系
    flipped: bool = True              # 深度图是否已 180° 翻转（见约定 2）

    # ---------------------------------------------------------------- 构造
    @staticmethod
    def from_json(path: str | Path) -> "Camera":
        d = json.loads(Path(path).read_text())
        return Camera(fovy=float(d["fovy"]), height=int(d["height"]),
                      width=int(d["width"]),
                      pos=torch.tensor(d["pos"], dtype=torch.float64),
                      rot=torch.tensor(d["rot"], dtype=torch.float64).reshape(3, 3),
                      flipped=bool(d.get("flipped", True)))

    def to_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "fovy": self.fovy, "height": self.height, "width": self.width,
            "pos": self.pos.tolist(), "rot": self.rot.reshape(-1).tolist(),
            "flipped": self.flipped}, indent=2))

    # ---------------------------------------------------------------- 内参
    def focal(self) -> float:
        """f = (H/2) / tan(fovy/2)。方形图像下 fx = fy。"""
        return (self.height / 2.0) / math.tan(math.radians(self.fovy) / 2.0)

    def intrinsics(self) -> torch.Tensor:
        f = self.focal()
        return torch.tensor([[f, 0.0, self.width / 2.0],
                             [0.0, f, self.height / 2.0],
                             [0.0, 0.0, 1.0]], dtype=torch.float64)

    # ---------------------------------------------------------------- 反投影
    def backproject(self, uv: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """
        uv: (..., 2) 像素坐标 (u=列, v=行)，**以未翻转的原始图像为准**。
        depth: (...,) 沿光轴的垂直距离，米。
        返回 (..., 3) 世界坐标。
        """
        f = self.focal()
        u, v = uv[..., 0], uv[..., 1]
        x = (u - self.width / 2.0) * depth / f
        y = -(v - self.height / 2.0) * depth / f        # 约定 1：图像 v 向下，相机 +y 向上
        z = -depth                                      # 约定 1：相机看向 −z
        cam = torch.stack([x, y, z], dim=-1)            # (..., 3)
        rot = self.rot.to(cam.dtype)
        return cam @ rot.transpose(-1, -2) + self.pos.to(cam.dtype)

    def project(self, world: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """反投影的逆。返回 (uv, depth)。仅用于自检与排错。"""
        rot = self.rot.to(world.dtype)
        cam = (world - self.pos.to(world.dtype)) @ rot
        f = self.focal()
        depth = -cam[..., 2]
        u = cam[..., 0] * f / depth + self.width / 2.0
        v = -cam[..., 1] * f / depth + self.height / 2.0
        return torch.stack([u, v], dim=-1), depth

    # ---------------------------------------------------------------- patch 级
    def patch_uv(self, device=None) -> torch.Tensor:
        """
        (GRID*GRID, 2) 每个 patch 中心的像素坐标，行优先，**已按 `flipped` 翻回原始朝向**。

        `flipped=True` 时缓存里的 (r, c) 对应原始图像的 (H-1-r*, W-1-c*)——
        180° 翻转就是两个轴都倒过来。
        """
        r = torch.arange(GRID, dtype=torch.float64, device=device) * PATCH + (PATCH - 1) / 2
        c = torch.arange(GRID, dtype=torch.float64, device=device) * PATCH + (PATCH - 1) / 2
        vv, uu = torch.meshgrid(r, c, indexing="ij")
        if self.flipped:
            uu = (self.width - 1) - uu
            vv = (self.height - 1) - vv
        return torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=-1)

    def patch_xyz(self, depth: torch.Tensor) -> torch.Tensor:
        """
        depth: (..., GRID*GRID) patch 级深度（米）。返回 (..., GRID*GRID, 3) 世界坐标。
        """
        uv = self.patch_uv(device=depth.device)                       # (256, 2)
        uv = uv.expand(*depth.shape[:-1], GRID * GRID, 2)
        return self.backproject(uv, depth.to(uv.dtype))


# ---------------------------------------------------------------- 自检
def _rot_x(a: float) -> torch.Tensor:
    c, s = math.cos(a), math.sin(a)
    return torch.tensor([[1.0, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float64)


def _selftest() -> None:
    eye = Camera(fovy=45.0, height=IMG, width=IMG,
                 pos=torch.zeros(3, dtype=torch.float64),
                 rot=torch.eye(3, dtype=torch.float64), flipped=False)

    # 1. 光心射线
    ctr = torch.tensor([[IMG / 2.0, IMG / 2.0]], dtype=torch.float64)
    p = eye.backproject(ctr, torch.tensor([1.5], dtype=torch.float64))
    assert torch.allclose(p, torch.tensor([[0.0, 0.0, -1.5]], dtype=torch.float64), atol=1e-9), p
    print("✅ 1/6 图像中心 + 深度 1.5 m → 相机系 (0,0,−1.5)：看向 −z")

    # 2. 往返
    uv = torch.rand(64, 2, dtype=torch.float64) * IMG
    d = torch.rand(64, dtype=torch.float64) * 2 + 0.3
    w = eye.backproject(uv, d)
    uv2, d2 = eye.project(w)
    assert torch.allclose(uv, uv2, atol=1e-8) and torch.allclose(d, d2, atol=1e-8)
    print("✅ 2/6 project ∘ backproject = 恒等（随机 64 点，误差 < 1e-8）")

    # 3. 位姿不变：换个相机位姿，往返仍成立，且世界点随之刚体变换
    cam2 = Camera(fovy=45.0, height=IMG, width=IMG,
                  pos=torch.tensor([0.6, -1.1, 1.4], dtype=torch.float64),
                  rot=_rot_x(math.radians(-35.0)), flipped=False)
    w2 = cam2.backproject(uv, d)
    uv3, d3 = cam2.project(w2)
    assert torch.allclose(uv, uv3, atol=1e-8) and torch.allclose(d, d3, atol=1e-8)
    expect = w @ cam2.rot.T + cam2.pos
    assert torch.allclose(w2, expect, atol=1e-9)
    print("✅ 3/6 任意位姿下往返成立，且世界点 = R·相机点 + t")

    # 4. 相似三角形：深度翻倍，横向偏移翻倍
    off = torch.tensor([[IMG / 2.0 + 40.0, IMG / 2.0]], dtype=torch.float64)
    a = eye.backproject(off, torch.tensor([1.0], dtype=torch.float64))
    b = eye.backproject(off, torch.tensor([2.0], dtype=torch.float64))
    assert abs(b[0, 0] - 2 * a[0, 0]) < 1e-9 and a[0, 0] > 0
    print("✅ 4/6 深度翻倍 → 横向偏移翻倍（相似三角形），且 u 增大对应 +x")

    # 5. v 向下 = 世界 −y（约定 1 的回归测试；写成 + 号就会挂在这里）
    down = eye.backproject(torch.tensor([[IMG / 2.0, IMG / 2.0 + 40.0]], dtype=torch.float64),
                           torch.tensor([1.0], dtype=torch.float64))
    assert down[0, 1] < 0, "图像下方必须映到相机系 −y；这一条挂了说明 y 的符号写反了"
    print("✅ 5/6 图像下方 → 相机系 −y（符号写反会得到一个上下镜像的世界）")

    # 6. 翻转：flipped 版的 patch (0,0) 必须等于未翻转版的 patch (15,15)
    fl = Camera(fovy=45.0, height=IMG, width=IMG, pos=eye.pos, rot=eye.rot, flipped=True)
    d256 = torch.arange(GRID * GRID, dtype=torch.float64) * 0 + 1.0
    a = fl.patch_xyz(d256).reshape(GRID, GRID, 3)
    b = eye.patch_xyz(d256).reshape(GRID, GRID, 3)
    assert torch.allclose(a[0, 0], b[GRID - 1, GRID - 1], atol=1e-9)
    assert torch.allclose(a[3, 5], b[GRID - 1 - 3, GRID - 1 - 5], atol=1e-9)
    print("✅ 6/6 flipped=True 时 patch (r,c) ≡ 未翻转的 (15−r, 15−c)")

    print("\n⚠️ 以上全是**自洽性**检验：坐标系约定与 MuJoCo 是否一致，"
          "\n   必须在真机上用 scripts/dump_camera.py 与 robosuite 自己的"
          "\n   transform_from_pixels_to_world 对齐才算数。")


if __name__ == "__main__" and __package__ in (None, ""):
    # 直接 `python src/.../x.py` 跑自检时把 src/ 放上 sys.path。
    # 各脚本用的都是 `from common.x import ...` 这种以 src/ 为根的写法
    # （scripts/*.py 里那句 sys.path.insert(..., "src") 就是干这个的），
    # 少了这几行就只有 rope4d 一个模块得换个跑法，是纯粹的绊脚石。
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    _selftest()
