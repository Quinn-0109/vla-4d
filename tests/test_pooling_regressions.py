import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "需要 CPU torch 运行张量回归")
class PoolingRegressions(unittest.TestCase):
    def setUp(self):
        import torch
        from pooling.coord_pool import grid_coords, grid_extent
        self.torch = torch
        self.coord = grid_coords(8).unsqueeze(0)
        self.lo, self.hi = grid_extent(8)

    def pool(self, real, partition, framewise=False):
        from pooling.coord_pool import coord_bin_pool
        valid = (self.torch.arange(8) >= 8-real).repeat_interleave(256).unsqueeze(0)
        kw = dict(group_axes=(0,), n_group=(8,)) if framewise else dict(n_t=2)
        return coord_bin_pool(self.coord, self.coord, 256, self.lo, self.hi,
                              valid=valid, partition=partition, **kw)

    def test_legacy_failure_reproduced(self):
        for real in (2, 3, 4):
            with self.subTest(real=real):
                self.assertTrue((self.pool(real,"quantile").assign[0,-256:] < 0).all())

    def test_fixed_preserves_every_valid_patch_and_latest(self):
        for real in range(1,9):
            with self.subTest(real=real):
                out = self.pool(real,"quantile_fixed")
                self.assertTrue((out.assign[0,-real*256:] >= 0).all())
                self.assertTrue((out.assign[0,:-real*256] == -1).all())
                self.assertEqual(out.feat.shape[1],256)

    def test_full_history_unchanged(self):
        a,b = self.pool(8,"quantile"),self.pool(8,"quantile_fixed")
        self.assertTrue(self.torch.equal(a.assign,b.assign))
        self.assertTrue(self.torch.equal(a.feat,b.feat))

    def test_g2_unchanged_at_every_history_length(self):
        for real in range(1,9):
            self.assertTrue(self.torch.equal(self.pool(real,"quantile",True).assign,
                                              self.pool(real,"quantile_fixed",True).assign))

    def test_camera_crop_survives_serialization_and_transfer(self):
        from common.camera import Camera
        torch = self.torch
        cam = Camera(45,224,224,torch.zeros(3),torch.eye(3),True,.9)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"camera.json"
            cam.to_json(path)
            restored=Camera.from_json(path).to("cpu")
            self.assertEqual(restored.crop_scale,.9)
            self.assertTrue(torch.equal(cam.patch_uv(),restored.patch_uv()))

    def test_fixed_partition_reaches_g3_and_m3_wiring(self):
        from common.camera import Camera
        from pooling.wire import WireConfig, _Batch, _pool_and_coords
        torch = self.torch
        cam = Camera(45,224,224,torch.zeros(3),torch.eye(3))
        bt = _Batch(depth=torch.ones(1,8,256),
                    frame_pad_mask=(torch.arange(8) >= 6).unsqueeze(0), cameras=[cam])
        # 最新帧携带唯一非零信号，旧算子丢弃它后输出全零。
        emb = torch.zeros(1,2048,1)
        emb[:,-256:] = 1
        bbox = torch.tensor([[-2.,-2.,-2.],[2.,2.,2.]])
        for arm in ("G3", "M3"):
            # ⚠️ 显式要旧算法。本用例验的就是旧行为，不能靠"默认恰好是它" ——
            #    默认已于 2026-09-15 翻成 quantile_fixed（`docs/05` §13.9）。
            old = _pool_and_coords(emb, WireConfig(arm=arm,K=8,bbox=bbox,
                                                 partition="quantile"), bt)[0]
            new = _pool_and_coords(emb, WireConfig(arm=arm,K=8,bbox=bbox,
                                                 partition="quantile_fixed"), bt)[0]
            self.assertEqual(float(old.abs().sum()), 0)
            self.assertGreater(float(new.abs().sum()), 0)

    def test_frame_cache_has_no_autograd_graph(self):
        from pooling.wire import _State, frame_feats
        torch = self.torch
        state = _State()
        state.orig["vision"] = torch.nn.Linear(4,4)
        out = frame_feats(state,None,torch.randn(1,4,requires_grad=True))
        self.assertFalse(out.requires_grad)
        self.assertIsNone(out.grad_fn)

    def test_alloc_stats_definition_holds(self):
        """
        `alloc_stats` 的四件事**本身就是判据的一部分**，所以这里钉死它的定义。

        钉的是三条恒等式/不变量，而不是某次运行的数值：
          ① Σ_帧 token_share == n_used  —— 分数归属必须正好把用掉的槽分完；
             若某帧的贡献被重复计入或漏计，这条等式**立刻不成立**。
          ② 帧独立池化（G2）下 mixed_frac 必须是 0 —— 它的槽不跨帧，
             这是"跨帧混合程度"这一列有没有量对的最直接检验。
          ③ 补帧期用旧算子（quantile）时，最新帧 keep_rate 必须是 0 ——
             `docs/05` §13.9 的那个缺陷，经由这四件事应当看得见。
        """
        from pooling.coord_pool import coord_bin_pool, grid_coords, grid_extent
        from pooling.wire import alloc_stats
        torch = self.torch
        for real, framewise, partition in [(8, False, "quantile_fixed"),
                                           (2, False, "quantile_fixed"),
                                           (8, True,  "quantile_fixed"),
                                           (2, False, "quantile")]:
            with self.subTest(real=real, framewise=framewise, partition=partition):
                out = self.pool(real, partition, framewise=framewise)
                valid = (torch.arange(8) >= 8-real).repeat_interleave(256).unsqueeze(0)
                st = alloc_stats(out.assign, 8, 256, valid)[0]
                self.assertEqual(st["n_valid"], real)
                # ① 分数归属之和 == 实际用掉的槽数
                self.assertAlmostEqual(sum(st["token_share"]), st["n_used"], places=1)
                self.assertLessEqual(st["newest_slots"], st["n_used"])
                if framewise:
                    # ② 帧独立：一个槽只会来自一帧
                    self.assertEqual(st["mixed_frac"], 0.0)
                if partition == "quantile" and real < 8:
                    # ③ 旧算子在补帧期把最新帧整帧丢掉
                    self.assertEqual(st["keep_rate"][-1], 0.0)
                    self.assertEqual(st["newest_slots"], 0)
                else:
                    self.assertEqual(st["keep_rate"][-1], 1.0)

    def test_alloc_stats_does_not_change_pooling(self):
        """
        ⚠️ 诊断开关**不能改变被测的东西**。`--dump_traj` 会打开
        `state.collect_alloc`，那条路径多传一个 sink dict —— 这里要求
        开与不开的池化输出逐位相同，否则"演示用的那次评测"就不是评测了。
        """
        from common.camera import Camera
        from pooling.wire import WireConfig, _Batch, _pool_and_coords
        torch = self.torch
        cam = Camera(45,224,224,torch.zeros(3),torch.eye(3))
        bt = _Batch(depth=torch.rand(1,8,256)+.5,
                    frame_pad_mask=(torch.arange(8) >= 6).unsqueeze(0), cameras=[cam])
        emb = torch.randn(1,2048,3)
        bbox = torch.tensor([[-2.,-2.,-2.],[2.,2.,2.]])
        for arm in ("G3", "M3", "G2"):
            cfg = WireConfig(arm=arm,K=8,bbox=bbox)
            sink = {}
            a = _pool_and_coords(emb, cfg, bt)
            b = _pool_and_coords(emb, cfg, bt, sink=sink)
            for x, y in zip(a, b):
                self.assertTrue(x is None or torch.equal(x, y))
            self.assertIn("assign", sink)


if __name__ == "__main__":
    unittest.main()
