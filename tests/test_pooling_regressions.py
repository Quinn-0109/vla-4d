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
            old = _pool_and_coords(emb, WireConfig(arm=arm,K=8,bbox=bbox), bt)[0]
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


if __name__ == "__main__":
    unittest.main()
