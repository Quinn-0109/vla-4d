import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "需要 CPU torch 运行张量回归")
class G3QTests(unittest.TestCase):
    """
    G3Q 相对 G3 只改“最新帧是否被旧帧稀释”。这里钉住这一点在每个历史长度上都成立，
    以及它给最新帧的待遇与 G2 逐位相同——否则 G3Q 与 G3、G2 的对照就不只差一件事。
    """

    def setUp(self):
        import torch
        from pooling.wire import N_PATCH, WireConfig, _Batch, _pool_and_coords
        self.torch, self.N, self.Cfg, self.Batch, self.pool = (
            torch, N_PATCH, WireConfig, _Batch, _pool_and_coords)
        torch.manual_seed(0)
        self.emb = torch.randn(1, 8 * N_PATCH, 6)

    def run_arm(self, arm, real, emb=None):
        torch = self.torch
        mask = (torch.arange(8) >= 8 - real).unsqueeze(0)
        sink = {}
        feat, pos1d, coord, m = self.pool(self.emb if emb is None else emb,
                                          self.Cfg(arm=arm, K=8),
                                          self.Batch(frame_pad_mask=mask), sink=sink)
        return feat, coord, m, sink["assign"][0], mask[0]

    def frame_slots(self, assign, frame):
        a = assign[frame * self.N:(frame + 1) * self.N]
        return set(a[a >= 0].tolist())

    def test_latest_frame_is_pure_and_matches_g2_bitwise(self):
        for real in range(1, 9):
            with self.subTest(real=real):
                fq, _, _, aq, _ = self.run_arm("G3Q", real)
                f2, _, _, a2, _ = self.run_arm("G2", real)
                q_slots, g2_slots = self.frame_slots(aq, 7), self.frame_slots(a2, 7)
                self.assertEqual(len(q_slots), len(g2_slots))
                # 纯度：最新帧的槽里没有任何旧帧 patch
                for f in range(7):
                    self.assertFalse(q_slots & self.frame_slots(aq, f))
                # 与 G2 的最新帧 token 逐位相同（同一个算子、同一份预算）
                q_idx, g_idx = sorted(q_slots), sorted(g2_slots)
                self.assertTrue(self.torch.equal(fq[0, q_idx], f2[0, g_idx]))

    def test_history_still_mixes_like_g3(self):
        _, _, _, a, _ = self.run_arm("G3Q", 8)
        hist = [self.frame_slots(a, f) for f in range(7)]
        shared = set.union(*[hist[i] & hist[j] for i in range(7) for j in range(i + 1, 7)])
        self.assertGreater(len(shared), 0)            # 历史帧之间仍有跨帧混合

    def test_every_valid_patch_assigned_once_padding_never(self):
        for real in range(1, 9):
            with self.subTest(real=real):
                _, coord, m, a, mask = self.run_arm("G3Q", real)
                valid = mask.repeat_interleave(self.N)
                self.assertTrue(bool((a[valid] >= 0).all()))
                self.assertTrue(bool((a[~valid] == -1).all()))
                used = int(m.sum())
                self.assertLessEqual(used, 256)
                self.assertTrue(bool(m[0, :used].all()) and not bool(m[0, used:].any()))
                self.assertEqual(set(a[a >= 0].tolist()), set(range(used)))
                t = coord[0, :used, 0]
                self.assertTrue(bool((t[1:] >= t[:-1] - 1e-6).all()))   # 仍按 t̄ 升序

    def test_batch_with_different_history_lengths(self):
        torch = self.torch
        emb = torch.randn(2, 8 * self.N, 6)
        mask = torch.stack([torch.arange(8) >= 0, torch.arange(8) >= 5])   # 8 帧与 3 帧
        sink = {}
        feat, _, _, m = self.pool(emb, self.Cfg(arm="G3Q", K=8),
                                  self.Batch(frame_pad_mask=mask), sink=sink)
        for i, real in ((0, 8), (1, 3)):
            f1, _, m1, _, _ = self.run_arm("G3Q", real, emb[i:i + 1])
            self.assertTrue(torch.equal(feat[i], f1[0]) and torch.equal(m[i], m1[0]))

    def test_quota_rule_and_config_guards(self):
        from pooling.wire import current_quota
        self.assertEqual([current_quota(r, 256) for r in (1, 2, 8)], [256, 128, 32])
        cfg = self.Cfg(arm="G3Q", K=8)
        self.assertTrue(cfg.pools and not cfg.metric and cfg.pe_axes == 3)
        with self.assertRaises(ValueError):
            self.Cfg(arm="G3Q", K=8, enforce_n=242)

    def test_g3q_accepted_by_launch_validation(self):
        from common.provenance import validate_config
        sys.path.insert(0, str(ROOT / "tests"))
        from test_provenance import defaults
        validate_config(dict(defaults("run_eval_kframe.py"), arm="G3Q"), training=False)
        validate_config(dict(defaults("finetune_kframe.py"), arm="G3Q"), training=True)


if __name__ == "__main__":
    unittest.main()
