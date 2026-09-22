import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


@unittest.skipUnless(HAS_NUMPY, "需要 numpy")
class TrajDump(unittest.TestCase):
    """
    演示素材的落盘/读回。⚠️ 这条路径**在真机上一次只跑几局**，
    而写错了不会让评测报错 —— 评测照样给出成功率，只是演示素材是坏的。
    所以格式约定在这里过一遍，而不是等到答辩前回放时才发现。
    """

    def head(self, **kw):
        return dict({"run_id": "EVAL-x", "arm": "G3", "task_id": 2, "episode": 0,
                     "task": "put the mug on the plate", "success": 1,
                     "pools": True, "K": 8, "budget": 256}, **kw)

    def steps(self, n=3):
        return [{"t": 10 + i, "action": [0.1] * 7, "n_valid": min(i + 1, 8),
                 "alloc": {"n_used": 242, "keep_rate": [1.0] * 8}} for i in range(n)]

    def test_round_trip_with_and_without_rgb(self):
        import numpy as np
        from common.traj import read_episode, write_episode
        for with_rgb in (True, False):
            with self.subTest(rgb=with_rgb), tempfile.TemporaryDirectory() as tmp:
                steps = self.steps()
                rgb = [np.zeros((4, 4, 3), np.uint8) + i for i in range(3)] if with_rgb else None
                path = write_episode(Path(tmp), self.head(), steps, rgb)
                head, got, back = read_episode(path)
                self.assertEqual(head["run_id"], "EVAL-x")
                self.assertEqual(head["steps"], 3)
                self.assertEqual(got, steps)
                if with_rgb:
                    self.assertEqual(back.shape, (3, 4, 4, 3))
                    self.assertTrue((back[2] == 2).all())
                else:
                    self.assertIsNone(back)

    def test_misaligned_rgb_refused(self):
        import numpy as np
        from common.traj import write_episode
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "对不上"):
                write_episode(Path(tmp), self.head(), self.steps(3),
                              [np.zeros((4, 4, 3), np.uint8)] * 2)

    def test_truncated_and_headerless_refused(self):
        from common.traj import read_episode, write_episode
        with tempfile.TemporaryDirectory() as tmp:
            path = write_episode(Path(tmp), self.head(), self.steps(3))
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "步"):
                read_episode(path)          # 评测被 kill 的那种残局
            path.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "表头"):
                read_episode(path)          # 不知道是哪次测量的，不许用


if __name__ == "__main__":
    unittest.main()
