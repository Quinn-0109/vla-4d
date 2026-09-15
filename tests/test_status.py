import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "status", Path(__file__).resolve().parents[1] / "scripts/status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.old = Path.cwd()
        os.chdir(self.temp.name)
        self.addCleanup(os.chdir, self.old)
        Path("results/logs").mkdir(parents=True)

    def log(self, suffix, tasks):
        Path(f"results/logs/EVAL-libero_10-G3-K8s16-step30000-b8-N256-nt2{suffix}--sub255.txt").write_text(
            "\n".join(f"task {t} sample: 20/50 = 0.4" for t in tasks) + "\nPARTIAL\n")

    def output(self):
        out = io.StringIO()
        with patch.object(status.subprocess, "run", side_effect=OSError), contextlib.redirect_stdout(out):
            status.eval_status()
        return out.getvalue()

    def test_segments_complete(self):
        self.log("-t0_4", range(4))
        self.log("-t4_10", range(4, 10))
        self.assertIn("合并 200/500", self.output())

    def test_mixed_configuration_rejected(self):
        self.log("-t0_4", range(4))
        self.log("-e242-t4_10", range(4, 10))
        self.assertIn("拒绝自动合并", self.output())

    def test_overlapping_segments_rejected(self):
        self.log("-t0_5", range(5))
        self.log("-t4_10", range(4, 10))
        self.assertIn("拒绝自动汇总", self.output())
