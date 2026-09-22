from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "run_diagnostic_cases", ROOT / "scripts/run_diagnostic_cases.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class DiagnosticLauncherTests(unittest.TestCase):
    def value(self, cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_g0_is_single_frame_and_all_runs_are_b1(self):
        for arm, k in (("G0", "1"), ("G2", "8"), ("G3", "8")):
            with self.subTest(arm=arm):
                cmd = mod.command(arm, Path("adapter/step30000"),
                                  Path("cases.json"), "diag", "python")
                self.assertEqual(self.value(cmd, "--K"), k)
                self.assertEqual(self.value(cmd, "--eval_batch"), "1")
                self.assertEqual(self.value(cmd, "--dump_traj"), "1")
                self.assertEqual(self.value(cmd, "--case_manifest"), "cases.json")
                self.assertIn("--verify_batch", cmd)

    def test_other_arms_rejected(self):
        with self.assertRaises(ValueError):
            mod.command("G4", Path("x"), Path("y"), "diag")

    def test_registered_adapter_requires_frozen_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "checkpoints.json"
            p.write_text(json.dumps({
                "schema_version": 1,
                "purpose": "fixed_case_diagnostic_checkpoint_identity",
                "arms": {"G3": {"adapter": "/runs/g3/adapter/step30000",
                                  "adapter_weight_bytes": 162015576,
                                  "adapter_sha256": "a" * 64}}}), encoding="utf-8")
            self.assertEqual(mod.registered_adapter("G3", p),
                             "/runs/g3/adapter/step30000")
            with self.assertRaisesRegex(ValueError, "没有 G2"):
                mod.registered_adapter("G2", p)


if __name__ == "__main__":
    unittest.main()
