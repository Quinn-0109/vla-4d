import ast
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import tempfile
from typing import Optional
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common.provenance import (check_config, config_dict, eval_paths, prepare_training,
                               validate_config)
from common.checkpoints import checkpoint_directory


def defaults(script):
    tree = ast.parse((ROOT / "scripts" / script).read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    ns = dict(dataclass=dataclass, Optional=Optional, Path=Path)
    exec(compile(ast.Module(body=[node], type_ignores=[]), script, "exec"), ns)
    return config_dict(ns["Config"]())


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_actual_defaults_validate(self):
        validate_config(defaults("finetune_kframe.py"), training=True)
        validate_config(defaults("run_eval_kframe.py"), training=False)

    def test_eval_bounds_before_loading(self):
        c = defaults("run_eval_kframe.py")
        for bad in ({"num_trials_per_task": 51}, {"eval_batch": 0},
                    {"start_task": 5,"end_task": 5}, {"arm": "G1"},
                    {"dump_traj": 6}, {"dump_traj": -1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(dict(c, **bad), training=False)

    def test_diagnostic_dump_requires_frozen_manifest(self):
        c = defaults("run_eval_kframe.py")
        with self.assertRaisesRegex(ValueError, "case_manifest"):
            validate_config(dict(c, dump_traj=True), training=False)
        with self.assertRaisesRegex(ValueError, "一起使用"):
            validate_config(dict(c, case_manifest="cases.json"), training=False)
        validate_config(dict(c, dump_traj=True, case_manifest="cases.json", eval_batch=1), training=False)
        with self.assertRaisesRegex(ValueError, "eval_batch=1"):
            validate_config(dict(c, dump_traj=True, case_manifest="cases.json"), training=False)
        with self.assertRaisesRegex(ValueError, "task 范围"):
            validate_config(dict(c, dump_traj=True, case_manifest="cases.json",
                                 start_task=2, end_task=6, eval_batch=1), training=False)

    def test_resume_rejects_changed_data_selection(self):
        c = defaults("finetune_kframe.py")
        with self.assertRaisesRegex(ValueError, "no_subset"):
            check_config(dict(c, arm="G3", no_subset=True), dict(c, arm="G3"))

    def test_same_step_different_adapter_no_collision(self):
        c = defaults("run_eval_kframe.py")
        c["local_log_dir"] = str(self.root / "logs")
        a, _ = eval_paths(dict(c, adapter="runs/a/adapter/step30000"), self.root)
        b, _ = eval_paths(dict(c, adapter="runs/b/adapter/step30000"), self.root)
        self.assertNotEqual(a, b)

    def test_partial_results_protected(self):
        c = defaults("run_eval_kframe.py")
        c["local_log_dir"] = str(self.root)
        _, paths = eval_paths(c, self.root)
        paths["episodes"].write_text("partial")
        with self.assertRaises(ValueError):
            eval_paths(c, self.root)

    def test_segments_share_identity(self):
        c = defaults("run_eval_kframe.py")
        c["local_log_dir"] = str(self.root)
        a, _ = eval_paths(dict(c,start_task=0,end_task=4),self.root)
        b, _ = eval_paths(dict(c,start_task=4,end_task=10),self.root)
        self.assertEqual(a.split("-t0_4")[0], b.split("-t4_10")[0])

    def test_dump_flags_share_identity(self):
        """
        `--dump_traj` 只是把动作与分配统计抄一份，**不改变被测量的成功率**，
        所以开与不开必须是同一个 run_id —— 否则演示素材那次会被当成另一次
        独立测量，而它并不是。反过来，凡是改变测量的开关（这里用中心裁）
        必须换身份。
        """
        c = defaults("run_eval_kframe.py")
        c["local_log_dir"] = str(self.root / "dump")
        base, _ = eval_paths(c, self.root)
        same, _ = eval_paths(dict(c, dump_traj=2, dump_rgb=False,
                                  dump_dir="results/traj/demo"), self.root)
        self.assertEqual(base, same)
        diff, _ = eval_paths(dict(c, center_crop=not c["center_crop"]), self.root)
        self.assertNotEqual(base, diff)

    def test_training_record_and_reentry_guard(self):
        c = defaults("finetune_kframe.py")
        run = self.root / "run"
        rec = prepare_training(c, run, self.root, None)
        self.assertEqual(rec["dataset"]["selection"], "full")
        (run / "metrics.jsonl").write_text("{}")
        with self.assertRaisesRegex(ValueError, "已有训练产物"):
            prepare_training(c, run, self.root, None)

    def test_checkpoint_failure_never_publishes(self):
        dest = self.root / "step1"
        with self.assertRaises(RuntimeError):
            with checkpoint_directory(dest) as tmp:
                (tmp / "adapter_config.json").write_text("{}")
                raise RuntimeError("interrupted")
        self.assertFalse(dest.exists())

    def test_unknown_legacy_history_stays_unknown_after_resume(self):
        c = defaults("finetune_kframe.py")
        legacy = self.root / "legacy" / "adapter" / "step2500"
        first = self.root / "first"
        record = prepare_training(c, first, self.root, legacy)
        self.assertFalse(record["historical_config_verified"])
        again = prepare_training(c, self.root / "second", self.root,
                                 first / "adapter" / "step5000")
        self.assertFalse(again["historical_config_verified"])

    def test_checkpoint_success_then_no_overwrite(self):
        dest = self.root / "step1"
        with checkpoint_directory(dest) as tmp:
            for name in ("adapter_config.json","adapter_model.safetensors","trainer_state.pt"):
                (tmp / name).write_text("test")
        self.assertTrue(dest.is_dir())
        with self.assertRaises(FileExistsError):
            with checkpoint_directory(dest):
                pass


if __name__ == "__main__":
    unittest.main()
