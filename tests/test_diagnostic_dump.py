import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.alloc_stats import summarize_assignment
from analysis.diagnostic_dump import (DiagnosticDump, cases_by_task, load_case_manifest,
                                      read_case_dump, sha256)


class AllocationStatsTests(unittest.TestCase):
    def test_cross_frame_weights_and_padding(self):
        # 3 帧 × 2 patch；第 0 帧是补帧，真实帧 1/2 混入 slot 0。
        stats = summarize_assignment([-1, -1, 0, 1, 0, 2],
                                     [False, True, True], [0, 0, 4], 2)
        self.assertEqual(stats["padding_assigned_patches"], 0)
        self.assertEqual(stats["mixed_frame_slots"], 1)
        self.assertAlmostEqual(stats["slots"][0]["latest_frame_weight"], .5)
        self.assertEqual(stats["latest_frame_exclusive_slots"], 1)
        self.assertEqual(stats["max_step_span"], 4)
        self.assertEqual(stats["per_frame"][2]["assigned_patches"], 2)

    def test_frame_independent_slots_have_no_mixing(self):
        stats = summarize_assignment([0, 0, 1, 1], [True, True], [0, 16], 2)
        self.assertEqual(stats["mixed_frame_slots"], 0)
        self.assertEqual(stats["latest_frame_exclusive_slots"], 1)
        self.assertEqual(stats["latest_frame_mean_slot_weight"], .5)

    def test_invalid_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "K"):
            summarize_assignment([0], [True, True], [0, 1], 2)


class DiagnosticDumpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self):
        source = self.root / "source.jsonl"
        source.write_text("{}\n", encoding="utf-8")
        doc = {"schema_version": 1, "purpose": "mechanism_diagnosis_and_demo_only",
               "statistical_use": "forbidden",
               "source": {"path": "source.jsonl", "sha256": sha256(source)},
               "cases": [{"task_id": 4, "episode": 13, "g3_success": 1},
                         {"task_id": 2, "episode": 1, "g3_success": 0}]}
        path = self.root / "manifest.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        return path

    def test_manifest_source_and_grouping(self):
        doc = load_case_manifest(self.manifest(), self.root)
        self.assertEqual(cases_by_task(doc), {2: [1], 4: [13]})

    def test_changed_source_rejected(self):
        path = self.manifest()
        (self.root / "source.jsonl").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_case_manifest(path, self.root)

    def test_atomic_publication_and_gzip(self):
        final = self.root / "run"
        dump = DiagnosticDump(final, {"schema_version": 1})
        dump.write_trajectory({"step": 0})
        dump.write_alloc({"slots_used": 2})
        self.assertFalse(final.exists())
        dump.finalize()
        self.assertTrue(final.is_dir())
        self.assertEqual(json.loads((final / "trajectory.jsonl").read_text())["step"], 0)
        with gzip.open(final / "alloc.jsonl.gz", "rt", encoding="utf-8") as f:
            self.assertEqual(json.loads(f.read())["slots_used"], 2)

    def test_abort_never_publishes(self):
        final = self.root / "run"
        dump = DiagnosticDump(final, {})
        partial = dump.abort()
        self.assertFalse(final.exists())
        self.assertTrue(partial.is_dir())

    def test_reader_checks_schema_and_step_alignment(self):
        final = self.root / "run"
        meta = {"diagnostic_dump": {"schema_version": 1, "statistical_use": "forbidden"}}
        dump = DiagnosticDump(final, meta)
        base = {"schema_version": 1, "arm": "G3", "task_id": 2,
                "episode": 0, "env_step": 0}
        dump.write_trajectory(dict(base, source_steps=[0], frame_pad_mask=[True],
                                   raw_action=[0], executed_action=[0], observation_before={},
                                   observation_after={}, success=0, frame=None))
        dump.write_alloc(dict(base, slots_used=1, per_frame=[], slots=[],
                              latest_frame_mean_slot_weight=1.0,
                              mixed_frame_slot_fraction=0.0, padding_assigned_patches=0))
        dump.finalize()
        doc = read_case_dump(final)
        self.assertEqual(len(doc["trajectory"]), 1)
        (final / "trajectory.jsonl").write_text(json.dumps(base) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "缺字段"):
            read_case_dump(final)


if __name__ == "__main__":
    unittest.main()
