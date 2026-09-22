import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_case_manifest", ROOT / "scripts" / "build_case_manifest.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class CaseManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "episodes.jsonl"
        rows = [
            {"arm": "G3", "task_id": task, "episode": ep, "success": int(ep in successes)}
            for task, successes in ((2, {3, 7}), (4, {1}), (5, {0, 4}))
            for ep in range(8)
        ]
        self.path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    def test_freezes_lowest_success_and_failure(self):
        doc = mod.build(self.path, [2, 4, 5])
        got = [(r["task_id"], r["episode"], r["g3_success"]) for r in doc["cases"]]
        self.assertEqual(got, [(2, 3, 1), (2, 0, 0),
                               (4, 1, 1), (4, 0, 0),
                               (5, 0, 1), (5, 1, 0)])
        self.assertEqual(doc["statistical_use"], "forbidden")
        self.assertEqual(doc["source"]["unique_keys"], 24)

    def test_duplicate_source_key_rejected(self):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"arm": "G3", "task_id": 2, "episode": 0, "success": 0}) + "\n")
        with self.assertRaisesRegex(ValueError, "重复键"):
            mod.build(self.path, [2])

    def test_missing_stratum_rejected(self):
        rows = [{"arm": "G3", "task_id": 2, "episode": ep, "success": 1}
                for ep in range(3)]
        self.path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "failure"):
            mod.build(self.path, [2])


if __name__ == "__main__":
    unittest.main()
