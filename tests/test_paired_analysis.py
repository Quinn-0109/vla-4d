"""无需 GPU 的结果来源与完整性回归测试。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts/paired_analysis.py"
spec = importlib.util.spec_from_file_location("paired", MODULE)
paired = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paired)


class PairingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        template = json.loads((MODULE.parents[1] / "docs/paired-manifest.example.json").read_text())
        self.doc = {"arms": {a: template["arms"][a] for a in ("G3", "G4")}}
        for arm, entry in self.doc["arms"].items():
            entry["checkpoint"] = f"runs/{arm}-sub255/adapter/step30000"
            entry["protocol"].update(dataset_sha256="a" * 64, training_seed=7,
                                      eval_code_commit="b" * 40)
            rows = [dict(arm=arm, task_id=t, episode=e, success=e % 2)
                    for t in range(10) if t != 1 for e in range(50)]
            self.write_rows(arm, rows)
        self.manifest = self.root / "manifest.json"

    def write_rows(self, arm, rows):
        raw = "\n".join(json.dumps(r) for r in rows).encode()
        path = self.root / f"{arm}.episodes.jsonl"
        path.write_bytes(raw)
        self.doc["arms"][arm]["files"] = [dict(path=path.name, sha256=hashlib.sha256(raw).hexdigest())]

    def load(self):
        self.manifest.write_text(json.dumps(self.doc))
        return paired.load(self.manifest)

    def rows(self):
        return [json.loads(s) for s in (self.root / "G3.episodes.jsonl").read_text().splitlines()]

    def test_complete_ignores_unlisted_full_run(self):
        (self.root / "EVAL-G3--fullG3.episodes.jsonl").write_text("invalid")
        self.assertEqual(len(self.load()["G3"]), 450)

    def test_missing_episode_rejected(self):
        self.write_rows("G3", self.rows()[:-1])
        with self.assertRaisesRegex(ValueError, "缺 1 局"):
            self.load()

    def test_duplicate_rejected(self):
        rows = self.rows()
        self.write_rows("G3", rows + rows[:1])
        with self.assertRaisesRegex(ValueError, "重复配对键"):
            self.load()

    def test_protocol_mismatch_rejected(self):
        self.doc["arms"]["G4"]["protocol"]["eval_batch"] = 1
        with self.assertRaisesRegex(ValueError, "协议不一致"):
            self.load()

    def test_full_run_rejected(self):
        self.doc["arms"]["G3"]["checkpoint"] = "runs/G3+fulldata/adapter/step30000"
        with self.assertRaisesRegex(ValueError, "不属于"):
            self.load()

    def test_changed_file_rejected(self):
        (self.root / "G3.episodes.jsonl").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.load()

    def test_wrong_arm_rejected(self):
        rows = self.rows()
        rows[0]["arm"] = "M2"
        self.write_rows("G3", rows)
        with self.assertRaisesRegex(ValueError, "实验臂不匹配"):
            self.load()

    def test_balanced_discordances(self):
        base = {0: 1, 1: 0}
        treat = {0: 0, 1: 1}
        self.assertEqual(paired.mcnemar(base, treat, [0, 1])[2:], (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
