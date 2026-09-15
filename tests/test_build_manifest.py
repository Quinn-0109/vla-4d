import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_paired_manifest import build
from test_provenance import defaults


class ManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = {}
        for arm in ("G3", "G4"):
            path = self.root / f"{arm}.episodes.jsonl"
            rows = [dict(arm=arm, task_id=t, episode=e, success=e % 2)
                    for t in range(10) if t != 1 for e in range(50)]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            tc = dict(defaults("finetune_kframe.py"), arm=arm, seed=7)
            meta = dict(config=dict(defaults("run_eval_kframe.py"), arm=arm),
                        training=dict(config=tc, historical_config_verified=True,
                            dataset=dict(selection="sub255", subset_sha256="a"*64),
                            code=dict(source_sha256="b"*64)),
                        adapter=f"runs/{arm}-sub255/adapter/step30000",
                        adapter_sha256="c"*64, code=dict(source_sha256="d"*64),
                        action_stats={"q01": [0]*7}, initial_states_sha256={"0": "e"*64})
            (self.root / f"{arm}.meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self.files[arm] = [path]

    def change_meta(self, transform):
        p = self.root / "G4.meta.json"
        data = json.loads(p.read_text())
        transform(data)
        p.write_text(json.dumps(data))

    def test_build_complete_and_refuse_overwrite(self):
        output = self.root / "manifest.json"
        result = build(self.files, output)
        self.assertEqual(set(result["arms"]), {"G3", "G4"})
        self.assertTrue(output.is_file())
        with self.assertRaises(FileExistsError):
            build(self.files, output)

    def test_changed_partition_rejected(self):
        self.change_meta(lambda d: d["training"]["config"].update(partition="quantile_fixed"))
        with self.assertRaises(ValueError):
            build(self.files, self.root / "manifest.json")

    def test_unknown_history_rejected(self):
        self.change_meta(lambda d: d["training"].update(historical_config_verified=False))
        with self.assertRaises(ValueError):
            build(self.files, self.root / "manifest.json")


if __name__ == "__main__":
    unittest.main()
