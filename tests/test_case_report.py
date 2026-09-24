import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

from analysis.case_report import (check_runs, discover, episode_table,  # noqa: E402
                                  frame_profile, history_table, load_run, write_report)
from case_fixtures import DEFAULT_CASES, make_case_dir  # noqa: E402


class CaseReportTests(unittest.TestCase):
    """阶段 C 验收与 D1 表。合成目录逐字段模仿评测脚本的导出（见 case_fixtures）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def runs(self, arms=("G0", "G2", "G3"), **kw):
        # 满历史要从第 (K-1)*stride = 112 步开始，局长必须超过它，否则满历史的表是空的
        kw.setdefault("outcomes", {(2, 0): (125, 1), (2, 1): (140, 0)})
        return [load_run(make_case_dir(self.root, a, frames=False, **kw)) for a in arms]

    def test_clean_runs_pass_acceptance(self):
        self.assertEqual(check_runs(self.runs()), [])

    def test_purity_separates_dilution_from_allocation(self):
        """满历史时 G2 与 G3 给最新帧的份额相同（32 token），区别只在稀释：
        G2 纯度 1.0，G3 每个含最新帧的 token 里它只占 1/4。全槽平均权重把两者算成一样，
        这正是表里以纯度为主的原因。"""
        rows = {(r["arm"], r["real_frames"]): r for r in history_table(self.runs())
                if r["observed_outcome"] == "failure"}
        g2, g3 = rows[("G2", 8)], rows[("G3", 8)]
        self.assertAlmostEqual(g2["latest_share_mean"], g3["latest_share_mean"])
        self.assertAlmostEqual(g2["latest_weight_mean"], g3["latest_weight_mean"])
        self.assertEqual(g2["latest_purity_mean"], 1.0)
        self.assertAlmostEqual(g3["latest_purity_mean"], 0.25)
        self.assertAlmostEqual(g3["latest_slots_mean"] * g3["latest_purity_mean"],
                               g3["latest_share_mean"])

    def test_episode_table_structure(self):
        eps = {(e["arm"], e["episode"]): e for e in episode_table(self.runs())}
        self.assertEqual(len(eps), 6)
        g0 = eps[("G0", 0)]
        self.assertEqual((g0["latest_purity_mean"], g0["mixed_fraction_mean"]), (1.0, 0.0))
        self.assertEqual(eps[("G2", 1)]["mixed_fraction_mean"], 0.0)
        self.assertGreater(eps[("G3", 1)]["mixed_fraction_mean"], 0.0)
        self.assertEqual(eps[("G3", 0)]["observed_success"], 1)
        self.assertTrue(all(e["valid_dropped_patches_total"] == 0 for e in eps.values()))

    def test_frame_profile_sums_to_used_tokens(self):
        prof = [p for p in frame_profile(self.runs()) if p["arm"] == "G2"]
        self.assertEqual(sorted(p["offset_from_latest"] for p in prof), list(range(8)))
        self.assertAlmostEqual(sum(p["token_share_mean"] for p in prof), 256.0)

    def test_extra_and_missing_episodes_rejected(self):
        run = load_run(make_case_dir(self.root, "G3", frames=False, extra_episode=(9, 9)))
        self.assertTrue(any("manifest 之外" in p for p in check_runs([run])))
        run = load_run(make_case_dir(self.root, "G2", frames=False,
                                     outcomes={(2, 0): (20, 1)}))
        self.assertTrue(any("缺少 manifest 中的局" in p for p in check_runs([run])))

    def test_mismatched_manifests_and_duplicate_arms_rejected(self):
        a = load_run(make_case_dir(self.root, "G3", frames=False))
        b = load_run(make_case_dir(self.root, "G2", frames=False, manifest_sha="x" * 64))
        self.assertTrue(any("manifest 不同" in p for p in check_runs([a, b])))
        c = load_run(make_case_dir(self.root, "G3", frames=False, name="another-G3"))
        self.assertTrue(any("多个案例目录" in p for p in check_runs([a, c])))

    def test_non_finite_values_rejected(self):
        run = load_run(make_case_dir(self.root, "G3", frames=False, nan_at=(2, 1, 5)))
        self.assertTrue(any("非有限" in p for p in check_runs([run])))

    def test_reference_disagreement_is_recorded_not_rejected(self):
        run = load_run(make_case_dir(self.root, "G3", frames=False,
                                     outcomes={(2, 0): (30, 0), (2, 1): (30, 0)}))
        self.assertEqual(check_runs([run]), [])
        ep0 = [e for e in episode_table([run]) if e["episode"] == 0][0]
        self.assertFalse(ep0["matches_reference"])
        paths = write_report([run], [], [], self.root / "out")
        self.assertIn("不一致", paths["markdown"].read_text(encoding="utf-8"))

    def test_partial_dirs_reported_not_read(self):
        make_case_dir(self.root, "G3", frames=False)
        (self.root / "EVAL-half.partial").mkdir()
        dirs, partial = discover(self.root)
        self.assertEqual([d.name for d in partial], ["EVAL-half.partial"])
        self.assertTrue(all(not d.name.endswith(".partial") for d in dirs))

    def test_report_states_boundary(self):
        paths = write_report(self.runs(), [], [], self.root / "out")
        md = paths["markdown"].read_text(encoding="utf-8")
        self.assertIn("不做显著性检验", md)
        self.assertIn("定义上的选择", md)
        self.assertTrue(paths["episodes"].read_text(encoding="utf-8").startswith("arm,"))


@unittest.skipUnless(HAS_PIL and HAS_NUMPY, "需要 Pillow 与 numpy 生成合成帧和画面")
class CaseDemoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_missing_frame_rejected_by_acceptance(self):
        d = make_case_dir(self.root, "G3", outcomes={(2, 0): (10, 1), (2, 1): (10, 0)})
        next((d / "frames").rglob("*.jpg")).unlink()
        self.assertTrue(any("关键帧" in p for p in check_runs([load_run(d)])))

    def test_render_panels_video_and_sheet(self):
        from analysis.case_demo import (PANEL_H, PANEL_W, EpisodeView, compose,
                                        default_fonts, render_sheet, render_video)
        outcomes = {(2, 0): (20, 1), (2, 1): (30, 0)}
        runs = {a: load_run(make_case_dir(self.root, a, outcomes=outcomes)) for a in ("G3", "G0")}
        views = [EpisodeView(runs["G3"], 2, 1), EpisodeView(runs["G0"], 2, 1)]
        img = compose(views, 5, default_fonts())
        self.assertEqual(img.size, (PANEL_W, 2 * PANEL_H))
        self.assertEqual(PANEL_W % 8, 0)                    # H.264 不需要缩放
        # 超过某方法的最后一步时停在最后一帧而不是报错
        compose([EpisodeView(runs["G3"], 2, 0), views[0]], 29, default_fonts())
        gif = render_video(views, self.root / "demo" / "case", fps=5, every=5, fmt="gif")
        self.assertEqual(gif.suffix, ".gif")
        from PIL import Image
        with Image.open(gif) as g:
            self.assertEqual(g.n_frames, 7)                # 0,5,...,25 加最后一步 29
        sheet = render_sheet(views, self.root / "fig" / "case")
        self.assertTrue(sheet.is_file())
        with self.assertRaisesRegex(ValueError, "没有 task"):
            EpisodeView(runs["G3"], 7, 0)

    def test_cli_end_to_end(self):
        outcomes = {(2, 0): (12, 1), (2, 1): (12, 0)}
        cases = self.root / "cases"
        for a in ("G0", "G2", "G3"):
            make_case_dir(cases, a, outcomes=outcomes)
        r = subprocess.run([sys.executable, str(ROOT / "scripts/diagnose_cases.py"),
                            "--root", str(cases), "--out", str(self.root / "tables")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.root / "tables/diagnostic_cases.md").is_file())
        r = subprocess.run([sys.executable, str(ROOT / "scripts/render_case_demo.py"),
                            "--root", str(cases), "--task", "2", "--episode", "1",
                            "--format", "gif", "--every", "4",
                            "--video_dir", str(self.root / "demo"),
                            "--figure_dir", str(self.root / "fig")],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertTrue((self.root / "fig/case_task2_ep1.png").is_file())
        self.assertTrue((self.root / "demo/case_task2_ep1.gif").is_file())


if __name__ == "__main__":
    unittest.main()
