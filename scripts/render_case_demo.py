#!/usr/bin/env python
"""阶段 D2 最小演示：把已录制的固定案例画成视频与静态拼图。不需要 GPU 与仿真器。

    python scripts/render_case_demo.py --task 2 --episode 1               # results/cases/ 下全部方法
    python scripts/render_case_demo.py --task 2 --episode 1 --arms G3 G2  # 指定方法与上下顺序
    python scripts/render_case_demo.py DIR_G3 DIR_G2 --task 3 --episode 0 --every 2

输出：
  results/demo/case_task{T}_ep{E}.mp4     逐步视频（缺 imageio/ffmpeg 时改写 GIF；不入库）
  results/figures/case_task{T}_ep{E}.png  开局 / 历史未满 / 满历史 / 结束 四个时刻的拼图（入库）

画面是对已记录帧与动作的**离线播放**，画面上有标注；它不是重新运行的闭环。
资源：三种方法、每种六局、最长 520 步时，读入约需 1.5 GB 内存、半分钟（实测于合成数据）。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analysis.case_report import check_runs, discover, load_run  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("dirs", nargs="*", type=Path, help="案例目录；不给则扫描 --root")
    p.add_argument("--root", type=Path, default=ROOT / "results/cases")
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--arms", nargs="+", help="只画这些方法，并按此顺序从上到下排")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--every", type=int, default=1, help="每隔几步取一帧")
    p.add_argument("--format", choices=("auto", "mp4", "gif"), default="auto")
    p.add_argument("--video_dir", type=Path, default=ROOT / "results/demo")
    p.add_argument("--figure_dir", type=Path, default=ROOT / "results/figures")
    p.add_argument("--sheet_steps", type=int, nargs="+", help="拼图用哪些动作步；默认自动选四个时刻")
    p.add_argument("--no_video", action="store_true")
    args = p.parse_args()

    from analysis.case_demo import EpisodeView, render_sheet, render_video

    dirs = list(args.dirs) or discover(args.root)[0]
    if not dirs:
        print(f"没有找到已发布的案例目录（{args.root}）", file=sys.stderr)
        return 2
    runs = [load_run(d) for d in dirs]
    problems = check_runs(runs)
    if problems:
        print("⚠️ 案例目录验收有问题（仍然渲染，便于排查）：")
        for msg in problems:
            print(f"  - {msg}")
    by_arm = {r["arm"]: r for r in runs}
    order = args.arms or [a for a in ("G3", "G2", "G0") if a in by_arm] + \
        sorted(a for a in by_arm if a not in ("G3", "G2", "G0"))
    missing = [a for a in order if a not in by_arm]
    if missing:
        print(f"找不到这些方法的案例目录: {missing}", file=sys.stderr)
        return 2
    views = [EpisodeView(by_arm[a], args.task, args.episode) for a in order]

    stem = f"case_task{args.task}_ep{args.episode}"
    sheet = render_sheet(views, args.figure_dir / stem, args.sheet_steps)
    print(f"拼图: {sheet}")
    if not args.no_video:
        video = render_video(views, args.video_dir / stem, args.fps, args.every, args.format)
        print(f"视频: {video}")
    for v in views:
        print(f"  {v.arm}: {len(v)} 步，本次 {'成功' if v.success else '失败'}，"
              f"正式 b8 参考 {'成功' if v.reference else '失败'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
