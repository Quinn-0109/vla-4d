#!/usr/bin/env bash
# ============================================================================
# OpenVLA LIBERO 评测启动器
#
#   bash scripts/run_eval.sh smoke              # 冒烟测试: 1 suite × 2 trials/task
#   bash scripts/run_eval.sh full               # 完整: 4 suite × 50 trials/task, seed 7
#   bash scripts/run_eval.sh full 3             # 完整 + 3 个种子(对齐论文协议)
#   bash scripts/run_eval.sh single libero_goal # 单个 suite
#   bash scripts/run_eval.sh single libero_10 2 7 True   # 末位 True = 保存 MP4
# ============================================================================
set -euo pipefail

MODE="${1:-smoke}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export OPENVLA_ROOT="${OPENVLA_ROOT:-$HOME/vla-work/openvla}"
[ -d "$OPENVLA_ROOT" ] || { echo "❌ OPENVLA_ROOT 不存在: $OPENVLA_ROOT"; exit 1; }

# --- 环境前置检查 ---------------------------------------------------------
# tmux/新终端会走 conda init 落回 base，此时 draccus/torch 都不在，
# 直接跑会得到一个看不出根因的 ModuleNotFoundError。在这里拦住。
python - <<'PYCHK' || { echo ""; echo "   修复: conda activate <你的 openvla 环境路径>"; echo "   例如: conda activate /root/autodl-tmp/conda-envs/openvla"; exit 1; }
import sys
import importlib.util as iu
missing = [m for m in ("draccus", "torch", "libero") if iu.find_spec(m) is None]
if missing:
    print(f"\u274c 当前 Python 环境缺少: {', '.join(missing)}")
    print(f"   sys.prefix = {sys.prefix}")
    sys.exit(1)
print(f"环境 OK: {sys.prefix}")
PYCHK

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_PY="$REPO_DIR/scripts/run_libero_eval_traj.py"

# suite -> 官方 checkpoint
declare -A CKPT=(
  [libero_spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [libero_object]="openvla/openvla-7b-finetuned-libero-object"
  [libero_goal]="openvla/openvla-7b-finetuned-libero-goal"
  [libero_10]="openvla/openvla-7b-finetuned-libero-10"
)

run_suite () {
  local suite=$1 trials=$2 seed=$3 video=$4
  echo ""
  echo "############################################################"
  echo "# $suite | trials/task=$trials | seed=$seed"
  echo "############################################################"
  python "$EVAL_PY" \
    --model_family openvla \
    --pretrained_checkpoint "${CKPT[$suite]}" \
    --task_suite_name "$suite" \
    --center_crop True \
    --num_trials_per_task "$trials" \
    --seed "$seed" \
    --save_video "$video" \
    --traj_dir "$REPO_DIR/results/trajectories" \
    --local_log_dir "$REPO_DIR/results/logs"
}

case "$MODE" in
  smoke)
    echo "==> 冒烟测试: 只验证全流程能跑通，不看数字"
    run_suite libero_spatial 2 7 True
    ;;
  full)
    NSEED="${2:-1}"
    # 满量评测关掉视频: 每 suite 500 个 MP4 会吃掉大量磁盘
    for seed in $(seq 7 $((7 + NSEED - 1))); do
      for suite in libero_spatial libero_object libero_goal libero_10; do
        run_suite "$suite" 50 "$seed" False
      done
    done
    ;;
  single)
    # 用法: single <suite> [trials=50] [seed=7] [save_video=False]
    # 满量评测默认不存视频(每 suite 500 个 MP4 太占盘)；要做演示时末位传 True
    SUITE="${2:?用法: run_eval.sh single <suite> [trials] [seed] [save_video]}"
    run_suite "$SUITE" "${3:-50}" "${4:-7}" "${5:-False}"
    ;;
  *)
    echo "未知模式: $MODE (可选 smoke | full | single)"; exit 1
    ;;
esac

echo ""
echo "==> 完成。生成分析图表:"
echo "    python src/analysis/analyze.py --traj_dir results/trajectories"
