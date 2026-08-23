#!/usr/bin/env bash
# ============================================================================
# 冻结模型的视觉 token 预算消融
#
# 回答的问题: 不做任何训练，把 OpenVLA 的 256 个视觉 token 压到多少，
#            成功率才开始掉。这是投入训练前的可行性门槛。
#
#   bash scripts/run_token_ablation.sh budget            # 阶段一: 找拐点(tome)
#   bash scripts/run_token_ablation.sh method 64         # 阶段二: 拐点附近比算子
#   bash scripts/run_token_ablation.sh diag              # 诊断: 掉点是信息没了还是位置错位
#   bash scripts/run_token_ablation.sh cost              # 只打印耗时/费用估算
#
# ---------------------------------------------------------------------------
# 为什么每个 task 只跑 TRIALS(默认 5) 次而不是 50 次:
#   评测是完全确定的(见 docs/05 3.2)，第 k 次试验的初始状态由 init_files 固定。
#   因此"每 task 前 5 次"是满量 500 局的严格子集 —— 基线不需要重跑，
#   直接从已有的 500 局轨迹里筛 episode_idx < 5 就是同一批初始状态。
#   这把消融成本压到原来的 1/10。
#   代价是每个配置 n=50，只能定位拐点的大致区间；拐点两侧再加量确认。
# ============================================================================
set -euo pipefail

MODE="${1:-budget}"
SUITE="${SUITE:-libero_spatial}"
TRIALS="${TRIALS:-5}"
SEED="${SEED:-7}"

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

declare -A CKPT=(
  [libero_spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [libero_object]="openvla/openvla-7b-finetuned-libero-object"
  [libero_goal]="openvla/openvla-7b-finetuned-libero-goal"
  [libero_10]="openvla/openvla-7b-finetuned-libero-10"
)

# 阶段一扫的预算。256=1.00x 是对照(不压缩，用来验证挂载本身没引入偏差)
BUDGETS=(192 128 96 64 48 32 24 16)
# 阶段二比的算子。tome 是主算子，其余是基线
ALL_METHODS=(random uniform norm avgpool tome)
# 诊断用的信息量档位。expand 保持 256 长度，只把不同值的个数降到该档
DIAG_KEEPS=(128 64 32)

run_one () {
  local keep=$1 method=$2
  echo ""
  echo "############################################################"
  echo "# $SUITE | keep=$keep ($(python -c "print(f'{$keep/256*100:.0f}')")%) | method=$method | trials/task=$TRIALS"
  echo "############################################################"
  python "$EVAL_PY" \
    --model_family openvla \
    --pretrained_checkpoint "${CKPT[$SUITE]}" \
    --task_suite_name "$SUITE" \
    --center_crop True \
    --num_trials_per_task "$TRIALS" \
    --seed "$SEED" \
    --save_video False \
    --token_keep "$keep" \
    --token_method "$method" \
    --traj_dir "$REPO_DIR/results/trajectories" \
    --local_log_dir "$REPO_DIR/results/logs"
}

estimate () {
  local n=$1
  # 满量 500 局 libero_spatial 实测约 2.75h => 单局约 20s
  local eps=$(( n * 10 * TRIALS ))
  python - "$eps" <<'PY'
import sys
eps = int(sys.argv[1])
h = eps * 20 / 3600
print(f"  约 {eps} 局 -> {h:.1f} 小时 (按单局 20s 估)，4090 按 ¥2/h 约 ¥{h*2:.0f}")
PY
}

case "$MODE" in
  cost)
    echo "阶段一(budget): ${#BUDGETS[@]} 个预算 x tome"
    estimate ${#BUDGETS[@]}
    echo "阶段二(method): ${#ALL_METHODS[@]} 个算子 x 1 个预算"
    estimate ${#ALL_METHODS[@]}
    echo "诊断(diag): ${#DIAG_KEEPS[@]} 个 expand 档位 + 1 个 shuffle"
    estimate $(( ${#DIAG_KEEPS[@]} + 1 ))
    echo ""
    echo "基线无需重跑: 从已有 500 局轨迹里筛 episode_idx < $TRIALS 即可(评测确定性)"
    ;;

  budget)
    echo "==> 阶段一: 用 tome 扫 token 预算，定位拐点"
    estimate ${#BUDGETS[@]}
    for keep in "${BUDGETS[@]}"; do
      run_one "$keep" tome
    done
    ;;

  diag)
    # 压缩同时改了两件事: 信息量 和 序列位置。这一组把位置固定住，只降信息量。
    #   expand  : 仍输出 256 个 token，但只有 k 个不同的值
    #   shuffle : 保留全部 256 个 token，只打乱顺序(位置敏感度的上界参照)
    # 若 expand 不掉点而直接压缩掉点 -> 掉的是位置，不是信息
    echo '==> 诊断: 分离「信息损失」与「位置错位」' 
    estimate $(( ${#DIAG_KEEPS[@]} + 1 ))
    for keep in "${DIAG_KEEPS[@]}"; do
      run_one "$keep" expand
    done
    run_one 256 shuffle
    ;;

  method)
    KEEP="${2:?用法: run_token_ablation.sh method <keep>}"
    echo "==> 阶段二: 在 keep=$KEEP 处比较 ${#ALL_METHODS[@]} 个算子"
    estimate ${#ALL_METHODS[@]}
    for m in "${ALL_METHODS[@]}"; do
      run_one "$KEEP" "$m"
    done
    ;;

  *)
    echo "未知模式: $MODE (可选 budget | method | diag | cost)"; exit 1
    ;;
esac

echo ""
echo "==> 完成。汇总:"
echo "    python src/analysis/collect_ablation.py --traj_dir results/trajectories --suite $SUITE --trials $TRIALS"
