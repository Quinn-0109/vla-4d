#!/usr/bin/env bash
# ============================================================================
# 预下载 OpenVLA 的 LIBERO checkpoint
#
# 💡 这个脚本不需要 GPU —— 在「无卡模式」下跑最省钱。
#    每个 checkpoint 约 15 GB，四个约 60 GB。
#
#   bash scripts/download_checkpoints.sh libero_spatial      # 下单个(推荐先这个)
#   bash scripts/download_checkpoints.sh all                 # 四个全下
#   bash scripts/download_checkpoints.sh --list              # 看已下了哪些/占多大
#   bash scripts/download_checkpoints.sh --rm libero_object  # 删掉腾空间
# ============================================================================
set -euo pipefail

: "${HF_HOME:?请先 source ~/.bashrc（HF_HOME 未设会把 15GB 模型下到系统盘）}"

declare -A REPO=(
  [libero_spatial]="openvla/openvla-7b-finetuned-libero-spatial"
  [libero_object]="openvla/openvla-7b-finetuned-libero-object"
  [libero_goal]="openvla/openvla-7b-finetuned-libero-goal"
  [libero_10]="openvla/openvla-7b-finetuned-libero-10"
)

python -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub

# 默认走镜像；真正管用的是下面 snapshot_download 的断点续传。
# 若直连更快可覆盖: HF_ENDPOINT=https://huggingface.co bash scripts/download_checkpoints.sh ...
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "==> HF_ENDPOINT=$HF_ENDPOINT   (若下载慢可改回 https://huggingface.co)"
echo "==> HF_HOME=$HF_HOME"

case "${1:-}" in
  --list)
    echo "已下载的 checkpoint:"
    for s in "${!REPO[@]}"; do
      d="$HF_HOME/hub/models--${REPO[$s]//\//--}"
      [ -d "$d" ] && printf "  ✅ %-16s %s\n" "$s" "$(du -sh "$d" | cut -f1)" \
                  || printf "  ⬜ %-16s 未下载\n" "$s"
    done
    echo; df -h "$HF_HOME" | tail -1
    exit 0 ;;
  --rm)
    s="${2:?用法: --rm <suite>}"
    d="$HF_HOME/hub/models--${REPO[$s]//\//--}"
    [ -d "$d" ] && { rm -rf "$d"; echo "已删除 $s"; } || echo "$s 本来就不在"
    exit 0 ;;
  all)     SUITES=(libero_spatial libero_object libero_goal libero_10) ;;
  "")      echo "用法: $0 <suite|all|--list|--rm SUITE>"; exit 1 ;;
  *)       SUITES=("$1") ;;
esac

# 用 Python API 而不是命令行: huggingface-hub 1.x 把 huggingface-cli 改名成 hf
# 并废弃了旧命令，而 0.x 只有 huggingface-cli。snapshot_download 可跨版本通用。
for s in "${SUITES[@]}"; do
  [ -n "${REPO[$s]:-}" ] || { echo "❌ 未知 suite: $s"; exit 1; }
  echo ""
  echo "==> 下载 $s  (${REPO[$s]}, 约 15 GB)"
  REPO_ID="${REPO[$s]}" python - <<'HFDL'
import os, sys
from huggingface_hub import snapshot_download
try:
    # 断点续传是默认行为，中断后重跑本脚本即可接着下
    path = snapshot_download(repo_id=os.environ["REPO_ID"], max_workers=4)
    print(f"    -> {path}")
except Exception as e:
    print(f"❌ 下载失败: {type(e).__name__}: {e}")
    print("   链路不稳属常见情况，重跑本脚本即可续传；也可改 HF_ENDPOINT 换源")
    sys.exit(1)
HFDL
done

echo ""
echo "✅ 完成。当前占用:"
bash "$0" --list
