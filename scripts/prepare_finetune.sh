#!/usr/bin/env bash
# ============================================================================
# 阶段 B 的准备工作：训练侧依赖 + LIBERO RLDS 数据集
#
#   bash scripts/prepare_finetune.sh check                  # 只检查，不动手
#   bash scripts/prepare_finetune.sh deps                   # 装/校正训练侧依赖
#   bash scripts/prepare_finetune.sh data                   # 下全部四个 suite
#   bash scripts/prepare_finetune.sh data libero_spatial    # 只下一个
#
# 下载耗时以小时计，建议放 tmux 里跑。它和显存实测不冲突——
# 一个吃网络和磁盘，一个吃 GPU。
# ============================================================================
set -euo pipefail

MODE="${1:-check}"
DATA_ROOT="${DATA_ROOT:-$HOME/autodl-tmp/datasets/modified_libero_rlds}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ⚠️ 版本必须与 openvla 的 pyproject.toml 对齐。
#    peft 尤其重要: 微调要复现官方水平，LoRA 实现的行为差异会直接体现在结果上，
#    而这种差异不会报错，只会让成功率对不上——最难查的那类问题。
declare -A PIN=(
  [peft]="0.11.1"
  [tensorflow_datasets]="4.9.3"
  [transformers]="4.40.1"
  [timm]="0.9.10"
  [tokenizers]="0.19.1"
)

check_env () {
  # ⚠️ 不能只查 torch: base 环境里也有 torch(2.3.0)，查它等于没查。
  #    要查 openvla 环境**独有**的包，否则在 base 里跑完整套检查，
  #    会把 base 的情况当成 openvla 的情况报出来。
  python - <<'PYCHK' || { echo ""; echo "   修复: conda activate /root/autodl-tmp/conda-envs/openvla"; exit 1; }
import sys, importlib.util as iu
missing = [m for m in ("torch", "draccus", "libero") if iu.find_spec(m) is None]
if missing:
    print(f"❌ 这不是 openvla 环境（缺 {', '.join(missing)}）")
    print(f"   sys.prefix = {sys.prefix}")
    sys.exit(1)
print(f"环境: {sys.prefix}")
PYCHK
}

check_disk () {
  echo ""
  echo "--- 磁盘 ---"
  # 目标目录多半还不存在，要一路上溯到第一个存在的父目录再问 df。
  # 直接 df 一个不存在的路径会失败，若此时静默回退到 $HOME，
  # 报出来的就是系统盘而不是数据盘——数字看着合理，结论却是错的。
  local d="$DATA_ROOT"
  while [ ! -d "$d" ] && [ "$d" != "/" ]; do d=$(dirname "$d"); done
  echo "  (数据集将落在 $DATA_ROOT，下面是它所在的文件系统)"
  df -h "$d"
}

check_deps () {
  echo ""
  echo "--- 训练侧依赖 ---"
  for pkg in "${!PIN[@]}"; do
    want="${PIN[$pkg]}"
    got=$(python -c "
import importlib.metadata as m
try: print(m.version('${pkg//_/-}'))
except Exception:
    try: print(m.version('$pkg'))
    except Exception: print('未安装')" 2>/dev/null || echo "未安装")
    if [ "$got" = "$want" ]; then
      printf "  %-22s %-12s ✅\n" "$pkg" "$got"
    else
      printf "  %-22s %-12s ⚠️  应为 %s\n" "$pkg" "$got" "$want"
    fi
  done
  python -c "import dlimp" 2>/dev/null \
    && printf "  %-22s %-12s ✅\n" "dlimp" "已安装" \
    || printf "  %-22s %-12s ⚠️  RLDS 数据管线需要它\n" "dlimp" "未安装"
}

case "$MODE" in
  check)
    check_env; check_disk; check_deps
    echo ""
    echo "数据集目录: $DATA_ROOT"
    [ -d "$DATA_ROOT" ] && du -sh "$DATA_ROOT" 2>/dev/null || echo "  (尚未下载)"
    ;;

  deps)
    check_env
    echo "==> 按 openvla/pyproject.toml 校正版本"
    # 逐个装，避免 pip 的依赖求解把已经对的版本又改掉
    for pkg in "${!PIN[@]}"; do
      pip install -q "${pkg//_/-}==${PIN[$pkg]}"
    done
    pip install -q "dlimp @ git+https://github.com/moojink/dlimp_openvla"
    check_deps
    echo ""
    echo "⚠️ 装完务必重跑一次评测冒烟测试，确认没有把推理链路弄坏:"
    echo "    bash scripts/run_eval.sh smoke"
    ;;

  data)
    check_env
    SUITES="${2:-}"
    mkdir -p "$DATA_ROOT"
    python - "$DATA_ROOT" "$SUITES" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

root, suites = sys.argv[1], sys.argv[2]
repo = "openvla/modified_libero_rlds"
pat = [f"{suites}_no_noops/**"] if suites else None

api = HfApi()
info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
files = info.siblings
if suites:
    files = [f for f in files if f.rfilename.startswith(f"{suites}_no_noops/")]
total = sum(f.size or 0 for f in files)
print(f"仓库: {repo}")
print(f"待下载: {len(files)} 个文件, 约 {total/1e9:.1f} GB")

free = __import__("shutil").disk_usage(root).free
print(f"目标目录可用空间: {free/1e9:.1f} GB")
if total and free < total * 1.15:
    print(f"❌ 空间不足(需要约 {total*1.15/1e9:.1f} GB 含解压余量)，先清理或扩容")
    sys.exit(1)

print("\n开始下载（支持断点续传，中断后重跑本命令即可）...")
snapshot_download(repo, repo_type="dataset", local_dir=root,
                  allow_patterns=pat, max_workers=4)
print(f"\n完成 -> {root}")
PY
    du -sh "$DATA_ROOT"
    ;;

  *)
    echo "未知模式: $MODE (可选 check | deps | data)"; exit 1
    ;;
esac
