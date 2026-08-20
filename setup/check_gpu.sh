#!/usr/bin/env bash
# 切回 GPU 模式后跑这个，10 秒确认这台机器能不能跑 OpenVLA。
set -uo pipefail

# 用 `bash setup/check_gpu.sh` 启动的是非交互 shell，不会读 ~/.bashrc，
# HF_HOME / MUJOCO_GL 等会显示未设置。这里自己补上。
if [ -z "${OPENVLA_ROOT:-}" ] && [ -f "$HOME/.bashrc" ]; then
  set +u; source "$HOME/.bashrc" >/dev/null 2>&1 || true; set -u
fi

echo "=========================================="
echo " OpenVLA 运行环境自检"
echo "=========================================="

command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1 || {
  echo "❌ 没有 GPU（还在无卡模式？）"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
[ "$CC" -ge 80 ] 2>/dev/null \
  && echo "✅ 算力 $(echo $CC | sed 's/./&./1') ≥ 8.0，满足 flash-attn2 + bf16" \
  || { echo "❌ 算力 $(echo $CC | sed 's/./&./1') < 8.0，跑不了（V100/T4 属此列）"; exit 1; }

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM" -ge 16000 ] && echo "✅ 显存 ${VRAM}MB，够 bf16 (需 15GB)" \
                      || echo "⚠️  显存 ${VRAM}MB < 16GB，请加 --load_in_4bit True"

python - <<'PY'
import torch, os
print(f"✅ torch {torch.__version__} | CUDA {torch.version.cuda} | available={torch.cuda.is_available()}")
print(f"{'✅' if torch.cuda.is_bf16_supported() else '❌'} bfloat16 支持")
try:
    import flash_attn; print(f"✅ flash-attn {flash_attn.__version__}")
except ImportError:
    print("❌ flash-attn 未安装 —— 评测会直接加载失败(openvla_utils.py:45 写死)")
for k in ("HF_HOME", "MUJOCO_GL", "OPENVLA_ROOT"):
    v = os.environ.get(k)
    print(f"{'✅' if v else '❌'} {k}={v or '未设置 (source ~/.bashrc)'}")
PY

python - <<'PY'
import os; os.environ.setdefault("MUJOCO_GL","egl")
try:
    from libero.libero import benchmark
    print(f"✅ LIBERO 可启动 ({benchmark.get_benchmark_dict()['libero_spatial']().n_tasks} 个任务)")
except Exception as e:
    print(f"❌ LIBERO: {e}")
PY

echo "=========================================="
