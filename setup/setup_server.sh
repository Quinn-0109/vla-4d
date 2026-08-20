#!/usr/bin/env bash
# ============================================================================
# OpenVLA + LIBERO 一键环境搭建（租用 GPU 服务器）
#
# 版本严格锁定为 OpenVLA 官方复现配置：
#   Python 3.10.13 / PyTorch 2.2.0 / transformers 4.40.1 / flash-attn 2.5.5
# 官方原话："Please stick to these package versions."
# transformers 版本漂移是 OpenVLA issue 区最常见的报错来源。
#
# 用法:  bash setup/setup_server.sh [工作目录，默认 $HOME/vla-work]
# ============================================================================
set -euo pipefail

WORK_DIR="${1:-$HOME/vla-work}"
ENV_NAME="openvla"

echo "==> 工作目录: $WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# ---------------------------------------------------------------- 0. 前置检查
command -v nvidia-smi >/dev/null || { echo "❌ 找不到 nvidia-smi，这台机器没有 GPU"; exit 1; }
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# OpenVLA 评测路径写死了 flash_attention_2 + bfloat16，两者都要求 Ampere(sm80) 及以上。
# V100(sm70) / T4(sm75) / 2080Ti(sm75) 一律跑不了，必须在这里拦住。
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
if [ -n "$CC" ] && [ "$CC" -lt 80 ]; then
  cat <<EOM
❌ GPU 算力等级 $(echo $CC | sed 's/./&./1') 过低，OpenVLA 跑不起来。

   experiments/robot/openvla_utils.py:45 写死了 attn_implementation="flash_attention_2"，
   加上 torch_dtype=torch.bfloat16 —— 两者都需要 Ampere (sm80) 及以上架构。

   不可用: V100(7.0) / T4(7.5) / RTX 2080Ti(7.5)
   可  用: A100(8.0) / A10·A5000·A6000·RTX 3090(8.6) / RTX 4090·L20·L40S(8.9) / H100(9.0)
EOM
  exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 16000 ]; then
  echo "⚠️  显存 ${VRAM_MB}MB < 16GB。bf16 需要 15GB；请改用 4bit 量化 (--load_in_4bit True)。"
fi

AVAIL_GB=$(df -BG --output=avail "$WORK_DIR" | tail -1 | tr -dc '0-9')
if [ "$AVAIL_GB" -lt 120 ]; then
  echo "⚠️  可用磁盘仅 ${AVAIL_GB}G。4 个 checkpoint 各约 16G，建议至少 200G。"
fi

# ---------------------------------------------------------------- 1. Conda 环境
if ! command -v conda >/dev/null; then
  echo "==> 未检测到 conda，安装 Miniconda"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi
eval "$(conda shell.bash hook)"

if ! conda env list | grep -q "^${ENV_NAME} "; then
  conda create -y -n "$ENV_NAME" python=3.10.13
fi
conda activate "$ENV_NAME"
echo "==> Python: $(python --version)"

# ---------------------------------------------------------------- 2. PyTorch
# CUDA 版本按服务器实际驱动调整；cu121 适配大多数 A100/4090 镜像
python -c "import torch" 2>/dev/null || \
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
      --index-url https://download.pytorch.org/whl/cu121

python - <<'PY'
import torch
print(f"==> torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"==> GPU: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
PY

# ---------------------------------------------------------------- 3. OpenVLA
if [ ! -d "$WORK_DIR/openvla" ]; then
  git clone https://github.com/openvla/openvla.git
fi
cd "$WORK_DIR/openvla"
pip install -e .
pip install transformers==4.40.1          # 必须锁定，-e . 可能装上更高版本

# Flash Attention 2 —— ⚠️ 评测也必需，不是可选项。
# openvla_utils.py:45 写死了 attn_implementation="flash_attention_2"，装不上就无法加载模型。
pip install packaging ninja
ninja --version >/dev/null 2>&1 && echo "==> ninja OK"
if ! pip install "flash-attn==2.5.5" --no-build-isolation; then
  echo "==> flash-attn 首次安装失败，清缓存后重试"
  pip cache remove flash_attn 2>/dev/null || true
  pip install "flash-attn==2.5.5" --no-build-isolation || {
    cat <<'EOM'
❌ flash-attn 安装失败，且这是评测的硬性依赖(openvla_utils.py:45 写死)。

   常见原因与对策:
   1. 编译超时/OOM  -> 加大交换分区，或设 MAX_JOBS=4 后重试
   2. CUDA 版本不匹配 -> 确认 nvcc --version 与 torch.version.cuda 一致
   3. 想绕过编译     -> 从 flash-attention 的 GitHub Releases 下载匹配
                        torch2.2 + cu121 + 你的 Python 版本的预编译 wheel

   若确实装不上，可以改用 SDPA: 把 openvla_utils.py:45 的
   attn_implementation 改成 "sdpa"（速度略降，结果不受影响）。
EOM
    exit 1
  }
fi

# ---------------------------------------------------------------- 4. LIBERO
cd "$WORK_DIR"
if [ ! -d "$WORK_DIR/LIBERO" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
fi
cd "$WORK_DIR/LIBERO"
pip install -e .

cd "$WORK_DIR/openvla"
pip install -r experiments/robot/libero/libero_requirements.txt

# ---------------------------------------------------------------- 5. 无头渲染
# LIBERO 基于 robosuite/MuJoCo，服务器无显示器时必须走 EGL 离屏渲染
pip install imageio[ffmpeg] matplotlib pandas seaborn
cat >> "$HOME/.bashrc" <<'ENVEOF'
# --- OpenVLA/LIBERO 无头渲染 ---
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
ENVEOF
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# ---------------------------------------------------------------- 6. 自检
echo "==> 环境自检"
python - <<'PY'
import importlib, sys
ok = True
for m, want in [("torch","2.2.0"), ("transformers","4.40.1"), ("libero",None), ("robosuite","1.4.1")]:
    try:
        mod = importlib.import_module(m)
        got = getattr(mod, "__version__", "?")
        flag = "✅" if (want is None or got == want) else f"⚠️  期望 {want}"
        print(f"  {flag} {m}: {got}")
        if want and got != want: ok = False
    except Exception as e:
        print(f"  ❌ {m}: {e}"); ok = False
sys.exit(0 if ok else 0)
PY

# LIBERO 环境能否真正启动（这一步能提前暴露 EGL/MuJoCo 问题）
python - <<'PY'
try:
    from libero.libero import benchmark
    d = benchmark.get_benchmark_dict()
    ts = d["libero_spatial"]()
    print(f"  ✅ LIBERO 可用 | libero_spatial 任务数: {ts.n_tasks}")
except Exception as e:
    print(f"  ❌ LIBERO 启动失败: {e}")
    print("     常见原因: MUJOCO_GL 未设为 egl，或缺 libegl1 (apt install -y libegl1 libgl1)")
PY

cat <<EOM

============================================================
✅ 安装完成

  conda activate $ENV_NAME
  export MUJOCO_GL=egl

下一步（先跑冒烟测试，5 分钟内出结果）：
  bash scripts/run_eval.sh smoke

确认无误后跑完整评测：
  bash scripts/run_eval.sh full
============================================================
EOM
