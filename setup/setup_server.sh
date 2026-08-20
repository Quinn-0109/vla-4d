#!/usr/bin/env bash
# ============================================================================
# OpenVLA + LIBERO 一键环境搭建（租用 GPU 服务器）
#
# 版本严格锁定为 OpenVLA 官方复现配置：
#   Python 3.10.13 / PyTorch 2.2.0 / transformers 4.40.1 / flash-attn 2.5.5
# 官方原话："Please stick to these package versions."
#
# 用法:
#   bash setup/setup_server.sh                 # 自动探测数据盘
#   DATA_DIR=/root/autodl-tmp bash setup/setup_server.sh   # 手动指定
#
# 💡 支持「无卡模式」安装：检测不到 GPU 时不会退出，照常装环境。
#    这样可以在不计 GPU 费的模式下装环境 + 下模型，装完再切正常模式跑评测。
#    切回 GPU 模式后先跑: bash setup/check_gpu.sh
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------- 0. 定位数据盘
# 系统盘通常很小(30G)，环境和模型都必须放数据盘，否则装到一半爆盘。
if [ -z "${DATA_DIR:-}" ]; then
  for cand in /root/autodl-tmp /root/autodl-fs /root/data /data /mnt/data "$HOME"; do
    if [ -d "$cand" ] && [ -w "$cand" ]; then DATA_DIR="$cand"; break; fi
  done
fi
DATA_DIR="${DATA_DIR:-$HOME}"
WORK_DIR="$DATA_DIR/vla-work"
ENV_NAME="openvla"

echo "==> 数据盘: $DATA_DIR"
echo "==> 工作目录: $WORK_DIR"
mkdir -p "$WORK_DIR"

# ---------------------------------------------------------- 1. 磁盘检查
AVAIL_GB=$(df -BG --output=avail "$DATA_DIR" | tail -1 | tr -dc '0-9')
echo "==> 数据盘可用: ${AVAIL_GB} GB"
cat <<EOM
    空间需求参考:
      conda + PyTorch + CUDA 库     ~18 GB
      单个 checkpoint (7B bf16)     ~15 GB
      4 个 checkpoint 全下          ~60 GB
      LIBERO + assets                ~1 GB
      => 只跑一个 suite 约 35 GB；四个全跑约 80 GB
EOM
if [ "$AVAIL_GB" -lt 35 ]; then
  echo "❌ 可用空间不足 35 GB，连跑一个 suite 都不够。请先扩容数据盘。"; exit 1
elif [ "$AVAIL_GB" -lt 80 ]; then
  echo "⚠️  空间够跑单个 suite，但存不下 4 个 checkpoint。"
  echo "    对策: 一次只下一个，跑完删掉再下下一个（见 scripts/download_checkpoints.sh）"
fi

# ---------------------------------------------------------- 2. GPU 检查（可跳过）
HAS_GPU=0
if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
  nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

  # OpenVLA 评测路径写死了 flash_attention_2 + bfloat16，均要求 Ampere(sm80)+
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
  if [ -n "$CC" ] && [ "$CC" -lt 80 ]; then
    cat <<EOM
❌ GPU 算力等级过低，OpenVLA 跑不起来。

   experiments/robot/openvla_utils.py:45 写死了 attn_implementation="flash_attention_2"，
   加上 torch_dtype=torch.bfloat16 —— 两者都需要 Ampere (sm80) 及以上。

   不可用: V100(7.0) / T4(7.5) / RTX 2080Ti(7.5)
   可  用: A100(8.0) / A10·A5000·A6000·RTX 3090(8.6) / RTX 4090·L20·L40S(8.9) / H100(9.0)
EOM
    exit 1
  fi

  VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  [ "$VRAM_MB" -lt 16000 ] && \
    echo "⚠️  显存 ${VRAM_MB}MB < 16GB。bf16 需 15GB，请改用 4bit: --load_in_4bit True"
else
  echo "⚠️  未检测到 GPU —— 按「无卡模式安装」继续。"
  echo "    装完切回 GPU 模式后，先跑 bash setup/check_gpu.sh 验证再开始评测。"
fi

# ---------------------------------------------------------- 3. 环境变量
# HuggingFace 默认把模型缓存到 ~/.cache（系统盘），30G 系统盘必然爆。
HF_CACHE="$DATA_DIR/huggingface"
mkdir -p "$HF_CACHE"
add_env () { grep -qF "$1" "$HOME/.bashrc" 2>/dev/null || echo "$1" >> "$HOME/.bashrc"; }
add_env "# --- OpenVLA/LIBERO ---"
add_env "export HF_HOME=$HF_CACHE"          # 模型缓存放数据盘
add_env "export MUJOCO_GL=egl"              # 服务器无显示器，走 EGL 离屏渲染
add_env "export PYOPENGL_PLATFORM=egl"
add_env "export OPENVLA_ROOT=$WORK_DIR/openvla"
export HF_HOME="$HF_CACHE" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENVLA_ROOT="$WORK_DIR/openvla"
echo "==> HF_HOME=$HF_HOME  (模型缓存已重定向到数据盘)"

# ---------------------------------------------------------- 4. Conda
if ! command -v conda >/dev/null; then
  echo "==> 安装 Miniconda 到数据盘"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$DATA_DIR/miniconda3"
  export PATH="$DATA_DIR/miniconda3/bin:$PATH"
  add_env "export PATH=$DATA_DIR/miniconda3/bin:\$PATH"
fi
eval "$(conda shell.bash hook)"

# 镜像自带的 conda 默认把环境建在系统盘(/root/miniconda3/envs)，而本环境约 18GB，
# 30GB 系统盘装不下。强制把环境目录指到数据盘。
CONDA_ENVS="$DATA_DIR/conda-envs"
mkdir -p "$CONDA_ENVS"
conda config --add envs_dirs "$CONDA_ENVS" 2>/dev/null || true
echo "==> conda 环境目录: $CONDA_ENVS"

# 租用镜像常预置清华源，其中 anaconda/pkgs/free 已被 Anaconda 废弃、清华下线，
# 留在 channels 里会让 conda create 直接 404 失败。逐个剔除已失效的频道。
for dead in "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free" \
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro" \
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2"; do
  conda config --remove channels "$dead" 2>/dev/null || true
done

# ⚠️ 必须用 Python 3.10.13：OpenVLA 的 classifiers 只到 3.10，且 robosuite==1.4.1 /
#    timm==0.9.10 / tokenizers==0.19.1 在 3.11+ 上未必有 wheel。
#    镜像自带的 Python(可能是 3.12)和 PyTorch 一律不用 —— pyproject 硬 pin torch==2.2.0。
if ! conda env list | grep -qE "(^|/)${ENV_NAME}\\s"; then
  # 先用现有频道配置；失败则绕开一切自定义频道，直连官方 defaults 重试。
  conda create -y -n "$ENV_NAME" python=3.10.13 || {
    echo "==> 频道配置有问题，改用 --override-channels 直连官方源重试"
    conda create -y -n "$ENV_NAME" python=3.10.13 --override-channels -c defaults || {
      cat <<'EOM'
❌ conda 建环境失败。多半是镜像预置的 .condarc 里有失效频道。
   查看: conda config --show channels
   重置: mv ~/.condarc ~/.condarc.bak && bash setup/setup_server.sh
EOM
      exit 1
    }
  }
fi
conda activate "$ENV_NAME"
PYV=$(python -c "import sys;print('%d.%d'%sys.version_info[:2])")
[ "$PYV" = "3.10" ] || { echo "❌ 当前 Python $PYV，应为 3.10。环境激活有误，请检查。"; exit 1; }

# 核实环境真的建在数据盘上。若 conda 早先已在系统盘建过同名环境，
# envs_dirs 不会把它挪走，必须删掉重建，否则 ~18GB 会把 30GB 系统盘撑爆。
ENV_PREFIX=$(python -c "import sys;print(sys.prefix)")
case "$ENV_PREFIX" in
  "$CONDA_ENVS"/*) echo "==> 环境位置正确: $ENV_PREFIX" ;;
  *) cat <<EOM
❌ 环境建在了系统盘: $ENV_PREFIX
   （期望在 $CONDA_ENVS 下）

   这多半是之前已在系统盘建过同名环境。请删掉后重跑本脚本:
     conda deactivate
     conda env remove -n $ENV_NAME -y
     conda clean -a -y && pip cache purge
     bash setup/setup_server.sh
EOM
     exit 1 ;;
esac
echo "==> Python: $(python --version)  (镜像自带版本已隔离，不影响)"

# ---------------------------------------------------------- 5. PyTorch
python -c "import torch" 2>/dev/null || \
  pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
      --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------- 6. OpenVLA
cd "$WORK_DIR"
[ -d openvla ] || git clone https://github.com/openvla/openvla.git
cd openvla
pip install -e .
pip install transformers==4.40.1     # 必须锁定：-e . 可能装上更高版本，是最常见的报错源

# Flash Attention 2 —— ⚠️ 评测也必需，不是可选项（openvla_utils.py:45 写死）
#
# flash-attn 的 setup.py 会先去 GitHub Releases 下预编译 wheel，国内服务器
# 常常连不上，下载一失败整个构建就崩（"Remote end closed connection"）。
# FLASH_ATTENTION_FORCE_BUILD=TRUE 直接跳过那次下载，走本地源码编译。
pip install packaging ninja
if ! python -c "import flash_attn" 2>/dev/null; then
  RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
  JOBS=$(( RAM_GB / 8 )); [ "$JOBS" -lt 1 ] && JOBS=1
  [ "$JOBS" -gt 8 ] && JOBS=8
  echo "==> 编译 flash-attn (内存 ${RAM_GB}GB, MAX_JOBS=$JOBS, 约 20-30 分钟)"

  if [ "${RAM_GB:-0}" -lt 8 ]; then
    cat <<'EOM'
❌ 内存不足 8GB，flash-attn 源码编译会 OOM。
   请切到 GPU 模式（内存充足）后重跑本脚本。
EOM
    exit 1
  fi

  FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=$JOBS \
    pip install "flash-attn==2.5.5" --no-build-isolation || {
    echo "==> 首次失败，清缓存后单线程重试"
    pip cache remove flash_attn 2>/dev/null || true
    FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=2 \
      pip install "flash-attn==2.5.5" --no-build-isolation || {
      cat <<'EOM'
❌ flash-attn 编译失败，而它是评测的硬性依赖。

   兜底方案: 把 openvla_utils.py:45 的 attn_implementation 改成 "sdpa"
     sed -i 's/flash_attention_2/sdpa/' \
       "$OPENVLA_ROOT/experiments/robot/openvla_utils.py"
   速度略降，评测结果不受影响。
EOM
      exit 1
    }
  }
fi

# ---------------------------------------------------------- 7. 系统库
# EGL/OpenGL 运行库：缺了 PyOpenGL 会返回 None，报
# "'NoneType' object has no attribute 'eglQueryString'"。必须先 apt-get update。
if command -v apt-get >/dev/null; then
  apt-get update -qq || true
  apt-get install -y -qq libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 \
                        fonts-wqy-zenhei || echo "⚠️  apt 安装失败，稍后可跑 setup/fix_env.sh 补"
fi

# ---------------------------------------------------------- 8. LIBERO
cd "$WORK_DIR"
[ -d LIBERO ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
# ⚠️ LIBERO/libero/ 下没有 __init__.py，其 setup.py 里
#    packages=[p for p in find_packages() if p.startswith("libero")] 会返回空列表。
#    新版 setuptools 按 PEP 660 严格安装 => 什么都装不上。compat 模式恢复旧行为。
cd LIBERO && pip install -e . --config-settings editable_mode=compat
python -c "import libero" 2>/dev/null || {
  echo "==> compat 模式无效，改用 PYTHONPATH 兜底"
  add_env "export PYTHONPATH=$WORK_DIR/LIBERO:\$PYTHONPATH"
  export PYTHONPATH="$WORK_DIR/LIBERO:${PYTHONPATH:-}"
}

cd "$WORK_DIR/openvla" && pip install -r experiments/robot/libero/libero_requirements.txt
pip install "imageio[ffmpeg]" matplotlib pandas

# ⚠️ robosuite 1.4.1 声明 numpy>=1.13.3 / mujoco>=2.3.0（上界全开），会拉来 numpy 2.x，
#    而 torch 2.2.0 与 tensorflow 2.15.0 都要求 numpy<2 —— 不钉死会导致 import torch 直接崩。
#    必须放在所有 pip 安装之后，把被顶上去的 numpy 压回来。
# opencv-python 5.x 同样要求 numpy>=2，robosuite 未锁版本会把它拉上来，一起钉死。
pip install "numpy==1.26.4" "opencv-python==4.10.0.84"

# LIBERO 首次 import 会交互式问数据集路径，非交互环境下会 EOF 报错。
# 喂 "N" 走默认路径，提前把 ~/.libero/config.yaml 生成好。
[ -f "$HOME/.libero/config.yaml" ] || echo "N" | python -c "import libero.libero" >/dev/null 2>&1 || true

# AutoDL 等镜像可能把 OMP_NUM_THREADS 设成非法值，libgomp 会持续报警
add_env "export OMP_NUM_THREADS=8"
export OMP_NUM_THREADS=8

# ---------------------------------------------------------- 9. 自检
echo ""
echo "==> 环境自检"
python - <<'PY'
import importlib
for m, want in [("numpy","1.26.4"), ("torch","2.2.0"), ("transformers","4.40.1"),
                ("robosuite","1.4.1"), ("libero",None), ("flash_attn",None)]:
    try:
        got = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {'✅' if (want is None or got == want) else f'⚠️  期望 {want}'} {m}: {got}")
    except Exception as e:
        print(f"  ❌ {m}: {type(e).__name__}: {e}")
PY

# LIBERO 能否真正启动 —— 这一步提前暴露 EGL/MuJoCo 问题
python - <<'PY'
import os; os.environ.setdefault("MUJOCO_GL", "egl")
try:
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()["libero_spatial"]()
    print(f"  ✅ LIBERO 可用 | libero_spatial: {ts.n_tasks} 个任务")
    print(f"     示例指令: {ts.get_task(0).language}")
except Exception as e:
    print(f"  ❌ LIBERO 启动失败: {e}")
    print("     多半是 MUJOCO_GL 未设为 egl，或缺 libegl1 (apt install -y libegl1 libgl1)")
PY

cat <<EOM

============================================================
✅ 安装完成

  source ~/.bashrc
  conda activate $ENV_NAME

下一步:
  1) 下载 checkpoint（无卡模式下做最省钱）
       bash scripts/download_checkpoints.sh libero_spatial
  2) 熟悉 LIBERO（不加载模型）
       python scripts/explore_libero.py inspect
  3) 冒烟测试
       bash scripts/run_eval.sh smoke

  数据盘: $DATA_DIR   HF_HOME: $HF_CACHE
============================================================
EOM
