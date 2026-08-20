#!/usr/bin/env bash
# ============================================================================
# 修复 OpenVLA + LIBERO 环境的三个已知问题
#
# 实测发现（2026-08，租用 AutoDL RTX 4090）：
#  ① robosuite 1.4.1 声明 numpy>=1.13.3 / mujoco>=2.3.0，上界全开，
#     pip 会拉来 numpy 2.x，而 torch 2.2.0 与 tensorflow 2.15.0 都要求 numpy<2
#     => import torch 直接崩
#  ② LIBERO/libero/ 下没有 __init__.py，其 setup.py 的
#     packages=[p for p in find_packages() if p.startswith("libero")] 返回空列表，
#     新版 setuptools 按 PEP 660 严格安装 => 一个包都没装上
#  ③ EGL 运行库缺失 => PyOpenGL 返回 None => 'NoneType' has no attribute 'eglQueryString'
#
# 用法: bash setup/fix_env.sh   （需先 conda activate openvla）
# ============================================================================
set -uo pipefail

DATA_DIR="${DATA_DIR:-/root/autodl-tmp}"
LIBERO_DIR="${LIBERO_DIR:-$DATA_DIR/vla-work/LIBERO}"

# ⚠️ 必须硬失败：之前有过在 base 环境里跑本脚本、把包装错地方的事故。
PREFIX=$(python -c "import sys; print(sys.prefix)" 2>/dev/null || echo "")
case "$PREFIX" in
  *openvla*) echo "==> 环境: $PREFIX" ;;
  *) cat <<EOM
❌ 当前不在 openvla 环境里（sys.prefix = ${PREFIX:-未知}），拒绝继续，
   否则会把包装进 base 把镜像环境搞坏。

   正确顺序（注意 source 要在 activate 之前，
   因为 source ~/.bashrc 会重跑 conda init 把你踢回 base）:

     source ~/.bashrc
     conda activate openvla
     bash setup/fix_env.sh
EOM
     exit 1 ;;
esac

# 环境必须在数据盘上。系统盘通常只有 30GB，放不下 ~18GB 的环境。
case "$PREFIX" in
  /root/miniconda3/*|/usr/*|/opt/*)
    echo "⚠️  环境位于系统盘: $PREFIX"
    df -h / | tail -1
    echo "    系统盘一旦占满会导致各种诡异失败。建议删掉重建到数据盘:"
    echo "      conda deactivate && conda env remove -n openvla -y"
    echo "      bash setup/setup_server.sh" ;;
esac

export TMPDIR="${TMPDIR:-$DATA_DIR/tmp}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_DIR/pip-cache}"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "==> ① 安装 EGL / OpenGL 运行库"
if command -v apt-get >/dev/null; then
  apt-get update -qq && \
  apt-get install -y -qq libegl1 libgl1 libglvnd0 libosmesa6 libglib2.0-0 fonts-wqy-zenhei \
    || echo "⚠️  apt 安装失败，可能需要手动处理"
fi

echo "==> ② numpy 降到 1.x，并同步降 opencv-python"
# opencv-python 5.x 要求 numpy>=2，与 torch 2.2 / tf 2.15 的 numpy<2 直接冲突。
# robosuite 只写了 opencv-python(无版本)，会被拉到 5.x —— 必须一起钉死。
pip install -q "numpy==1.26.4" "opencv-python==4.10.0.84"

echo "==> ③ 重装 LIBERO（compat 模式绕开 find_packages 空列表问题）"
if [ -d "$LIBERO_DIR" ]; then
  ( cd "$LIBERO_DIR" && pip install -q -e . --config-settings editable_mode=compat )
  # 兜底：compat 模式在个别 setuptools 版本下仍可能失效，直接把源码目录挂到 PYTHONPATH
  python -c "import libero" 2>/dev/null || {
    echo "    compat 模式无效，改用 PYTHONPATH 兜底"
    grep -qF "$LIBERO_DIR" "$HOME/.bashrc" || \
      echo "export PYTHONPATH=$LIBERO_DIR:\$PYTHONPATH" >> "$HOME/.bashrc"
    export PYTHONPATH="$LIBERO_DIR:${PYTHONPATH:-}"
  }
else
  echo "❌ 找不到 $LIBERO_DIR，请用 LIBERO_DIR=... 指定"
fi

echo "==> ④ 初始化 LIBERO 配置文件"
# LIBERO 首次 import 会交互式询问数据集路径以生成 ~/.libero/config.yaml，
# 非交互环境下直接 EOF 报错。喂一个 "N" 用默认路径把它生成掉。
if [ ! -f "$HOME/.libero/config.yaml" ]; then
  echo "N" | python -c "import libero.libero" 2>&1 | tail -6
fi
[ -f "$HOME/.libero/config.yaml" ] && echo "    ✅ ~/.libero/config.yaml 已生成" \
                                   || echo "    ❌ 配置文件仍未生成"

echo "==> ⑤ 修正 OMP_NUM_THREADS（AutoDL 默认值非法，libgomp 会报警）"
grep -qF "OMP_NUM_THREADS" "$HOME/.bashrc" || echo "export OMP_NUM_THREADS=8" >> "$HOME/.bashrc"
export OMP_NUM_THREADS=8

echo ""
echo "==> 复检"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
python - <<'PY'
import importlib
ok = True
for m, want in [("numpy","1.26.4"), ("cv2","4.10"), ("torch","2.2.0"),
                ("transformers","4.40.1"), ("robosuite","1.4.1"), ("flash_attn",None)]:
    try:
        got = getattr(importlib.import_module(m), "__version__", "已安装")
        good = want is None or got.startswith(want)
        print(f"  {'✅' if good else '⚠️ 期望 '+want} {m}: {got}")
        ok &= good
    except Exception as e:
        print(f"  ❌ {m}: {type(e).__name__}: {e}"); ok = False

try:
    from libero.libero import benchmark
    ts = benchmark.get_benchmark_dict()["libero_spatial"]()
    print(f"  ✅ LIBERO 可启动 | {ts.n_tasks} 个任务 | 示例: {ts.get_task(0).language}")
except Exception as e:
    print(f"  ❌ LIBERO: {e}"); ok = False

print("\n" + ("✅ 全部通过，可以跑 bash scripts/run_eval.sh smoke"
              if ok else "⚠️  仍有问题，把上面输出贴出来"))
PY
