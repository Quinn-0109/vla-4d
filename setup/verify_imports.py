"""
开跑前的导入自检 —— 在下载 15GB checkpoint 之前先确认整条依赖链是通的。

    python setup/verify_imports.py

之所以需要它: OpenVLA 的 get_model() 会连锁导入
prismatic -> vla.datasets.rlds -> dlimp -> tensorflow_datasets -> protobuf，
任何一环版本不对都要等到评测启动时才暴露，白白浪费下载时间。
"""
import os
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

CHECKS = [
    ("numpy",                       "import numpy"),
    ("torch (CUDA)",                "import torch; assert torch.cuda.is_available()"),
    ("cv2",                         "import cv2"),
    ("protobuf",                    "import google.protobuf"),
    ("tensorflow",                  "import tensorflow"),
    ("tensorflow_datasets",         "import tensorflow_datasets"),
    ("dlimp",                       "import dlimp"),
    ("transformers",                "import transformers"),
    ("prismatic (完整依赖链)",       "import prismatic.models"),
    ("robosuite",                   "import robosuite"),
    ("libero",                      "from libero.libero import benchmark"),
    ("draccus",                     "import draccus"),
    ("openvla experiments",         "from experiments.robot.robot_utils import get_model"),
]

def main() -> int:
    root = os.environ.get("OPENVLA_ROOT")
    if root:
        sys.path.insert(0, root)
    else:
        print("⚠️  OPENVLA_ROOT 未设置，最后一项会失败")

    failed = []
    for name, code in CHECKS:
        try:
            exec(code, {})
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            failed.append((name, e))

    print()
    if failed:
        print(f"❌ {len(failed)} 项失败。先跑 bash setup/fix_env.sh")
        print("\n首个失败的完整栈:")
        try:
            exec(dict(CHECKS)[failed[0][0]], {})
        except Exception:
            traceback.print_exc()
        return 1

    print("✅ 依赖链完整，可以下载 checkpoint 并开跑")
    return 0


if __name__ == "__main__":
    sys.exit(main())
