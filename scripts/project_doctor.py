"""无 GPU 项目检查：只报告可见文件与可验证条件，不把目录存在当实验完成。"""
import argparse
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common.provenance import code_identity
from common.runs import list_adapters


def inspect(root: Path) -> dict:
    files = sorted(p for folder in ("src", "scripts", "setup", "tests")
                   for p in (root / folder).rglob("*.py"))
    errors = []
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8-sig"), feature_version=(3, 10))
        except (SyntaxError, UnicodeError) as e:
            errors.append(f"{path.relative_to(root)}: {e}")
    versions = {}
    for pkg in ("torch", "numpy", "transformers", "peft", "tensorflow", "draccus"):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    logs = root / "results/logs"
    adapters = list_adapters(root / "runs")
    records = sorted(logs.glob("*.episodes.jsonl"))
    return {"root": str(root),"code": code_identity(root),"python_files": len(files),
            "syntax_errors": errors,"package_versions": versions,
            "openvla_root": os.environ.get("OPENVLA_ROOT"),
            "adapters": [str(p) for p in adapters],
            "episode_files": [str(p) for p in records],
            "metadata_missing": [str(p) for p in records if not
                p.with_name(p.name.removesuffix(".episodes.jsonl")+".meta.json").is_file()],
            "scope": "文件/语法/已安装版本检查；不验证 CUDA、EGL、模型前向、训练或成功率",
            "next_action": ("没有本地结果；在训练机运行同一命令，先收集 checkpoint 与逐局日志"
                            if not adapters and not records else
                            "显式选择逐局结果，核验来源后分析；G3-full 独立归档")}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = inspect(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return 1 if report["syntax_errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
