"""
训练产物的定位 —— 只为一件事：**别让"路径写错"长得像"代码坏了"**。

`PeftModel.from_pretrained` 拿到一个不存在的目录时，会先把它当成 HuggingFace
仓库名去联网查，于是抛出来的是 `HFValidationError: Repo id must be in the form
'namespace/repo_name'` ——一句和"你的 checkpoint 在哪"毫无关系的话。
本模块在调用 peft 之前先自己查一遍，查不到就把**实际存在的 checkpoint 列出来**。

（同一道理的第三次：`finetune_kframe.py` 对数据集路径、`probe_vram.py` 对子进程
 失败都做过这件事。别人的报错信息不是为你的场景写的。）
"""

from pathlib import Path

CONFIG = "adapter_config.json"


def is_adapter(p: Path) -> bool:
    return (p / CONFIG).is_file()


def list_adapters(run_root="runs"):
    """run_root 下所有真正可加载的 checkpoint，按 (运行, 步数) 排序。"""
    root = Path(run_root).expanduser()
    if not root.is_dir():
        return []
    out = [d for d in root.glob("*/adapter/step*") if is_adapter(d)]

    def key(d: Path):
        s = d.name[4:]
        return (d.parents[1].name, int(s) if s.isdigit() else -1)

    return sorted(out, key=key)


def resolve_adapter(path, run_root="runs") -> Path:
    """
    把 `--adapter` 解析成一个确实能加载的目录，否则抛出**带清单**的错误。
    """
    p = Path(path).expanduser()
    if is_adapter(p):
        return p

    have = list_adapters(run_root)
    why = ("这个目录不存在" if not p.exists() else
           f"目录在，但里面没有 {CONFIG}（存的是"
           f" {sorted(x.name for x in p.iterdir())[:6]} …）")
    lines = [f"加载不了 adapter: {p}", f"  {why}"]
    if p.exists() and not is_adapter(p):
        # 上一级/下一级常常才是对的：runs/<exp>/adapter 与 .../adapter/step30000
        near = [d for d in list(p.glob("step*")) + [p.parent] if is_adapter(d)]
        if near:
            lines.append(f"  你要的多半是: {near[0]}")
    lines.append(f"\n{Path(run_root).expanduser()} 下真正存在的 checkpoint：")
    lines += [f"  {d}" for d in have] or ["  （一个都没有 —— 训练的 runs/ 是不是在别的盘？"
                                          "用 --run_root 指过去，或给 --adapter 绝对路径）"]
    raise SystemExit("\n".join(lines))


def _selftest() -> None:
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "runs"
        good = root / "G2+x" / "adapter" / "step30000"
        good.mkdir(parents=True)
        (good / CONFIG).write_text(json.dumps({}))
        (root / "G2+x" / "adapter" / "step2500").mkdir(parents=True)   # 半截的，不算

        assert resolve_adapter(good, root) == good
        print("✅ 1/3 正常路径原样返回")

        assert [d.name for d in list_adapters(root)] == ["step30000"]
        print("✅ 2/3 没有 adapter_config.json 的目录不算 checkpoint")

        try:
            resolve_adapter(root / "G2+x" / "adapter", root)
        except SystemExit as e:
            assert "你要的多半是" in str(e) and "step30000" in str(e), e
            print("✅ 3/3 指到上一级时，错误信息里直接给出正确路径")
        else:
            raise AssertionError("应当抛 SystemExit")


if __name__ == "__main__" and __package__ in (None, ""):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    _selftest()
