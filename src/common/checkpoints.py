"""在同一文件系统内完成 checkpoint 后再公开 step 目录。"""
from contextlib import contextmanager
from pathlib import Path
import tempfile


@contextmanager
def checkpoint_directory(destination: Path):
    if destination.exists():
        raise FileExistsError(f"checkpoint 已存在，拒绝覆盖: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "payload"
        staging.mkdir()
        yield staging
        required = ("adapter_config.json", "adapter_model.safetensors", "trainer_state.pt")
        if not all((staging / f).is_file() and (staging / f).stat().st_size for f in required):
            raise ValueError("checkpoint 文件缺失或为空，未发布 step 目录")
        staging.rename(destination)
