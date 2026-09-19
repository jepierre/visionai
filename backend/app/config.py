from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    images_dir: Path
    output_dir: Path
    thumbnails_dir: Path
    annotated_dir: Path
    falcon_model_id: str
    falcon_revision: str
    falcon_dtype: str
    cuda_device: str
    ollama_base_url: str
    ollama_model: str


def _load_dotenv(root_dir: Path) -> None:
    env_path = root_dir / ".env"
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def _getenv(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip()
    return normalized or default


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    _load_dotenv(root_dir)
    output_dir = root_dir / "output"
    return Settings(
        root_dir=root_dir,
        images_dir=root_dir / "images",
        output_dir=output_dir,
        thumbnails_dir=output_dir / "thumbnails",
        annotated_dir=output_dir / "annotated",
        falcon_model_id=_getenv("FALCON_HF_MODEL_ID", "tiiuae/Falcon-Perception"),
        falcon_revision=_getenv("FALCON_HF_REVISION", "main"),
        falcon_dtype=_getenv("FALCON_TORCH_DTYPE", "bfloat16"),
        cuda_device=_getenv("CUDA_DEVICE", "cuda:0"),
        ollama_base_url=_getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=_getenv("OLLAMA_MODEL", "gemma3:4b"),
    )