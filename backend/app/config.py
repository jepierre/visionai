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


def get_settings() -> Settings:
    root_dir = Path(__file__).resolve().parents[2]
    output_dir = root_dir / "output"
    return Settings(
        root_dir=root_dir,
        images_dir=root_dir / "images",
        output_dir=output_dir,
        thumbnails_dir=output_dir / "thumbnails",
        annotated_dir=output_dir / "annotated",
        falcon_model_id=os.getenv("FALCON_HF_MODEL_ID", "tiiuae/Falcon-Perception"),
        falcon_revision=os.getenv("FALCON_HF_REVISION", "main"),
        falcon_dtype=os.getenv("FALCON_TORCH_DTYPE", "bfloat16"),
        cuda_device=os.getenv("CUDA_DEVICE", "cuda:0"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
    )