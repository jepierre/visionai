"""Report CUDA, Python-package, and model-access readiness without loading models."""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
REQUIRED_PACKAGES = ("torch", "transformers", "falcon_perception", "PIL", "pycocotools")


def load_local_env() -> None:
    """Load simple KEY=VALUE pairs without adding python-dotenv as a dependency."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def nvidia_smi() -> list[str]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return [line for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []


def main() -> int:
    load_local_env()
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "nvidia_smi": nvidia_smi(),
        "packages": {name: importlib.util.find_spec(name) is not None for name in REQUIRED_PACKAGES},
        "models": {
            "falcon": os.getenv("FALCON_HF_MODEL_ID", "tiiuae/Falcon-Perception"),
            "gemma": os.getenv("GEMMA_HF_MODEL_ID", "google/gemma-4-E4B-it"),
        },
    }

    try:
        import torch

        report["torch"] = {
            "version": torch.__version__,
            "compiled_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
        }
    except ImportError:
        report["torch"] = None

    print(json.dumps(report, indent=2))
    ready = bool(report.get("torch") and report["torch"]["cuda_available"])
    packages_ready = all(report["packages"].values())
    if not ready or not packages_ready:
        print("\nNot ready: run scripts/bootstrap.ps1 in a PowerShell session, then retry.", file=sys.stderr)
        return 1
    print("\nEnvironment ready for individual Falcon and Gemma smoke tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
