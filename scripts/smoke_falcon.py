"""Load Falcon Perception and run a single CUDA segmentation inference."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from falcon_perception import (
    build_prompt_for_task,
    load_and_prepare_model,
    setup_torch_config,
)
from falcon_perception.batch_inference import (
    BatchInferenceEngine,
    process_batch_and_generate,
)
from PIL import Image


def _load_local_env() -> None:
    project_root = Path(__file__).resolve().parents[1]
    env_path = project_root / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
import torch


def main() -> int:
    _load_local_env()
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--query", default="dog")
    parser.add_argument(
        "--task", default="segmentation", choices=("segmentation", "detection")
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image does not exist: {args.image}")

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA PyTorch is required. Run scripts/bootstrap.ps1 and retry."
        )

    device = os.getenv("CUDA_DEVICE", "cuda:0")
    model_id = os.getenv("FALCON_HF_MODEL_ID", "tiiuae/Falcon-Perception")
    setup_torch_config()
    started = time.perf_counter()
    model, tokenizer, model_args = load_and_prepare_model(
        hf_model_id=model_id,
        hf_revision=os.getenv("FALCON_HF_REVISION", "main"),
        device=device,
        dtype=os.getenv("FALCON_TORCH_DTYPE", "bfloat16"),
        compile=False,
    )
    load_seconds = time.perf_counter() - started

    image = Image.open(args.image).convert("RGB")
    batch = process_batch_and_generate(
        tokenizer,
        [(image, build_prompt_for_task(args.query, args.task))],
        max_length=getattr(model_args, "max_seq_len", None) or 4096,
        min_dimension=256,
        max_dimension=1024,
        patch_size=getattr(model_args, "spatial_patch_size", 16),
    )
    batch = {
        key: value.to(model.device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    stop_ids = [tokenizer.eos_token_id]
    if getattr(tokenizer, "end_of_query_token_id", None) is not None:
        stop_ids.append(tokenizer.end_of_query_token_id)

    started = time.perf_counter()
    _, auxiliary = BatchInferenceEngine(model, tokenizer).generate(
        **batch,
        max_new_tokens=100,
        temperature=0.0,
        stop_token_ids=stop_ids,
        seed=42,
    )
    inference_seconds = time.perf_counter() - started
    bboxes = getattr(auxiliary[0], "bboxes_raw", [])
    masks = getattr(auxiliary[0], "masks_rle", []) or []
    print(f"Falcon loaded in {load_seconds:.1f}s")
    print(f"Query: {args.query!r}; boxes: {len(bboxes)}; masks: {len(masks)}")
    print(
        f"Inference: {inference_seconds:.1f}s; peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
