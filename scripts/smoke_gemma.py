"""Load Gemma 4 and run one image-question CUDA inference.

Run this separately from smoke_falcon.py. An RTX 4050 Laptop GPU has 6 GiB
of VRAM, which is unlikely to fit the referenced 4B CUDA checkpoint at its
native precision. The script makes that limitation observable rather than
attempting unsafe concurrent model loading.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", default="Describe this image in one sentence.")
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image does not exist: {args.image}")

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch is required. Run scripts/bootstrap.ps1 and retry.")

    model_id = os.getenv("GEMMA_HF_MODEL_ID", "google/gemma-4-E4B-it")
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    load_seconds = time.perf_counter() - started

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": args.prompt}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    inputs = inputs.to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.decode(output[0][input_length:], skip_special_tokens=True).strip()
    inference_seconds = time.perf_counter() - started
    print(f"Gemma loaded in {load_seconds:.1f}s")
    print(f"Answer: {answer}")
    print(f"Inference: {inference_seconds:.1f}s; peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
