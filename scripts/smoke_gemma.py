"""Run a single image prompt through a local Ollama-backed Gemma 4B model.

This smoke test intentionally avoids Hugging Face and instead uses the local
Ollama API running on the same machine. The default model name is a 4B-class
Gemma tag such as `gemma3:4b`, which can be overridden via the CLI or the
`OLLAMA_MODEL` environment variable.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def _call_ollama(model: str, prompt: str, image_path: Path, base_url: str) -> tuple[str, float, str]:
    image_bytes = image_path.read_bytes()
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
    }
    endpoint = f"{base_url.rstrip('/')}/api/generate"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Ollama is not reachable at {endpoint}. Start `ollama serve`, pull a local model, and retry.\n{exc}"
        ) from exc

    elapsed = time.perf_counter() - started
    answer = str(payload.get("response", "")).strip()
    if not answer:
        raise SystemExit(f"Ollama returned no response for model {model!r}: {payload!r}")
    return answer, elapsed, str(payload.get("model", model))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a local Gemma 4B smoke test via Ollama.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", default="Describe this image in one sentence.", help="Text question or instruction to send with the image.")
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        help="Local Ollama model tag to use. Typical 4B examples: gemma3:4b or a custom local tag.",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Base URL for the local Ollama API server.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image does not exist: {args.image}")

    with Image.open(args.image) as image:
        image.convert("RGB")

    answer, inference_seconds, model_name = _call_ollama(args.model, args.prompt, args.image, args.ollama_url)
    print(f"Ollama model: {model_name}")
    print(f"Answer: {answer}")
    print(f"Inference: {inference_seconds:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
