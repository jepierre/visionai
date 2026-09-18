# Phase 0 Validation Report

Date: 2026-09-18

## Detected machine

| Item | Result |
| --- | --- |
| Operating system | Linux |
| Python | 3.10.12 |
| GPU | NVIDIA GeForce RTX 3070 |
| GPU memory | 8,192 MiB (approximately 8 GiB) |
| NVIDIA driver | 595.58.03 |
| Existing Python environment | `.venv` created; local validation is still pending |
| Ollama runtime | Not installed in PATH yet |
| Sample input | `images/dog_running_in_park.jpg` |

## Readiness result

This machine is a better fit for local multimodal validation than the earlier Windows 4050 example: the RTX 3070 has about 8 GiB of VRAM, which is sufficient for a local 4B-class Gemma model via Ollama when the model is properly quantized and loaded one-at-a-time.

The workspace is still not ready to run the full validation flow until Ollama is installed and a local Gemma 4B model is pulled. This project is intentionally not using the Hugging Face Gemma checkpoint path; instead, the validation flow uses a local Ollama endpoint and a model tag such as `gemma3:4b` (or another 4B-class Gemma model that is available locally).

## Practical validation plan

1. Install Ollama locally on this Linux host.
2. Start the local server with `ollama serve`.
3. Pull a local 4B-class model, for example `ollama pull gemma3:4b`.
4. Run the Falcon smoke test separately from the Gemma smoke test.
5. Keep the model and image inspection isolated to avoid combined GPU memory pressure.

The original plan to use Hugging Face for Gemma is intentionally replaced here by a local Ollama-only validation path. The goal is to validate the actual runtime behavior of a locally served Gemma 4B-class model on this machine before Phase 1 begins.

## Capacity outlook

The RTX 3070 with 8 GiB VRAM is much more realistic for a compact local VLM path than the earlier 6 GiB laptop GPU. The risk is still that larger Gemma checkpoints may exceed memory at native precision, so the validation should prefer a quantized 4B-class model and keep Falcon and Gemma loaded separately.

This is the specific reason the smoke scripts remain single-model and not concurrent.

## Next commands

```bash
source .venv/bin/activate
ollama serve
ollama pull gemma3:4b
python scripts/validate_environment.py
python scripts/smoke_falcon.py --image images/dog_running_in_park.jpg --query dog
python scripts/smoke_gemma.py --image images/dog_running_in_park.jpg --model gemma3:4b
```

If your local Ollama registry uses a different 4B model tag, override it with the `--model` argument or the `OLLAMA_MODEL` environment variable.
