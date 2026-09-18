# AGENTS.md

## Project overview

This repo is a local grounded image-chat application under active planning/validation. The intended architecture is:

- Falcon Perception for object detection and segmentation
- a local Gemma 4B-class model served through Ollama for reasoning and answer generation
- a Python backend and React frontend planned for later phases

See [README.md](README.md), [PLAN.md](PLAN.md), and [PHASE0_REPORT.md](PHASE0_REPORT.md) for the authoritative product and implementation details.

## Working conventions

- Keep changes scoped and minimal.
- Prefer using existing project docs over re-deriving setup guidance.
- Do not assume Hugging Face is the Gemma runtime path; this repo is explicitly aligned to a local Ollama setup.
- Treat Phase 0 as validation-only, not production implementation.

## Environment and setup

- Use a Python virtual environment at the repo root: `.venv`
- Install CUDA PyTorch first, matching the host driver version before installing the Python package set.
- Use the local Ollama container on `http://localhost:11434` by default.
- Typical local model tag: `gemma3:4b`

### Typical commands

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-phase0.txt
cp .env.example .env
```

```bash
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama:latest

docker exec -it ollama ollama pull gemma3:4b
```

```bash
python scripts/validate_environment.py
python scripts/smoke_falcon.py --image images/dog_running_in_park.jpg --query dog
python scripts/smoke_gemma.py --image images/dog_running_in_park.jpg --model gemma3:4b
```

## Important project-specific cautions

- Falcon and Gemma are intentionally tested separately; do not load them concurrently.
- The machine-specific risk is GPU memory pressure, especially on 6–8 GiB cards.
- The repo currently expects local Ollama for Gemma rather than direct Hugging Face inference.
- Keep `.env` local and do not commit downloaded model caches or secrets.

## Files to consult first

- [README.md](README.md) for setup and workflow
- [PLAN.md](PLAN.md) for phased implementation direction
- [PHASE0_REPORT.md](PHASE0_REPORT.md) for machine-specific validation notes
- [scripts/validate_environment.py](scripts/validate_environment.py) for environment checks
- [scripts/smoke_falcon.py](scripts/smoke_falcon.py) and [scripts/smoke_gemma.py](scripts/smoke_gemma.py) for the isolated model smoke tests

## When adding or updating code

- Prefer small, testable changes.
- Keep Phase 0 scripts explicit and single-model.
- If implementing future server code, keep the architecture aligned with the plan in [PLAN.md](PLAN.md).
- Document any environment assumptions in the relevant README or phase report instead of burying them in code comments.
