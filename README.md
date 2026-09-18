# VisionAI — Local Grounded Image Chat

VisionAI is a planned local web application for asking questions about images stored in the `images/` directory. It combines Falcon Perception object masks and bounding boxes with a local Gemma 4B-class model served through Ollama, so answers can be grounded in visible evidence.

## Planned user experience

1. Place images in `images/`.
2. Open the local app and select an image from the gallery.
3. Enter a question in the prompt bar at the bottom and select **Submit**.
4. The agent decides whether it needs detection, visual reasoning, or both.
5. The app shows the answer and an annotated image with masks, boxes, or both.

Examples:

- “How many cars are there?”
- “Show all people.”
- “Are there more dogs than people?”
- “Describe the largest object.”

## Model roles

| Model | Role |
| --- | --- |
| Falcon Perception | Open-vocabulary detection, instance segmentation, bounding boxes, and exact counts. |
| Local Ollama Gemma 4B model | Handles the visual reasoning and final answer generation without using Hugging Face directly. |

Falcon provides spatial grounding; the local Gemma model provides conversational reasoning.

## Target platform

Version 1 is CUDA/PyTorch only.

- NVIDIA GPU with CUDA support.
- Python 3.10 or later.
- CUDA-compatible PyTorch and torchvision.
- Local Ollama server running in Docker.
- Access to the Falcon Perception model download via Hugging Face.
- A GPU with enough VRAM for Falcon and a compact Gemma 4B-class model; validate this during setup.

Apple Silicon and MLX support are intentionally not included in the first version.

## Planned stack

- **Backend:** Python, FastAPI, Uvicorn, PyTorch/CUDA.
- **Frontend:** React, TypeScript, Vite, Tailwind CSS.
- **Annotation rendering:** Pillow, NumPy, pycocotools.
- **Local VLM runtime:** Ollama with a Gemma 4B-class model in Docker.
- **Testing:** Pytest and Playwright.

OpenCV is not part of the planned dependency set. The initial scope only needs Pillow and NumPy for image loading and mask/box overlays. We will reconsider it only if a future capability specifically requires it.

## Phase 0: local validation

Phase 0 is a CUDA validation harness. It creates an isolated Python environment, checks GPU readiness, and includes separate smoke tests for Falcon and the local Gemma model. The tests intentionally load one model at a time.

### 1) Create the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
cp .env.example .env
```

> If your NVIDIA driver supports a newer CUDA wheel, replace `cu126` with the matching PyTorch wheel tag such as `cu128`.

### 2) Start Ollama in Docker

This project expects Ollama to run in a Docker container on the local machine.

```bash
docker run -d \
  --gpus=all \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --name ollama \
  ollama/ollama:latest
```

If Docker does not expose GPUs on your host, follow the NVIDIA Container Toolkit setup for your platform and keep the `--gpus=all` flag.

### 3) Pull a local Gemma 4B model

```bash
docker exec -it ollama ollama pull gemma3:4b
```

You can replace `gemma3:4b` with any local 4B-class model tag available in your Ollama registry. The default `.env.example` value is `gemma3:4b`.

### 4) Validate the environment and smoke tests

```bash
python scripts/validate_environment.py
python scripts/smoke_falcon.py --image images/dog_running_in_park.jpg --query dog
python scripts/smoke_gemma.py --image images/dog_running_in_park.jpg --model gemma3:4b
```

The model and image are intentionally handled one-at-a-time to avoid combined GPU memory pressure. This is especially important on 6–8 GiB VRAM cards, where a large quantized model can still be memory-constrained.

### 5) Local Ollama configuration

The repo uses `OLLAMA_BASE_URL` and `OLLAMA_MODEL` in the environment file.

```bash
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma3:4b
```

If a different service runs inside Docker and your Python app is also inside Docker on the same virtual network, use `OLLAMA_BASE_URL=http://ollama:11434` instead.

See [PHASE0_REPORT.md](PHASE0_REPORT.md) for the current machine-specific validation notes and acceptance status.

## Repository structure

```text
images/       Source images for the first version
PLAN.md       Phased implementation plan
README.md     Project overview and setup direction
scripts/      Validation and smoke-test utilities
```

The backend, frontend, generated-output, and test directories will be introduced in the implementation phases described in [PLAN.md](PLAN.md).

## Reference projects

- [Gemma4-Visual-Agent — CUDA/PyTorch branch](https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10)
- [mlx-vlm-falcon — pipeline reference](https://github.com/korale77/mlx-vlm-falcon)
- [Gemma 7 Vision Agent | Object Detection + VLM Pipeline](https://www.youtube.com/watch?v=VFYnD1WREdU&list=PLIsoxylL1dFw&index=6)

The first repository provides the relevant CUDA/PyTorch direction. The second is useful for its clear extract → detect/visualize → answer pipeline, but it targets Apple Silicon/MLX and is not a deployment target for this project.
