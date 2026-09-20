# VisionAI — Local Grounded Image Chat

VisionAI is a local web application in active development for asking questions about images stored in the `images/` directory. It combines Falcon Perception object masks and bounding boxes with a local Gemma 4B-class model served through Ollama, so answers can be grounded in visible evidence.

## User experience

1. Place images in `images/`.
2. Open the local app and select an image from the gallery.
3. Enter a question in the prompt bar at the bottom and select **Submit**.
4. Choose Falcon-only, Gemma-only, or agent mode, then submit the request.
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

## Stack

- **Backend:** Python, FastAPI, Uvicorn, PyTorch/CUDA.
- **Frontend:** React, TypeScript, Vite, and Tailwind CSS.
- **Annotation rendering:** Pillow, NumPy, pycocotools.
- **Local VLM runtime:** Ollama with a Gemma 4B-class model in Docker.
- **Testing:** Python `unittest` for backend API, planner, cache, and renderer coverage; the frontend is validated with the Vite production build.

OpenCV is not part of the planned dependency set. The initial scope only needs Pillow and NumPy for image loading and mask/box overlays. We will reconsider it only if a future capability specifically requires it.

See [docs/agentic-workflow.md](docs/agentic-workflow.md) for the current frontend/backend workflow diagram, API boundaries, and agent routing loop.

## Current repo state

The repository now includes a working Phase 1-5 application scaffold:

- A FastAPI backend with `GET /api/images`, safe image and thumbnail routes, `POST /api/detect`, `POST /api/chat`, and annotated-image serving.
- A React, TypeScript, Tailwind, and Vite frontend that loads the image catalog, supports image selection, mask/box/combined display modes, object detection requests, and single-turn grounded chat.
- Falcon detection normalization and Pillow-based annotation rendering.
- Ollama-backed visual question answering for direct scene questions and Falcon-first routing for counting, comparison, and location-style prompts.
- A bounded action planner with `DETECT`, `DETECT_EACH`, `CROP`, `COMPARE`, `VLM`, and `ANSWER` actions.
- Largest-object crop analysis, Falcon detection caching, serialized GPU inference, model warmup, and downloadable run artifacts.
- Unit tests for API routes, Falcon normalization/cache behavior, planner actions, and annotation rendering.

Current limitations:

- Falcon live inference can still fail if the host is missing Python development headers required by Triton.
- Ollama must be running locally and have the configured model pulled before direct VLM responses will work.
- Falcon and Ollama live inference still depends on local CUDA, model availability, and available GPU memory.
- The synchronous single-process backend cannot interrupt a CUDA kernel that is already running after a browser request is abandoned.

## Phase 0: local validation

Phase 0 is a CUDA validation harness. It creates an isolated Python environment, checks GPU readiness, and includes separate smoke tests for Falcon and the local Ollama model. The tests intentionally load one model at a time.

### 1) Create and use the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r backend/requirements.txt
```

Activate the environment with `source .venv/bin/activate` in each new shell before running the backend, validation scripts, or tests.

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

You can replace `gemma3:4b` with any local 4B-class model tag available in your Ollama registry. Pass the selected tag to the smoke test with `--model`.

### 4) Run the complete app with Docker Compose

The repository includes a resource-bounded Compose stack for Ollama, the CUDA backend, and the static frontend. The backend and Ollama services each reserve one NVIDIA GPU; CPU and memory limits prevent idle services from consuming the whole host.

```bash
cp .env.example .env
./scripts/composectl.sh build
./scripts/composectl.sh start
./scripts/composectl.sh pull-model
```

Open `http://127.0.0.1:5173`. Useful commands:

```bash
./scripts/composectl.sh status
./scripts/composectl.sh logs backend
./scripts/composectl.sh stop
```

#### Use an external Ollama container

If Ollama is already running separately, use the external Compose file. It starts only the backend and frontend and connects the backend to Ollama through `host.docker.internal:11434`.

```bash
COMPOSE_FILE=docker-compose.external.yml ./scripts/composectl.sh build
COMPOSE_FILE=docker-compose.external.yml ./scripts/composectl.sh start
COMPOSE_FILE=docker-compose.external.yml ./scripts/composectl.sh status
```

Override the external Ollama address with `EXTERNAL_OLLAMA_BASE_URL` when it is not running on the Docker host.

Pull the model with the separately managed Ollama service, not `composectl.sh`:

```bash
docker exec -it ollama ollama pull "$OLLAMA_MODEL"
```

Compose uses `http://ollama:11434` internally for the backend. Host-run scripts default to `http://localhost:11434`.

### 5) Validate the environment and smoke tests

```bash
python scripts/validate_environment.py
python scripts/smoke_falcon.py --image images/dog_running_in_park.jpg --query dog
python scripts/smoke_gemma.py --image images/dog_running_in_park.jpg --model gemma3:4b
```

The model and image are intentionally handled one-at-a-time to avoid combined GPU memory pressure. This is especially important on 6–8 GiB VRAM cards, where a large quantized model can still be memory-constrained.

If Falcon fails with a Triton compile error mentioning `Python.h`, install the Python development headers for your interpreter on the host before retrying.

```bash
sudo apt install python3.10-dev
```

## Run the app

Start the backend and frontend in separate terminals.

### 1) Start the backend

```bash
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints will be available at `http://127.0.0.1:8000`.

### 2) Start the frontend

The frontend expects Node 20 or newer. If you installed Node through `nvm`, load it first.

```bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm use 20
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

Open `http://127.0.0.1:5173` in your browser. The Vite dev server proxies `/api` requests to the backend on port 8000.

### 3) Expected behavior

- The gallery is populated from `images/`.
- `Run detection` calls Falcon and returns detections plus an annotated image when Falcon is available.
- Falcon mode runs detection and annotation only; Gemma mode sends the image directly to Ollama; agent mode selects a bounded workflow.
- Count and comparison questions use deterministic Falcon results, while grounded location/detail questions can combine Falcon with Ollama.
- General scene-description questions route directly to Ollama.
- If either runtime is unavailable, the API returns a 503 error with the blocking dependency in the message.
- `GET /api/health`, `GET /api/ollama/models`, and `POST /api/warmup` expose service and model readiness checks.
- The run panel exposes bounded agent actions and can download the annotated image and JSON run summary.

### 4) Run the tests

```bash
python -m unittest discover -s backend/tests
```

## Repository structure

```text
backend/      FastAPI app, routing, model adapters, agent logic, rendering, and tests
frontend/     React/Vite application
images/       Source images for the first version
docs/         Phased plan and agent workflow documentation
README.md     Project overview and setup direction
scripts/      Validation and smoke-test utilities
```

Additional backend subpackages are under `backend/app/` for models, agent orchestration, rendering, schemas, and image services.

## Reference projects

- [Falcon Perception](https://github.com/tiiuae/falcon-perception) — TII's open-vocabulary segmentation model
- [Falcon Perception Paper](https://arxiv.org/abs/2603.27365) — arXiv:2603.27365
- [Gemma 4](https://ai.google.dev/gemma/docs/gemma-4) — Google's efficient multimodal model
- [Gemma4-Visual-Agent — CUDA/PyTorch branch](https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10)
- [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) — Vision language models on Apple Silicon
- [mlx-vlm-falcon](https://github.com/korale77/mlx-vlm-falcon) — Inspiration for the combined pipeline
- [Gemma 4 Vision Agent | Object Detection + VLM Pipeline](https://www.youtube.com/watch?v=VFYnD1WREdU&list=PLIsoxylL1dFw&index=6)
- [What makes up a Vision Language Model](https://www.nvidia.com/en-us/glossary/vision-language-models/)

Falcon Perception and the Gemma 4 docs are the primary model references for this repo. The CUDA/PyTorch Gemma4-Visual-Agent branch and `mlx-vlm-falcon` are useful architecture references. `MLX-VLM` is relevant background for the Apple Silicon ecosystem, but MLX is not a deployment target for this project.
