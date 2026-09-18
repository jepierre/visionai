# VisionAI — Local Grounded Image Chat

VisionAI is a planned local web application for asking questions about images stored in the `images/` directory. It will combine Falcon Perception’s object masks and bounding boxes with Gemma 4’s visual reasoning, so answers can be tied to visible evidence.

## Planned user experience

1. Place images in `images/`.
2. Open the local app and select an image from the gallery.
3. Enter a question in the prompt bar at the bottom and select **Submit**.
4. The agent determines whether it needs detection, visual reasoning, or both.
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
| Gemma 4 E4B instruction-tuned VLM | Understands the user question, plans tool use, inspects images, and writes the final response. |

Falcon provides spatial grounding; Gemma provides the conversational reasoning layer.

## Target platform

Version 1 is CUDA/PyTorch only.

- NVIDIA GPU with CUDA support.
- Python 3.10 or later.
- CUDA-compatible PyTorch and torchvision.
- Access to the Falcon Perception and instruction-tuned Gemma 4 model downloads.
- A GPU with enough VRAM for both local inference workloads; validate this during setup.

Apple Silicon and MLX support are intentionally not included in the first version.

## Planned stack

- **Backend:** Python, FastAPI, Uvicorn, PyTorch/CUDA.
- **Frontend:** React, TypeScript, Vite, Tailwind CSS.
- **Annotation rendering:** Pillow, NumPy, pycocotools.
- **Testing:** Pytest and Playwright.

OpenCV is not part of the planned dependency set. The initial scope only needs Pillow and NumPy for image loading and mask/box overlays. We will reconsider it only if a future capability specifically requires it.

## Phase 0: local validation

Phase 0 is implemented as a CUDA validation harness. It creates an isolated environment, checks GPU readiness, and has separate smoke tests for each model. The tests intentionally load one model at a time.

```powershell
Copy-Item .env.example .env
# Authenticate first if the Gemma model requires approval for your account.
huggingface-cli login

.\scripts\bootstrap.ps1
.\.venv\Scripts\python.exe .\scripts\validate_environment.py
.\.venv\Scripts\python.exe .\scripts\smoke_falcon.py --image .\images\dog_running_in_park.jpg --query dog
.\.venv\Scripts\python.exe .\scripts\smoke_gemma.py --image .\images\dog_running_in_park.jpg
```

The detected local hardware is an RTX 4050 Laptop GPU with 6 GiB VRAM. That is likely sufficient to test Falcon, but unlikely to fit the referenced Gemma 4 E4B CUDA checkpoint at its native precision. The Gemma smoke test is included to measure the real outcome; if it fails for memory, Phase 0 will recommend either a supported quantized CUDA model path or a GPU with more VRAM before building the agent UI.

See [PHASE0_REPORT.md](PHASE0_REPORT.md) for the recorded local hardware result and the Phase 0 acceptance status.

## Repository structure

```text
images/       Source images for the first version
PLAN.md       Phased implementation plan
README.md     Project overview and setup direction
```

The backend, frontend, generated-output, and test directories will be introduced in the implementation phases described in [PLAN.md](PLAN.md).

## Reference projects

- [Gemma4-Visual-Agent — CUDA/PyTorch branch](https://github.com/PromtEngineer/Gemma4-Visual-Agent/tree/dgx-spark-gb10)
- [mlx-vlm-falcon — pipeline reference](https://github.com/korale77/mlx-vlm-falcon)
- [Gemma 7 Vision Agent | Object Detection + VLM Pipeline](https://www.youtube.com/watch?v=VFYnD1WREdU&list=PLIsoxylL1dFw&index=6)

The first repository provides the relevant CUDA/PyTorch direction. The second is useful for its clear extract → detect/visualize → answer pipeline, but it targets Apple Silicon/MLX and is not a deployment target for this project.
