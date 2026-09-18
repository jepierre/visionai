# Grounded Image Chat Agent — Implementation Plan

## Goal

Build a local CUDA/PyTorch web app that reads images from `images/`, lets a user select an image and ask a question in a bottom chat bar, then displays a grounded answer with Falcon Perception masks and/or bounding boxes.

The agent uses:

- **Falcon Perception** for open-vocabulary object detection and instance segmentation.
- **Gemma 4 E4B instruction-tuned VLM** for query planning, image reasoning, and final answers.
- **No Apple Silicon or MLX support** in the first release.
- **No OpenCV dependency** unless a later feature has a demonstrated need for it. Pillow, NumPy, and pycocotools cover image loading, annotation rendering, and mask decoding for this scope.

## Target environment

- Windows or Linux host with a CUDA-capable NVIDIA GPU.
- Python 3.10+.
- CUDA-compatible PyTorch and torchvision.
- A local Ollama runtime serving a quantized Gemma 4B-class model.
- Sufficient GPU memory for the selected Gemma 4 checkpoint and Falcon Perception; validate this before UI work begins.

## Architecture

```text
images/ directory
       |
       v
 FastAPI image catalog <--------------------- React + TypeScript UI
       |                                      - gallery / selected image
       |                                      - bottom query bar + submit
       v                                      - chat results / run status
 Agent orchestrator                           - mask / box visibility toggles
       |
       +--> Gemma planner / VLM response
       |
       +--> Falcon Perception -> masks, boxes, counts
       |
       v
 Pillow renderer -> annotated PNG + structured detections
```

The Python service owns models and inference. The browser never accesses the filesystem directly; it works through an image catalog API limited to `images/`.

## Phase 0 — Validate the local inference stack

**Status (2026-09-18): Partially ready; local inference validation is pending.**

- The detected host is Linux with Python 3.10.12, an NVIDIA GeForce RTX 3070, approximately 8 GiB of GPU memory, and NVIDIA driver 595.58.03.
- The repository `.venv` exists, but the CUDA/PyTorch and Falcon dependency installation still needs to be validated.
- The Gemma runtime is intentionally Ollama-only for this project. Ollama is not currently available in `PATH`, and a local 4B-class model has not yet been pulled.
- The sample validation image is `images/dog_running_in_park.jpg`.
- The RTX 3070 is a reasonable fit for a quantized Gemma 4B-class model, but Falcon and Gemma must be tested separately to avoid GPU memory pressure. Larger checkpoints may exceed the available VRAM.

1. Create the Python environment and install CUDA/PyTorch dependencies plus the Falcon Perception PyTorch package.
      **Status:** `.venv` has been created; dependency and CUDA validation remain pending.
2. Install and start Ollama, then pull a local 4B-class Gemma model such as `gemma3:4b`.
      **Status:** blocked until Ollama is installed and the model is available locally; Hugging Face is not the planned runtime path.
3. Add a small script that loads Falcon, runs a known image/object query, and writes normalized detection metadata.
      **Status:** smoke-test script is present; execution remains pending dependency validation.
4. Add a second script that loads Gemma and answers a visual question about the same image.
      **Status:** smoke-test script is present; execution remains pending Ollama setup.
5. Record GPU name, VRAM, load times, and inference timings in the developer notes.
      **Status:** hardware details are recorded above; model load and inference timings remain pending.

**Next validation commands:**

```bash
source .venv/bin/activate
ollama serve
ollama pull gemma3:4b
python scripts/validate_environment.py
python scripts/smoke_falcon.py --image images/dog_running_in_park.jpg --query dog
python scripts/smoke_gemma.py --image images/dog_running_in_park.jpg --model gemma3:4b
```

If the local Ollama registry uses a different 4B model tag, pass it with `--model` or set `OLLAMA_MODEL`.

**Exit criteria:** both models load locally, Falcon emits valid detections, and Gemma returns an answer for a test image.

## Phase 1 — Image catalog and viewer

1. Define `images/` as the initial input folder and support JPG, JPEG, PNG, and WebP.
2. Implement `GET /api/images` to return an ordered image catalog with IDs, names, dimensions, and thumbnail URLs.
3. Implement safe image/thumbnail routes that prevent paths outside `images/`.
4. Build the browser UI with an image gallery, main viewer, selection controls, and empty-folder state.
5. Add a bottom-fixed prompt field and Submit button; submit is unavailable until an image is selected.

**Exit criteria:** users can browse all supported images in `images/`, select one, and enter a query.

## Phase 2 — Falcon grounding and annotation

1. Implement `POST /api/detect` with `image_id`, `object_query`, and annotation mode.
2. Normalize each Falcon result into label, score, bounding box, encoded mask, and count position.
3. Decode COCO RLE masks with pycocotools and render masks, box strokes, and numbered labels with Pillow and NumPy.
4. Return an annotated PNG plus structured detection data to the UI.
5. Add mask, bounding-box, and combined display modes to the image viewer.

**Exit criteria:** a simple query such as `car` overlays all detected cars with user-selectable masks and/or boxes.

## Phase 3 — Single-turn grounded chat

1. Add `POST /api/chat` accepting an image ID and natural-language query.
2. Route simple scene descriptions and visual questions directly to Gemma.
3. Route detection, counting, and location questions through Falcon first, then pass the annotated image and detection summary to Gemma for the response.
4. Make count comparisons deterministic after detection rather than asking the VLM to calculate them from scratch.
5. Display chat bubbles, final answer, run timing, detections, and the resulting annotated image.

**Exit criteria:** “How many people are there?” returns an answer grounded in visible, countable detections.

## Phase 4 — Bounded agentic workflow

1. Add a planner with explicit actions: `DETECT`, `DETECT_EACH`, `CROP`, `COMPARE`, `VLM`, and `ANSWER`.
2. Permit re-planning after a model step, with a fixed maximum of 4–8 actions.
3. Use `CROP` for object-detail questions such as “What color is the largest car?”
4. Surface a compact activity trail in the UI, e.g. “Detecting cars → 9 found → analyzing.”
5. Preserve each message, execution trace, and annotated output within a browser session.

**Exit criteria:** compound questions use multiple tools and return a traceable answer without an unbounded tool loop.

## Phase 5 — Quality, performance, and delivery

1. Cache Falcon detections by image fingerprint and object query.
2. Add warm-up, an inference queue, request cancellation, and clear model/download failure messages.
3. Add API tests, renderer tests, and a small fixed image/query regression suite.
4. Add export for annotated images and response summaries.
5. Document setup, GPU requirements, model access, and troubleshooting.

**Exit criteria:** the app is repeatable to install, reliable under normal use, and has a tested demo dataset.

## Proposed project layout

```text
visionai/
├── images/                  # user-provided source images
├── backend/
│   ├── app/                 # FastAPI routes and services
│   ├── models/              # Falcon and Gemma adapters
│   ├── agent/               # planner and tool implementations
│   ├── rendering/           # Pillow/NumPy annotation renderer
│   └── tests/
├── frontend/                # React/Vite application
├── output/                  # generated annotations; gitignored
├── PLAN.md
└── README.md
```

## Technology choices

| Concern | Choice | Reason |
| --- | --- | --- |
| Model runtime | PyTorch + CUDA | Matches the NVIDIA-focused implementation path. |
| API | FastAPI + Uvicorn | Typed Python service that can keep models warm in one process. |
| Frontend | React + TypeScript + Vite | Responsive image viewer and durable UI structure. |
| Styling | Tailwind CSS | Fast layout of the gallery, canvas, chat bar, and status UI. |
| Image handling | Pillow + NumPy | Sufficient for image IO, compositing, masks, and boxes without OpenCV. |
| Mask format | pycocotools | Decodes Falcon’s COCO RLE segmentation masks. |
| Validation | Pytest + Playwright | Covers the backend pipeline and user-facing browser flow. |

## Out of scope for version 1

- Apple Silicon / MLX compatibility.
- Video analysis and tracking.
- Cloud deployment, multi-user accounts, and persistent shared history.
- Image editing or annotation authoring.
- OpenCV-based processing.
