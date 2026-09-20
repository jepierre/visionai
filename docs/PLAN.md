# Grounded Image Chat Agent — Implementation Plan

## Goal

Build a local CUDA/PyTorch web app that reads images from `images/`, lets a user select an image and ask a question in a bottom chat bar, then displays a grounded answer with Falcon Perception masks and/or bounding boxes.

The agent uses:

- **Falcon Perception** for open-vocabulary object detection and instance segmentation.
- **A local Ollama-served Gemma 4B-class VLM** for image reasoning and final answers. The planner is implemented in Python rather than delegated to Gemma.
- **No Apple Silicon or MLX support** in the first release.
- **No OpenCV dependency** unless a later feature has a demonstrated need for it. Pillow, NumPy, and pycocotools cover image loading, annotation rendering, and mask decoding for this scope.

## Target environment

- Windows or Linux host with a CUDA-capable NVIDIA GPU.
- Python 3.10+.
- CUDA-compatible PyTorch and torchvision.
- A local Ollama runtime serving a quantized Gemma 4B-class model.
- Sufficient GPU memory for Falcon Perception and the selected Ollama model; validate this before live inference.

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

**Status (2026-09-18): Partially ready; local inference validation is still pending for live model runs.**

- The detected host is Linux with Python 3.10.12, an NVIDIA GeForce RTX 3070, approximately 8 GiB of GPU memory, and NVIDIA driver 595.58.03.
- The Gemma runtime is intentionally Ollama-only for this project. The repo now targets Ollama over HTTP, usually through the local Docker container on `http://localhost:11434`.
- The sample validation image is `images/dog_running_in_park.jpg`.
- The RTX 3070 is a reasonable fit for a quantized Gemma 4B-class model, but Falcon and Gemma must be tested separately to avoid GPU memory pressure. Larger checkpoints may exceed the available VRAM.

1. Create the Python environment and install CUDA/PyTorch dependencies plus the Falcon Perception PyTorch package.
      **Status:** dependency and CUDA validation remain pending.
2. Install and start Ollama, then pull a local 4B-class Gemma model such as `gemma3:4b`.
      **Status:** blocked until Ollama is installed and the model is available locally; Hugging Face is not the planned runtime path.
3. Add a small script that loads Falcon, runs a known image/object query, and writes normalized detection metadata.
      **Status:** smoke-test script is present; live execution can still fail on hosts missing Python development headers required by Triton.
4. Add a second script that loads Gemma and answers a visual question about the same image.
      **Status:** smoke-test script is present and uses the local Ollama API instead of Hugging Face inference.
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

**Status (2026-09-18): Implemented.**

1. Define `images/` as the initial input folder and support JPG, JPEG, PNG, and WebP.
2. Implement `GET /api/images` to return an ordered image catalog with IDs, names, dimensions, and thumbnail URLs.
3. Implement safe image/thumbnail routes that prevent paths outside `images/`.
4. Build the browser UI with an image gallery, main viewer, selection controls, and empty-folder state.
5. Add a bottom-fixed prompt field and Submit button; submit is unavailable until an image is selected.

Implemented in the current repo:

- `GET /api/images` returns image IDs, names, dimensions, image URLs, and thumbnail URLs.
- Safe image and thumbnail routes are served from the backend.
- The frontend gallery loads from the API, supports selection, and handles the empty-image state.

**Exit criteria:** users can browse all supported images in `images/`, select one, and enter a query.

## Phase 2 — Falcon grounding and annotation

**Status (2026-09-18): Implemented at the app-code level; live Falcon execution remains environment-dependent.**

1. Implement `POST /api/detect` with `image_id`, `object_query`, and annotation mode.
2. Normalize each Falcon result into label, score, bounding box, encoded mask, and count position.
3. Decode COCO RLE masks with pycocotools and render masks, box strokes, and numbered labels with Pillow and NumPy.
4. Return an annotated PNG plus structured detection data to the UI.
5. Add mask, bounding-box, and combined display modes to the image viewer.

Implemented in the current repo:

- `POST /api/detect` accepts `image_id`, `object_query`, and annotation mode.
- Falcon output is normalized into labels, scores, bounding boxes, optional mask area, and count index.
- COCO RLE masks are decoded with `pycocotools` and rendered with Pillow and NumPy.
- The frontend supports mask, box, and combined display modes.

**Exit criteria:** a simple query such as `car` overlays all detected cars with user-selectable masks and/or boxes.

## Phase 3 — Single-turn grounded chat

**Status (2026-09-18): Implemented at the app-code level.**

1. Add `POST /api/chat` accepting an image ID and natural-language query.
2. Route simple scene descriptions and visual questions directly to Gemma.
3. Route detection, counting, and location questions through Falcon first, then pass the annotated image and detection summary to Gemma for the response.
4. Make count comparisons deterministic after detection rather than asking the VLM to calculate them from scratch.
5. Display chat bubbles, final answer, run timing, detections, and the resulting annotated image.

Implemented in the current repo:

- `POST /api/chat` accepts `image_id`, `query`, and annotation mode.
- Direct scene-description prompts route to Ollama.
- Detection, counting, and location-style prompts route through Falcon first.
- Count comparisons are handled deterministically after detection rather than delegated to the VLM.
- The frontend shows chat bubbles, answers, detections, run status, and the latest annotated image.

Current limitation:

- Full end-to-end chat depends on Falcon and Ollama both being available at runtime; code paths are in place, but the environment can still block inference.

**Exit criteria:** “How many people are there?” returns an answer grounded in visible, countable detections.

## Phase 4 — Bounded agentic workflow

**Status (2026-09-18): Implemented at the app-code level; live model execution remains environment-dependent.**

1. Add a planner with explicit actions: `DETECT`, `DETECT_EACH`, `CROP`, `COMPARE`, `VLM`, and `ANSWER`.
2. Keep each plan within a fixed maximum of six actions. Re-planning after a model step is not implemented.
3. Use `CROP` for object-detail questions such as “What color is the largest car?”
4. Surface a compact activity trail in the UI, e.g. “Detecting cars → 9 found → analyzing.”
5. Preserve each message, execution trace, and annotated output within a browser session.

Implemented in the current repo:

- The planner emits explicit bounded actions: `DETECT`, `DETECT_EACH`, `CROP`, `COMPARE`, `VLM`, and `ANSWER`.
- Agent plans are capped at six actions and expose action labels in API traces and the frontend.
- Largest-object detail questions use Falcon grounding, crop the selected bounding box, and send the crop to Ollama.
- The frontend preserves the current run trace, detections, reasoning, final output, and annotated output for the active session.

**Exit criteria:** compound questions use multiple tools and return a traceable answer without an unbounded tool loop.

## Phase 5 — Quality, performance, and delivery

**Status (2026-09-18): Implemented at the app-code level; production queueing and cancellation remain deployment concerns.**

1. Cache Falcon detections by image fingerprint and object query.
2. Add warm-up, an inference queue, request cancellation, and clear model/download failure messages.
3. Add API tests, renderer tests, and a small fixed image/query regression suite.
4. Add export for annotated images and response summaries.
5. Document setup, GPU requirements, model access, and troubleshooting.

Implemented in the current repo:

- Falcon detections are cached by source fingerprint, object query, and annotation mode.
- Falcon model access and inference are serialized per detector instance to avoid concurrent GPU inference.
- `POST /api/warmup` reports Falcon and Ollama readiness.
- Annotated images and JSON run summaries can be downloaded from the frontend.
- Backend API, planner, cache, and renderer unit tests are included under `backend/tests`.
- Setup and GPU troubleshooting are documented in `README.md` and `AGENTS.md`.

Remaining environment/deployment limits:

- The synchronous FastAPI process cannot forcibly cancel an in-flight CUDA kernel; abandoned requests may still finish in the backend.
- A multi-worker production inference queue is not enabled because Falcon model state is intentionally kept in one warm process.

**Exit criteria:** the app is repeatable to install, reliable under normal use, and has a tested demo dataset.

## Proposed project layout

```text
visionai/
├── images/                  # user-provided source images
├── backend/
│   └── app/                 # FastAPI routes, services, models, agent, rendering
├── frontend/                # React, TypeScript, Tailwind, and Vite application
├── docs/                  # plan and workflow documentation
├── backend/requirements.txt # app and validation Python dependencies
├── README.md
└── AGENTS.md
```

## Technology choices

| Concern | Choice | Reason |
| --- | --- | --- |
| Model runtime | PyTorch + CUDA | Matches the NVIDIA-focused implementation path. |
| API | FastAPI + Uvicorn | Typed Python service that can keep models warm in one process. |
| Frontend | React + TypeScript + Vite | Current implementation uses typed React components and an API-driven image viewer/chat UI. |
| Styling | Tailwind CSS | The repo currently uses Tailwind utilities plus a small Tailwind component layer. |
| Image handling | Pillow + NumPy | Sufficient for image IO, compositing, masks, and boxes without OpenCV. |
| Mask format | pycocotools | Decodes Falcon’s COCO RLE segmentation masks. |
| Validation | Python `unittest` + Vite build | Covers backend behavior and frontend compilation; browser E2E coverage is not currently checked in. |

## Out of scope for version 1

- Apple Silicon / MLX compatibility.
- Video analysis and tracking.
- Cloud deployment, multi-user accounts, and persistent shared history.
- Image editing or annotation authoring.
- OpenCV-based processing.
