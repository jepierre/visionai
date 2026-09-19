# VisionAI Agentic Workflow

This document describes the current Phase 1-5 application flow: what runs in the frontend, what runs in the backend, the API boundaries, and the bounded action loop inside the backend orchestrator.

## System split

```mermaid
flowchart LR
    subgraph FE[Frontend: React + Vite]
        Gallery[Load image catalog]
        Viewer[Select image and viewer mode]
        DetectForm[Run detection form]
        ChatForm[Submit grounded chat question]
        Results[Render actions, detections, answer, and annotated image]
        Export[Download image or JSON summary]
    end

    subgraph BE[Backend: FastAPI]
        ImagesAPI[GET /api/images]
        ImageFile[GET /api/images/:id/file]
        ThumbAPI[GET /api/images/:id/thumbnail/:name]
        DetectAPI[POST /api/detect]
        ChatAPI[POST /api/chat]
        WarmupAPI[POST /api/warmup]
        AnnotatedAPI[GET /api/annotated/:file]
    end

    subgraph Data[Local data and model services]
        ImagesDir[images/ directory]
        OutputDir[output/ annotated images and thumbnails]
        Falcon[Falcon Perception]
        Ollama[Ollama HTTP API]
    end

    Gallery --> ImagesAPI
    Viewer --> ImageFile
    Viewer --> ThumbAPI
    DetectForm --> DetectAPI
    ChatForm --> ChatAPI
    DetectAPI --> AnnotatedAPI
    ChatAPI --> AnnotatedAPI
    AnnotatedAPI --> Results
    Results --> Export

    ImagesAPI --> ImagesDir
    ImageFile --> ImagesDir
    ThumbAPI --> ImagesDir
    ThumbAPI --> OutputDir
    DetectAPI --> Falcon
    DetectAPI --> OutputDir
    ChatAPI --> Falcon
    ChatAPI --> Ollama
    ChatAPI --> OutputDir
```

## Current request and response paths

```mermaid
sequenceDiagram
    autonumber
    participant UI as Frontend UI
    participant API as FastAPI backend
    participant IMG as Image catalog service
    participant ORCH as Chat orchestrator
    participant FAL as Falcon detector
    participant REN as Annotation renderer
    participant OLL as Ollama /api/generate

    UI->>API: GET /api/images
    API->>IMG: scan images/ and build thumbnails
    IMG-->>API: image summaries
    API-->>UI: image catalog

    UI->>API: POST /api/detect
    API->>FAL: detect(image_id, object_query, annotation_mode)
    FAL->>REN: render masks and or boxes
    REN-->>API: annotated file name
    API-->>UI: detections + annotated_image_url

    UI->>API: POST /api/chat
    API->>ORCH: answer(image_id, query, annotation_mode)

    alt direct visual question
        ORCH->>OLL: generate(query, image)
        OLL-->>ORCH: answer
    else count question
        ORCH->>FAL: detect(single object)
        FAL-->>ORCH: detections
        ORCH-->>API: deterministic count answer
    else comparison question
        loop once per compared object
            ORCH->>FAL: detect(object)
            FAL-->>ORCH: detections
        end
        ORCH->>REN: render merged annotations
        ORCH-->>API: deterministic comparison answer
    else location or find question
        ORCH->>FAL: detect(single object)
        FAL->>REN: render annotations
        FAL-->>ORCH: detections + annotated file
        ORCH->>OLL: generate(grounded prompt, image)
        OLL-->>ORCH: answer
    end

    API-->>UI: answer + route + detections + annotated_image_url
```

## Current bounded agent loop

The current implementation is a bounded single-request action loop inside the backend. Every plan has an explicit action list and is capped at six actions.

```mermaid
flowchart TD
    Start[Receive POST /api/chat] --> Plan[Inspect natural-language query]
    Plan --> Compare{Contains more X than Y?}
    Compare -- Yes --> DetectA[Detect first object with Falcon]
    DetectA --> DetectB[Detect second object with Falcon]
    DetectB --> Merge[Merge detections and render annotation]
    Merge --> CompareAnswer[Return deterministic comparison answer]

    Compare -- No --> Count{Contains how many ... ?}
    Count -- Yes --> CountDetect[Detect one object with Falcon]
    CountDetect --> CountAnswer[Return deterministic count answer]

    Count -- No --> Locate{Contains show, find, detect, locate, where, highlight?}
    Locate -- Yes --> GroundDetect[Detect one object with Falcon]
    GroundDetect --> GroundPrompt[Build grounded prompt from detections]
    GroundPrompt --> GroundVLM[Ask Ollama for final answer]
    GroundVLM --> GroundAnswer[Return grounded answer]

    Locate -- No --> DirectVLM[Ask Ollama directly with image and user question]
    DirectVLM --> DirectAnswer[Return visual answer]
```

## Frontend responsibilities

- Load the image catalog from `GET /api/images` on startup.
- Let the user select an image and annotation mode.
- Call `POST /api/detect` for explicit object detection requests.
- Call `POST /api/chat` for natural-language questions.
- Display the selected image, the latest annotated image, detections, status messages, errors, and chat history.

## Backend responsibilities

- Restrict image access to the repo `images/` directory.
- Generate and serve thumbnails.
- Normalize Falcon output into a stable API response shape.
- Render annotated PNG files into `output/annotated`.
- Decide whether a chat question should go directly to Ollama or through Falcon first.
- Answer count and count-comparison prompts deterministically after detection.

## Current external calls

- Frontend to backend:
  - `GET /api/images`
  - `GET /api/images/{image_id}/file`
  - `GET /api/images/{image_id}/thumbnail/{thumbnail_name}`
  - `POST /api/detect`
  - `POST /api/chat`
  - `GET /api/annotated/{file_name}`
- Backend to Ollama:
  - `POST {OLLAMA_BASE_URL}/api/generate`
- Backend to Falcon:
  - in-process Python model load and inference through the Falcon Perception package

## Notes on scope

- The implemented loop is Phase 3 routing, not the Phase 4 multi-step replanning agent yet.
- There is no browser-side filesystem access; all image access goes through the backend.
- Falcon and Ollama are still sensitive to local runtime availability, GPU memory, and host dependencies.