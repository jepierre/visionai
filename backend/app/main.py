from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.app.agent.chat import ChatOrchestrator
from backend.app.config import get_settings
from backend.app.models.falcon import FalconDetector, FalconUnavailableError
from backend.app.models.ollama import OllamaUnavailableError, OllamaVisionClient
from backend.app.rendering.annotations import AnnotationRenderer
from backend.app.schemas import (
    ChatRequest,
    ChatResponse,
    DetectRequest,
    DetectResponse,
    DetectionRecord,
    ImageCatalogResponse,
    OllamaModelListResponse,
    OllamaModelSummary,
)
from backend.app.services.images import ImageCatalogService

settings = get_settings()
image_catalog = ImageCatalogService(settings)
renderer = AnnotationRenderer(settings)
falcon_detector = FalconDetector(settings, renderer)
ollama_client = OllamaVisionClient(settings)
chat_orchestrator = ChatOrchestrator(falcon_detector, ollama_client, renderer)

app = FastAPI(title="VisionAI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/images", response_model=ImageCatalogResponse)
def list_images() -> ImageCatalogResponse:
    return ImageCatalogResponse(images=image_catalog.list_images())


@app.get("/api/ollama/models", response_model=OllamaModelListResponse)
def list_ollama_models() -> OllamaModelListResponse:
    try:
        models = ollama_client.list_models()
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return OllamaModelListResponse(
        models=[OllamaModelSummary(name=name) for name in models],
        default_model=ollama_client.default_model,
    )


@app.get("/api/images/{image_id}/file")
def get_image_file(image_id: str) -> FileResponse:
    try:
        record = image_catalog.get_image(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(record.path)


@app.get("/api/images/{image_id}/thumbnail/{thumbnail_name}")
def get_thumbnail(image_id: str, thumbnail_name: str) -> FileResponse:
    try:
        expected_path = image_catalog.ensure_thumbnail(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if thumbnail_name != expected_path.name:
        raise HTTPException(status_code=404, detail="Thumbnail does not match the requested image")
    return FileResponse(expected_path)


@app.get("/api/annotated/{file_name}")
def get_annotated_image(file_name: str) -> FileResponse:
    path = settings.annotated_dir / Path(file_name).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Annotated image not found")
    return FileResponse(path)


@app.post("/api/detect", response_model=DetectResponse)
def detect_objects(request: DetectRequest) -> DetectResponse:
    try:
        record = image_catalog.get_image(request.image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        run = falcon_detector.detect(record.path, request.object_query, request.annotation_mode)
    except FalconUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    detections = [
        DetectionRecord(
            label=item.label,
            score=item.score,
            bbox=item.bbox,
            mask_area=item.mask_area,
            count_index=item.count_index,
        )
        for item in run.detections
    ]
    annotated_image_url = f"/api/annotated/{run.annotated_file_name}" if run.annotated_file_name else None
    return DetectResponse(
        image_id=request.image_id,
        object_query=request.object_query,
        annotation_mode=request.annotation_mode,
        original_image_url=f"/api/images/{request.image_id}/file",
        annotated_image_url=annotated_image_url,
        detections=detections,
        execution_mode="falcon",
        models_used=[falcon_detector.model_info()],
        trace=run.trace,
        reasoning=run.reasoning,
        final_output=run.final_output,
        message=run.message,
        timings=run.timings,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        record = image_catalog.get_image(request.image_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        run = chat_orchestrator.answer(
            record.path,
            request.query,
            request.annotation_mode,
            execution_mode=request.execution_mode,
            object_query=request.object_query,
            ollama_model=request.ollama_model,
        )
    except FalconUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    detections = [
        DetectionRecord(
            label=item.label,
            score=item.score,
            bbox=item.bbox,
            mask_area=item.mask_area,
            count_index=item.count_index,
        )
        for item in run.detections
    ]
    annotated_image_url = f"/api/annotated/{run.annotated_file_name}" if run.annotated_file_name else None
    return ChatResponse(
        image_id=request.image_id,
        query=request.query,
        answer=run.answer,
        route=run.route,
        execution_mode=run.execution_mode,
        original_image_url=f"/api/images/{request.image_id}/file",
        annotated_image_url=annotated_image_url,
        detections=detections,
        models_used=run.models_used,
        trace=run.trace,
        reasoning=run.reasoning,
        final_output=run.final_output,
        message=run.message,
        timings=run.timings,
    )