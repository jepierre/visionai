from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


ExecutionMode = Literal["agent", "falcon", "gemma"]


class ImageSummary(BaseModel):
    id: str
    name: str
    width: int
    height: int
    image_url: str
    thumbnail_url: str


class ImageCatalogResponse(BaseModel):
    images: list[ImageSummary]


class OllamaModelSummary(BaseModel):
    name: str


class OllamaModelListResponse(BaseModel):
    models: list[OllamaModelSummary]
    default_model: str


class ErrorResponse(BaseModel):
    detail: str


class DetectRequest(BaseModel):
    image_id: str
    object_query: str = Field(min_length=1)
    annotation_mode: Literal["mask", "box", "combined"] = "combined"


class ModelInfo(BaseModel):
    name: str
    role: str


class TraceStep(BaseModel):
    title: str
    detail: str
    model: str | None = None


class DetectionRecord(BaseModel):
    label: str
    score: float | None = None
    bbox: list[float] | None = None
    mask_area: int | None = None
    count_index: int


class DetectResponse(BaseModel):
    image_id: str
    object_query: str
    annotation_mode: str
    original_image_url: str
    annotated_image_url: str | None = None
    detections: list[DetectionRecord]
    execution_mode: ExecutionMode = "falcon"
    models_used: list[ModelInfo] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    reasoning: str | None = None
    final_output: str | None = None
    message: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    image_id: str
    query: str = Field(min_length=1)
    annotation_mode: Literal["mask", "box", "combined"] = "combined"
    execution_mode: Literal["agent", "gemma"] = "agent"
    object_query: str | None = None
    ollama_model: str | None = None


class ChatResponse(BaseModel):
    image_id: str
    query: str
    answer: str
    route: str
    execution_mode: Literal["agent", "gemma"]
    original_image_url: str
    annotated_image_url: str | None = None
    detections: list[DetectionRecord] = Field(default_factory=list)
    models_used: list[ModelInfo] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
    reasoning: str | None = None
    final_output: str | None = None
    message: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)