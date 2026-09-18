from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImageSummary(BaseModel):
    id: str
    name: str
    width: int
    height: int
    image_url: str
    thumbnail_url: str


class ImageCatalogResponse(BaseModel):
    images: list[ImageSummary]


class ErrorResponse(BaseModel):
    detail: str


class DetectRequest(BaseModel):
    image_id: str
    object_query: str = Field(min_length=1)
    annotation_mode: Literal["mask", "box", "combined"] = "combined"


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
    annotated_image_url: str | None = None
    detections: list[DetectionRecord]
    message: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)


class ChatRequest(BaseModel):
    image_id: str
    query: str = Field(min_length=1)
    annotation_mode: Literal["mask", "box", "combined"] = "combined"


class ChatResponse(BaseModel):
    image_id: str
    query: str
    answer: str
    route: str
    annotated_image_url: str | None = None
    detections: list[DetectionRecord] = Field(default_factory=list)
    message: str | None = None
    timings: dict[str, float] = Field(default_factory=dict)