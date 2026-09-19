from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PIL import Image

from backend.app.config import Settings
from backend.app.rendering.annotations import AnnotationRenderer, DetectedObject
from backend.app.schemas import ModelInfo, TraceStep


class FalconUnavailableError(RuntimeError):
    pass


@dataclass
class DetectionRun:
    object_query: str
    detections: list[DetectedObject]
    annotated_file_name: str | None
    timings: dict[str, float]
    model_name: str
    trace: list[TraceStep]
    reasoning: str
    final_output: str
    message: str | None = None


class FalconDetector:
    def __init__(self, settings: Settings, renderer: AnnotationRenderer):
        self._settings = settings
        self._renderer = renderer
        self._model = None
        self._tokenizer = None
        self._model_args = None
        self._torch = None
        self._batch_inference_engine = None
        self._build_prompt_for_task = None
        self._process_batch_and_generate = None

    def detect(self, image_path: Path, object_query: str, annotation_mode: str, render: bool = True) -> DetectionRun:
        model, tokenizer, model_args, torch = self._ensure_model()

        trace = [
            TraceStep(title="Load Falcon context", detail=f"Prepare Falcon for query {object_query!r}.", model=self._settings.falcon_model_id),
            TraceStep(title="Open image", detail=f"Load source image {image_path.name}.", model=None),
        ]

        with Image.open(image_path) as source:
            image = source.convert("RGB")

        started = perf_counter()
        try:
            batch = self._process_batch_and_generate(
                tokenizer,
                [(image, self._build_prompt_for_task(object_query, "segmentation"))],
                max_length=getattr(model_args, "max_seq_len", None) or 4096,
                min_dimension=256,
                max_dimension=1024,
                patch_size=getattr(model_args, "spatial_patch_size", 16),
            )
            batch = {key: value.to(model.device) if torch.is_tensor(value) else value for key, value in batch.items()}
            stop_ids = [tokenizer.eos_token_id]
            if getattr(tokenizer, "end_of_query_token_id", None) is not None:
                stop_ids.append(tokenizer.end_of_query_token_id)

            _, auxiliary = self._batch_inference_engine(model, tokenizer).generate(
                **batch,
                max_new_tokens=100,
                temperature=0.0,
                stop_token_ids=stop_ids,
                seed=42,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise FalconUnavailableError(
                "Falcon ran out of GPU memory. Stop other GPU processes or reduce GPU memory pressure, then retry."
            ) from exc

        detections = self._normalize_detections(auxiliary[0], object_query, image.size)
        annotated_file_name = self._renderer.render(image_path, detections, annotation_mode) if render else None
        message = None if detections else f"No detections found for {object_query!r}."
        trace.append(
            TraceStep(
                title="Run Falcon inference",
                detail=f"Falcon produced {len(detections)} detection(s) for {object_query!r}.",
                model=self._settings.falcon_model_id,
            )
        )
        if render:
            trace.append(
                TraceStep(
                    title="Render overlays",
                    detail=(
                        f"Rendered {annotation_mode} overlay to {annotated_file_name}."
                        if annotated_file_name
                        else "No overlay generated because there were no detections."
                    ),
                    model=None,
                )
            )
        final_output = f"Falcon detected {len(detections)} match(es) for {object_query!r}."
        return DetectionRun(
            object_query=object_query,
            detections=detections,
            annotated_file_name=annotated_file_name,
            message=message,
            model_name=self._settings.falcon_model_id,
            trace=trace,
            reasoning=f"Falcon-only mode ran segmentation for {object_query!r} and returned {len(detections)} detection(s).",
            final_output=final_output,
            timings={"inference_seconds": perf_counter() - started},
        )

    def model_info(self) -> ModelInfo:
        return ModelInfo(name=self._settings.falcon_model_id, role="Grounding and segmentation")

    def _ensure_model(self):
        if self._model is not None:
            return self._model, self._tokenizer, self._model_args, self._torch

        try:
            import torch
            from falcon_perception import build_prompt_for_task, load_and_prepare_model, setup_torch_config
            from falcon_perception.batch_inference import BatchInferenceEngine, process_batch_and_generate
        except ImportError as exc:
            raise FalconUnavailableError("Falcon Perception dependencies are not installed in the current environment.") from exc

        if not torch.cuda.is_available():
            raise FalconUnavailableError("CUDA is required for Falcon Perception.")

        setup_torch_config()
        started = perf_counter()
        try:
            model, tokenizer, model_args = load_and_prepare_model(
                hf_model_id=self._settings.falcon_model_id,
                hf_revision=self._settings.falcon_revision,
                device=self._settings.cuda_device,
                dtype=self._settings.falcon_dtype,
                compile=False,
            )
        except Exception as exc:
            raise FalconUnavailableError(str(exc)) from exc

        self._model = model
        self._tokenizer = tokenizer
        self._model_args = model_args
        self._torch = torch
        self._batch_inference_engine = BatchInferenceEngine
        self._build_prompt_for_task = build_prompt_for_task
        self._process_batch_and_generate = process_batch_and_generate
        os.environ.setdefault("VISIONAI_FALCON_LOAD_SECONDS", f"{perf_counter() - started:.3f}")
        return self._model, self._tokenizer, self._model_args, self._torch

    def _normalize_detections(
        self,
        auxiliary: object,
        object_query: str,
        image_size: tuple[int, int] | None = None,
    ) -> list[DetectedObject]:
        raw_bboxes = list(getattr(auxiliary, "bboxes_raw", []) or [])
        bboxes = self._pair_bbox_entries(raw_bboxes)
        masks = list(getattr(auxiliary, "masks_rle", []) or [])
        scores = list(getattr(auxiliary, "scores", []) or getattr(auxiliary, "scores_raw", []) or [])
        # Segmentation masks represent the instance results. Falcon can emit
        # extra raw box candidates while finalizing a segmentation query.
        total = len(masks) if masks else len(bboxes)
        detections: list[DetectedObject] = []
        for index in range(total):
            bbox = self._normalize_bbox(bboxes[index], image_size) if index < len(bboxes) else None
            mask_rle = self._normalize_mask(masks[index]) if index < len(masks) else None
            detections.append(
                DetectedObject(
                    label=object_query,
                    score=self._normalize_score(scores[index]) if index < len(scores) else None,
                    bbox=bbox,
                    mask_rle=mask_rle,
                    mask_area=self._mask_area(mask_rle),
                    count_index=index + 1,
                )
            )
        return detections

    @staticmethod
    def _pair_bbox_entries(raw_bboxes: list[object]) -> list[dict]:
        paired: list[dict] = []
        current: dict = {}
        for entry in raw_bboxes:
            if not isinstance(entry, dict):
                continue
            current.update(entry)
            if {"x", "y", "h", "w"}.issubset(current):
                paired.append(dict(current))
                current = {}
        return paired

    @staticmethod
    def _normalize_score(raw_score: object) -> float | None:
        try:
            return float(raw_score)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_bbox(raw_bbox: object, image_size: tuple[int, int] | None = None) -> list[float] | None:
        values: list[float] | None = None
        if isinstance(raw_bbox, dict):
            if {"x1", "y1", "x2", "y2"}.issubset(raw_bbox):
                values = [raw_bbox["x1"], raw_bbox["y1"], raw_bbox["x2"], raw_bbox["y2"]]
            elif {"x", "y", "w", "h"}.issubset(raw_bbox):
                x, y, width, height = [float(raw_bbox[key]) for key in ("x", "y", "w", "h")]
                if image_size and max(abs(x), abs(y), abs(width), abs(height)) <= 1.0:
                    image_width, image_height = image_size
                    half_width = width * image_width / 2
                    half_height = height * image_height / 2
                    return [
                        (x * image_width) - half_width,
                        (y * image_height) - half_height,
                        (x * image_width) + half_width,
                        (y * image_height) + half_height,
                    ]
                values = [x, y, x + width, y + height]
        elif hasattr(raw_bbox, "tolist"):
            values = raw_bbox.tolist()
        elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
            values = list(raw_bbox[:4])
        if values is None:
            return None
        x1, y1, third, fourth = [float(value) for value in values]
        if third <= x1 or fourth <= y1:
            x2 = x1 + max(third, 0.0)
            y2 = y1 + max(fourth, 0.0)
            return [x1, y1, x2, y2]
        return [x1, y1, third, fourth]

    @staticmethod
    def _normalize_mask(raw_mask: object) -> dict | None:
        if isinstance(raw_mask, dict):
            return raw_mask
        return None

    @staticmethod
    def _mask_area(mask_rle: dict | None) -> int | None:
        if not mask_rle:
            return None
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        decoded = mask_utils.decode(mask_rle)
        if decoded.ndim == 3:
            decoded = decoded[..., 0]
        return int(decoded.astype(bool).sum())