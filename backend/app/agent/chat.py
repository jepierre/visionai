from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from backend.app.models.falcon import DetectionRun, FalconDetector
from backend.app.models.ollama import OllamaUnavailableError, OllamaVisionClient
from backend.app.rendering.annotations import AnnotationRenderer, DetectedObject


@dataclass
class ChatRun:
    answer: str
    route: str
    annotated_file_name: str | None
    detections: list[DetectedObject]
    timings: dict[str, float]
    message: str | None = None


@dataclass
class QueryPlan:
    route: str
    object_queries: list[str]


class ChatOrchestrator:
    def __init__(self, detector: FalconDetector, ollama: OllamaVisionClient, renderer: AnnotationRenderer):
        self._detector = detector
        self._ollama = ollama
        self._renderer = renderer

    def answer(self, image_path: Path, query: str, annotation_mode: str) -> ChatRun:
        plan = self._plan(query)
        if plan.route == "vlm":
            result = self._ollama.generate(query, image_path)
            return ChatRun(
                answer=result.answer,
                route="vlm",
                annotated_file_name=None,
                detections=[],
                timings={"ollama_seconds": result.duration_seconds},
            )

        if plan.route == "compare_counts":
            return self._compare_counts(image_path, query, plan.object_queries, annotation_mode)

        detection = self._detector.detect(image_path, plan.object_queries[0], annotation_mode)
        if plan.route == "count":
            count = len(detection.detections)
            target = plan.object_queries[0]
            noun = target if count == 1 else self._pluralize(target)
            return ChatRun(
                answer=f"I found {count} {noun} in the image.",
                route="detect_count",
                annotated_file_name=detection.annotated_file_name,
                detections=detection.detections,
                timings=detection.timings,
                message=detection.message,
            )

        prompt = self._grounded_prompt(query, detection)
        try:
            ollama_result = self._ollama.generate(prompt, image_path)
            answer = ollama_result.answer
            timings = {**detection.timings, "ollama_seconds": ollama_result.duration_seconds}
            message = detection.message
        except OllamaUnavailableError:
            answer = self._fallback_detection_answer(query, detection)
            timings = detection.timings
            message = "Falcon completed, but Ollama is unavailable."
        return ChatRun(
            answer=answer,
            route="detect_then_vlm",
            annotated_file_name=detection.annotated_file_name,
            detections=detection.detections,
            timings=timings,
            message=message,
        )

    def _compare_counts(self, image_path: Path, query: str, object_queries: list[str], annotation_mode: str) -> ChatRun:
        runs: list[DetectionRun] = [
            self._detector.detect(image_path, object_query, annotation_mode, render=False) for object_query in object_queries
        ]
        counts = {run.object_query: len(run.detections) for run in runs}
        all_detections = self._merge_labeled_detections(runs)
        annotated_file_name = self._renderer.render(image_path, all_detections, annotation_mode)
        first, second = object_queries
        count_a = counts[first]
        count_b = counts[second]
        if count_a > count_b:
            answer = f"Yes. I found {count_a} {self._pluralize(first)} and {count_b} {self._pluralize(second)}."
        elif count_a < count_b:
            answer = f"No. I found {count_a} {self._pluralize(first)} and {count_b} {self._pluralize(second)}."
        else:
            answer = f"They are tied. I found {count_a} {self._pluralize(first)} and {count_b} {self._pluralize(second)}."
        timings = {f"{run.object_query}_seconds": run.timings.get("inference_seconds", 0.0) for run in runs}
        message = next((run.message for run in runs if run.message), None)
        return ChatRun(
            answer=answer,
            route="compare_counts",
            annotated_file_name=annotated_file_name,
            detections=all_detections,
            timings=timings,
            message=message,
        )

    def _grounded_prompt(self, query: str, detection: DetectionRun) -> str:
        lines = [
            "Answer the user's question using the detection summary as grounding.",
            f"User question: {query}",
            f"Detected label: {detection.object_query}",
            f"Detected count: {len(detection.detections)}",
        ]
        for item in detection.detections[:20]:
            parts = [f"{item.count_index}. label={item.label}"]
            if item.bbox:
                parts.append(f"bbox={','.join(f'{value:.1f}' for value in item.bbox)}")
            if item.score is not None:
                parts.append(f"score={item.score:.3f}")
            if item.mask_area is not None:
                parts.append(f"mask_area={item.mask_area}")
            lines.append("; ".join(parts))
        if not detection.detections:
            lines.append("No matching detections were found.")
        return "\n".join(lines)

    def _fallback_detection_answer(self, query: str, detection: DetectionRun) -> str:
        if detection.detections:
            return f"Falcon found {len(detection.detections)} matches for {detection.object_query!r}. Ollama is unavailable, so only the grounded detection summary is available right now."
        return f"Falcon did not find any matches for {detection.object_query!r}. Ollama is unavailable, so no richer answer could be generated for: {query}"

    def _merge_labeled_detections(self, runs: list[DetectionRun]) -> list[DetectedObject]:
        merged: list[DetectedObject] = []
        count_index = 1
        for run in runs:
            for item in run.detections:
                merged.append(
                    DetectedObject(
                        label=item.label,
                        score=item.score,
                        bbox=item.bbox,
                        mask_rle=item.mask_rle,
                        mask_area=item.mask_area,
                        count_index=count_index,
                    )
                )
                count_index += 1
        return merged

    def _plan(self, query: str) -> QueryPlan:
        lowered = query.lower().strip()
        compare = re.search(r"more\s+([a-z0-9\- ]+?)\s+than\s+([a-z0-9\- ]+)", lowered)
        if compare:
            return QueryPlan("compare_counts", [self._clean_object(compare.group(1)), self._clean_object(compare.group(2))])

        count_match = re.search(r"how many\s+([a-z0-9\- ]+?)(?:\s+(?:are|is|do|can)\b|\?|$)", lowered)
        if count_match:
            return QueryPlan("count", [self._clean_object(count_match.group(1))])

        detect_match = re.search(
            r"(?:show|find|detect|locate|where\s+(?:is|are)|highlight)\s+(?:all\s+)?([a-z0-9\- ]+?)(?:\?|$)",
            lowered,
        )
        if detect_match:
            return QueryPlan("detect_then_vlm", [self._clean_object(detect_match.group(1))])

        return QueryPlan("vlm", [])

    @staticmethod
    def _clean_object(value: str) -> str:
        cleaned = re.sub(r"\b(the|a|an|all)\b", "", value).strip(" .?!")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if cleaned.endswith("s") and len(cleaned) > 3:
            return cleaned[:-1]
        return cleaned

    @staticmethod
    def _pluralize(value: str) -> str:
        return value if value.endswith("s") else f"{value}s"