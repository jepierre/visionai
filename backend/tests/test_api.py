from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.falcon import FalconUnavailableError
from backend.app.models.falcon import FalconDetector
from backend.app.models.ollama import OllamaUnavailableError
from backend.app.rendering.annotations import DetectedObject
from backend.app.schemas import ModelInfo, TraceStep


class VisionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.image_path = self.temp_path / "sample.jpg"
        self.image_path.write_bytes(b"fake-image-bytes")
        self.thumb_path = self.temp_path / "thumb.jpg"
        self.thumb_path.write_bytes(b"fake-thumb-bytes")
        self.annotated_path = self.temp_path / "annotated.png"
        self.annotated_path.write_bytes(b"fake-annotated-bytes")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_healthcheck_returns_ok(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_warmup_reports_ready_models(self) -> None:
        with patch("backend.app.main.falcon_detector.warmup"), patch(
            "backend.app.main.ollama_client.list_models", return_value=["gemma4:latest"]
        ):
            response = self.client.post("/api/warmup")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"falcon_ready": True, "ollama_ready": True, "message": "Models are ready."},
        )

    def test_falcon_segmentation_uses_masks_as_instance_count(self) -> None:
        detector = FalconDetector.__new__(FalconDetector)
        auxiliary = SimpleNamespace(
            bboxes_raw=[
                {"x": 0.25, "y": 0.25},
                {"w": 0.2, "h": 0.3},
                {"x": 0.75, "y": 0.75},
                {"w": 0.4, "h": 0.5},
            ],
            masks_rle=[{"size": [2, 2], "counts": b"a"}, {"size": [2, 2], "counts": b"b"}],
        )

        with patch("backend.app.models.falcon.FalconDetector._mask_area", return_value=123):
            detections = detector._normalize_detections(auxiliary, "dog", (100, 200))

        self.assertEqual(len(detections), 2)
        self.assertEqual([item.bbox for item in detections], [[15.0, 20.0, 35.0, 80.0], [55.0, 100.0, 95.0, 200.0]])
        self.assertEqual([item.mask_area for item in detections], [123, 123])

    def test_list_images_returns_catalog(self) -> None:
        fake_images = [
            {
                "id": "img-1",
                "name": "sample.jpg",
                "width": 100,
                "height": 80,
                "image_url": "/api/images/img-1/file",
                "thumbnail_url": "/api/images/img-1/thumbnail/thumb.jpg",
            }
        ]

        with patch("backend.app.main.image_catalog.list_images", return_value=fake_images):
            response = self.client.get("/api/images")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"images": fake_images})

    def test_list_ollama_models_returns_available_models(self) -> None:
        with patch("backend.app.main.ollama_client.list_models", return_value=["gemma3:4b", "llava:7b"]), patch(
            "backend.app.models.ollama.OllamaVisionClient.default_model", new_callable=PropertyMock, return_value="gemma3:4b"
        ):
            response = self.client.get("/api/ollama/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "models": [{"name": "gemma3:4b"}, {"name": "llava:7b"}],
                "default_model": "gemma3:4b",
            },
        )

    def test_list_ollama_models_returns_503_when_unavailable(self) -> None:
        with patch(
            "backend.app.main.ollama_client.list_models", side_effect=OllamaUnavailableError("Ollama unavailable")
        ):
            response = self.client.get("/api/ollama/models")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Ollama unavailable"})

    def test_get_image_file_returns_bytes(self) -> None:
        record = SimpleNamespace(path=self.image_path)

        with patch("backend.app.main.image_catalog.get_image", return_value=record):
            response = self.client.get("/api/images/img-1/file")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-image-bytes")

    def test_get_image_file_returns_404_for_missing_image(self) -> None:
        with patch("backend.app.main.image_catalog.get_image", side_effect=FileNotFoundError("Unknown image id: img-1")):
            response = self.client.get("/api/images/img-1/file")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Unknown image id: img-1"})

    def test_get_thumbnail_returns_bytes(self) -> None:
        with patch("backend.app.main.image_catalog.ensure_thumbnail", return_value=self.thumb_path):
            response = self.client.get(f"/api/images/img-1/thumbnail/{self.thumb_path.name}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-thumb-bytes")

    def test_get_thumbnail_rejects_mismatched_name(self) -> None:
        with patch("backend.app.main.image_catalog.ensure_thumbnail", return_value=self.thumb_path):
            response = self.client.get("/api/images/img-1/thumbnail/wrong-name.jpg")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Thumbnail does not match the requested image"})

    def test_get_annotated_image_returns_file(self) -> None:
        fake_settings = SimpleNamespace(annotated_dir=self.temp_path)
        with patch("backend.app.main.settings", fake_settings):
            response = self.client.get(f"/api/annotated/{self.annotated_path.name}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-annotated-bytes")

    def test_get_annotated_image_returns_404_when_missing(self) -> None:
        fake_settings = SimpleNamespace(annotated_dir=self.temp_path)
        with patch("backend.app.main.settings", fake_settings):
            response = self.client.get("/api/annotated/missing.png")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Annotated image not found"})

    def test_detect_returns_response_payload(self) -> None:
        record = SimpleNamespace(path=self.image_path)
        detections = [
            DetectedObject(
                label="dog",
                score=0.98,
                bbox=[1.0, 2.0, 3.0, 4.0],
                mask_rle=None,
                mask_area=321,
                count_index=1,
            )
        ]
        fake_run = SimpleNamespace(
            detections=detections,
            annotated_file_name="annotated.png",
            message=None,
            trace=[TraceStep(title="Run Falcon inference", detail="Falcon produced 1 detection.", model="tiiuae/Falcon-Perception", action="DETECT")],
            reasoning="Falcon-only mode ran segmentation for 'dog' and returned 1 detection.",
            final_output="Falcon detected 1 match(es) for 'dog'.",
            timings={"inference_seconds": 1.25},
        )

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.falcon_detector.detect", return_value=fake_run
        ):
            response = self.client.post(
                "/api/detect",
                json={"image_id": "img-1", "object_query": "dog", "annotation_mode": "combined"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "image_id": "img-1",
                "object_query": "dog",
                "annotation_mode": "combined",
                "original_image_url": "/api/images/img-1/file",
                "annotated_image_url": "/api/annotated/annotated.png",
                "detections": [
                    {
                        "label": "dog",
                        "score": 0.98,
                        "bbox": [1.0, 2.0, 3.0, 4.0],
                        "mask_area": 321,
                        "count_index": 1,
                    }
                ],
                "execution_mode": "falcon",
                "models_used": [{"name": "tiiuae/Falcon-Perception", "role": "Grounding and segmentation"}],
                "trace": [
                    {
                        "title": "Run Falcon inference",
                        "detail": "Falcon produced 1 detection.",
                        "model": "tiiuae/Falcon-Perception",
                        "action": "DETECT",
                    }
                ],
                "reasoning": "Falcon-only mode ran segmentation for 'dog' and returned 1 detection.",
                "final_output": "Falcon detected 1 match(es) for 'dog'.",
                "message": None,
                "timings": {"inference_seconds": 1.25},
            },
        )

    def test_detect_returns_404_for_unknown_image(self) -> None:
        with patch("backend.app.main.image_catalog.get_image", side_effect=FileNotFoundError("Unknown image id: img-1")):
            response = self.client.post(
                "/api/detect",
                json={"image_id": "img-1", "object_query": "dog", "annotation_mode": "combined"},
            )

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Unknown image id: img-1"})

    def test_detect_returns_503_when_falcon_unavailable(self) -> None:
        record = SimpleNamespace(path=self.image_path)

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.falcon_detector.detect", side_effect=FalconUnavailableError("Falcon unavailable")
        ):
            response = self.client.post(
                "/api/detect",
                json={"image_id": "img-1", "object_query": "dog", "annotation_mode": "combined"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Falcon unavailable"})

    def test_chat_returns_response_payload(self) -> None:
        record = SimpleNamespace(path=self.image_path)
        detections = [
            DetectedObject(
                label="dog",
                score=0.91,
                bbox=[10.0, 20.0, 30.0, 40.0],
                mask_rle=None,
                mask_area=456,
                count_index=1,
            )
        ]
        fake_run = SimpleNamespace(
            answer="I found 1 dog in the image.",
            route="detect_count",
            execution_mode="agent",
            annotated_file_name="annotated.png",
            detections=detections,
            models_used=[ModelInfo(name="tiiuae/Falcon-Perception", role="Grounding and segmentation")],
            trace=[TraceStep(title="Plan query", detail="Planner recognized a count question for 'dog'.", model=None, action="DETECT")],
            reasoning="The planner chose deterministic counting after Falcon grounding for 'dog'.",
            final_output="I found 1 dog in the image.",
            message="done",
            timings={"inference_seconds": 0.9},
        )

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.chat_orchestrator.answer", return_value=fake_run
        ):
            response = self.client.post(
                "/api/chat",
                json={"image_id": "img-1", "query": "How many dogs are there?", "annotation_mode": "combined"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "image_id": "img-1",
                "query": "How many dogs are there?",
                "answer": "I found 1 dog in the image.",
                "route": "detect_count",
                "execution_mode": "agent",
                "original_image_url": "/api/images/img-1/file",
                "annotated_image_url": "/api/annotated/annotated.png",
                "detections": [
                    {
                        "label": "dog",
                        "score": 0.91,
                        "bbox": [10.0, 20.0, 30.0, 40.0],
                        "mask_area": 456,
                        "count_index": 1,
                    }
                ],
                "models_used": [{"name": "tiiuae/Falcon-Perception", "role": "Grounding and segmentation"}],
                "trace": [
                    {
                        "title": "Plan query",
                        "detail": "Planner recognized a count question for 'dog'.",
                        "model": None,
                        "action": "DETECT",
                    }
                ],
                "reasoning": "The planner chose deterministic counting after Falcon grounding for 'dog'.",
                "final_output": "I found 1 dog in the image.",
                "message": "done",
                "timings": {"inference_seconds": 0.9},
                "token_usage": None,
            },
        )

    def test_chat_passes_execution_mode_and_object_query(self) -> None:
        record = SimpleNamespace(path=self.image_path)
        fake_run = SimpleNamespace(
            answer="Grounded answer",
            route="forced_detect_then_vlm",
            execution_mode="agent",
            annotated_file_name="annotated.png",
            detections=[],
            models_used=[ModelInfo(name="tiiuae/Falcon-Perception", role="Grounding and segmentation")],
            trace=[],
            reasoning="Combined mode forced grounding.",
            final_output="Grounded answer",
            message=None,
            timings={"inference_seconds": 0.5},
        )

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.chat_orchestrator.answer", return_value=fake_run
        ) as answer_mock:
            response = self.client.post(
                "/api/chat",
                json={
                    "image_id": "img-1",
                    "query": "Where is the dog?",
                    "annotation_mode": "combined",
                    "execution_mode": "agent",
                    "object_query": "dog",
                },
            )

        self.assertEqual(response.status_code, 200)
        answer_mock.assert_called_once_with(
            self.image_path,
            "Where is the dog?",
            "combined",
            execution_mode="agent",
            object_query="dog",
            ollama_model=None,
        )

    def test_chat_passes_selected_ollama_model(self) -> None:
        record = SimpleNamespace(path=self.image_path)
        fake_run = SimpleNamespace(
            answer="Direct answer",
            route="gemma_only",
            execution_mode="gemma",
            annotated_file_name=None,
            detections=[],
            models_used=[ModelInfo(name="gemma3:12b", role="Visual reasoning and final answer generation")],
            trace=[],
            reasoning="Gemma-only mode.",
            final_output="Direct answer",
            message=None,
            timings={"ollama_seconds": 0.5},
        )

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.chat_orchestrator.answer", return_value=fake_run
        ) as answer_mock:
            response = self.client.post(
                "/api/chat",
                json={
                    "image_id": "img-1",
                    "query": "Describe this image.",
                    "annotation_mode": "combined",
                    "execution_mode": "gemma",
                    "ollama_model": "gemma3:12b",
                },
            )

        self.assertEqual(response.status_code, 200)
        answer_mock.assert_called_once_with(
            self.image_path,
            "Describe this image.",
            "combined",
            execution_mode="gemma",
            object_query=None,
            ollama_model="gemma3:12b",
        )

    def test_chat_returns_503_when_ollama_unavailable(self) -> None:
        record = SimpleNamespace(path=self.image_path)

        with patch("backend.app.main.image_catalog.get_image", return_value=record), patch(
            "backend.app.main.chat_orchestrator.answer", side_effect=OllamaUnavailableError("Ollama unavailable")
        ):
            response = self.client.post(
                "/api/chat",
                json={"image_id": "img-1", "query": "Describe this image.", "annotation_mode": "combined"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Ollama unavailable"})


if __name__ == "__main__":
    unittest.main()