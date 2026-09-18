from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.falcon import FalconUnavailableError
from backend.app.models.ollama import OllamaUnavailableError
from backend.app.rendering.annotations import DetectedObject


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
            annotated_file_name="annotated.png",
            detections=detections,
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
                "message": "done",
                "timings": {"inference_seconds": 0.9},
            },
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