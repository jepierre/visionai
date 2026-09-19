from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from backend.app.config import Settings
from backend.app.rendering.annotations import AnnotationRenderer, DetectedObject


class AnnotationRendererTests(unittest.TestCase):
    def test_render_combined_overlay_writes_png(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_path = root / "source.png"
            Image.new("RGB", (32, 24), (20, 30, 40)).save(image_path)
            settings = Settings(
                root_dir=root,
                images_dir=root,
                output_dir=root / "output",
                thumbnails_dir=root / "output" / "thumbnails",
                annotated_dir=root / "output" / "annotated",
                falcon_model_id="test",
                falcon_revision="main",
                falcon_dtype="float32",
                cuda_device="cpu",
                ollama_base_url="http://localhost:11434",
                ollama_model="test-model",
            )
            renderer = AnnotationRenderer(settings)
            detection = DetectedObject("dog", 0.9, [4, 4, 20, 18], None, None, 1)

            file_name = renderer.render(image_path, [detection], "combined")

            self.assertIsNotNone(file_name)
            output = settings.annotated_dir / file_name
            self.assertTrue(output.is_file())
            with Image.open(output) as rendered:
                self.assertEqual(rendered.size, (32, 24))

    def test_render_empty_detections_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings = Settings(
                root_dir=root,
                images_dir=root,
                output_dir=root / "output",
                thumbnails_dir=root / "output" / "thumbnails",
                annotated_dir=root / "output" / "annotated",
                falcon_model_id="test",
                falcon_revision="main",
                falcon_dtype="float32",
                cuda_device="cpu",
                ollama_base_url="http://localhost:11434",
                ollama_model="test-model",
            )
            renderer = AnnotationRenderer(settings)

            self.assertIsNone(renderer.render(root / "missing.png", [], "combined"))


if __name__ == "__main__":
    unittest.main()