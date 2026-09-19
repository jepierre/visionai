from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app.agent.chat import ChatOrchestrator
from backend.app.models.falcon import FalconDetector


class AgentPlannerTests(unittest.TestCase):
    def test_planner_assigns_bounded_actions(self) -> None:
        orchestrator = ChatOrchestrator.__new__(ChatOrchestrator)

        plan = orchestrator._plan("What color is the largest car?")

        self.assertEqual(plan.route, "crop")
        self.assertEqual(plan.object_queries, ["car"])
        self.assertEqual(plan.actions, ["DETECT", "CROP", "VLM", "ANSWER"])
        self.assertLessEqual(len(plan.actions), ChatOrchestrator.MAX_ACTIONS)

    def test_planner_handles_comparison_as_detect_each(self) -> None:
        orchestrator = ChatOrchestrator.__new__(ChatOrchestrator)

        plan = orchestrator._plan("Are there more dogs than cats?")

        self.assertEqual(plan.route, "compare_counts")
        self.assertEqual(plan.actions, ["DETECT_EACH", "COMPARE", "ANSWER"])

    def test_planner_recognizes_common_count_phrasings(self) -> None:
        orchestrator = ChatOrchestrator.__new__(ChatOrchestrator)

        for query in ("How many dogs are in this image?", "How many dogs are there?", "How many dogs do you see?"):
            with self.subTest(query=query):
                plan = orchestrator._plan(query)
                self.assertEqual(plan.route, "count")
                self.assertEqual(plan.object_queries, ["dog"])

    def test_falcon_detector_reuses_cached_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.jpg"
            image_path.write_bytes(b"image")
            detector = FalconDetector.__new__(FalconDetector)
            detector._cache = {}
            import threading

            detector._inference_lock = threading.RLock()
            expected = SimpleNamespace(annotated_file_name="annotated.png")
            detector._detect_uncached = lambda *args: expected

            with patch.object(detector, "_detect_uncached", return_value=expected) as detect_mock:
                first = detector.detect(image_path, "dog", "combined")
                second = detector.detect(image_path, "dog", "combined")

            self.assertIs(first, second)
            detect_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()