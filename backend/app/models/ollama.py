from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from backend.app.config import Settings
from backend.app.schemas import ModelInfo


class OllamaUnavailableError(RuntimeError):
    pass


@dataclass
class OllamaResult:
    answer: str
    model: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class OllamaVisionClient:
    def __init__(self, settings: Settings):
        self._settings = settings

    @property
    def default_model(self) -> str:
        return self._settings.ollama_model

    def generate(self, prompt: str, image_path: Path, model: str | None = None) -> OllamaResult:
        resolved_model = model or self._settings.ollama_model
        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "images": [base64.b64encode(image_path.read_bytes()).decode("utf-8")],
            "stream": False,
        }
        endpoint = f"{self._settings.ollama_base_url.rstrip('/')}/api/generate"
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=240) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"Ollama is not reachable at {endpoint}. Start the local Ollama service and pull {resolved_model}."
            ) from exc

        answer = str(response_payload.get("response", "")).strip()
        if not answer:
            raise OllamaUnavailableError(f"Ollama returned an empty response for model {resolved_model!r}.")
        return OllamaResult(
            answer=answer,
            model=str(response_payload.get("model", resolved_model)),
            duration_seconds=perf_counter() - started,
            prompt_tokens=int(response_payload.get("prompt_eval_count") or 0),
            completion_tokens=int(response_payload.get("eval_count") or 0),
        )

    def list_models(self) -> list[str]:
        endpoint = f"{self._settings.ollama_base_url.rstrip('/')}/api/tags"
        request = urllib.request.Request(endpoint, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise OllamaUnavailableError(
                f"Ollama is not reachable at {endpoint}. Start the local Ollama service and pull {self._settings.ollama_model}."
            ) from exc

        models = response_payload.get("models", [])
        names = [str(item.get("name", "")).strip() for item in models if str(item.get("name", "")).strip()]
        return names

    def model_info(self) -> ModelInfo:
        return ModelInfo(name=self._settings.ollama_model, role="Visual reasoning and final answer generation")