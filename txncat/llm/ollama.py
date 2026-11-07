"""Client for a local model served by Ollama. Free to run, no API key."""

import httpx

from .base import LLMError, post_json

DEFAULT_MODEL = "qwen2.5:3b"


class OllamaClient:
    name = "ollama"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)

    def complete_json(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        response = post_json(self._http, "/api/chat", json=payload)
        try:
            return response.json()["message"]["content"]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected Ollama response shape: {exc}") from exc
