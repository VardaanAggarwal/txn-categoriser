"""Client for the Gemini API, which has a free tier. Reads the key from GEMINI_API_KEY."""

import os

import httpx

from .base import LLMError, post_json

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient:
    name = "gemini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 60.0,
        http: httpx.Client | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model
        # The key goes in a header, never in the URL, so it doesn't end up in logs.
        self._http = http or httpx.Client(
            base_url=base_url, timeout=timeout, headers={"x-goog-api-key": api_key}
        )

    def complete_json(self, system: str, user: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        response = post_json(self._http, f"/v1beta/models/{self.model}:generateContent", json=payload)
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"unexpected Gemini response shape: {exc}") from exc
