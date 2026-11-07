"""The provider-neutral LLM interface and shared HTTP error handling."""

from typing import Protocol

import httpx


class LLMError(Exception):
    """A failure that retrying will not fix (bad request, bad credentials, malformed reply)."""


class TransientLLMError(LLMError):
    """A failure worth retrying: timeouts, connection errors, rate limits, server errors."""


class LLMClient(Protocol):
    name: str

    def complete_json(self, system: str, user: str) -> str:
        """Send one prompt and return the model's raw text, which should be JSON."""
        ...


def post_json(http: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """POST and translate transport and HTTP failures into the two error types above."""
    try:
        response = http.post(url, **kwargs)
    except httpx.TimeoutException as exc:
        raise TransientLLMError(f"timeout calling {url}") from exc
    except httpx.TransportError as exc:
        raise TransientLLMError(f"connection error calling {url}: {exc}") from exc
    if response.status_code == 429 or response.status_code >= 500:
        raise TransientLLMError(f"HTTP {response.status_code} from {url}")
    if response.status_code >= 400:
        raise LLMError(f"HTTP {response.status_code} from {url}: {response.text[:200]}")
    return response
