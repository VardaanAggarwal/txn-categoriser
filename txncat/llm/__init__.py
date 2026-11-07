from .base import LLMClient, LLMError, TransientLLMError
from .fake import FakeLLM
from .gemini import GeminiClient
from .ollama import OllamaClient

PROVIDERS = ("ollama", "gemini")


def make_client(provider: str, model: str | None = None) -> LLMClient:
    if provider == "ollama":
        return OllamaClient(model=model) if model else OllamaClient()
    if provider == "gemini":
        return GeminiClient(model=model) if model else GeminiClient()
    raise ValueError(f"unknown LLM provider {provider!r}; choose one of {PROVIDERS}")


__all__ = [
    "FakeLLM", "GeminiClient", "LLMClient", "LLMError", "OllamaClient", "PROVIDERS",
    "TransientLLMError", "make_client",
]
