"""Checks the request each client sends and how it reads replies, without any network."""

import json

import httpx
import pytest

from txncat.llm import make_client
from txncat.llm.base import LLMError, TransientLLMError
from txncat.llm.gemini import GeminiClient
from txncat.llm.ollama import OllamaClient


def transport(status: int, body: dict | str, seen: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if isinstance(body, str):
            return httpx.Response(status, text=body)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def ollama(status: int, body, seen) -> OllamaClient:
    http = httpx.Client(base_url="http://localhost:11434", transport=transport(status, body, seen))
    return OllamaClient(model="test-model", http=http)


def gemini(status: int, body, seen) -> GeminiClient:
    http = httpx.Client(
        base_url="https://example.test", headers={"x-goog-api-key": "k"},
        transport=transport(status, body, seen),
    )
    return GeminiClient(model="test-model", api_key="k", http=http)


def test_ollama_request_and_reply():
    seen: list[httpx.Request] = []
    client = ollama(200, {"message": {"content": '{"results": []}'}}, seen)
    assert client.complete_json("sys", "user") == '{"results": []}'
    sent = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/chat"
    assert sent["model"] == "test-model" and sent["format"] == "json" and sent["stream"] is False
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_errors_are_transient(status):
    with pytest.raises(TransientLLMError):
        ollama(status, "busy", []).complete_json("s", "u")


def test_client_errors_are_permanent():
    with pytest.raises(LLMError) as exc:
        ollama(400, "bad", []).complete_json("s", "u")
    assert not isinstance(exc.value, TransientLLMError)


def test_connection_failure_is_transient():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(refuse))
    with pytest.raises(TransientLLMError):
        OllamaClient(http=http).complete_json("s", "u")


def test_unexpected_reply_shape_is_an_error():
    with pytest.raises(LLMError):
        ollama(200, {"unexpected": True}, []).complete_json("s", "u")


def test_gemini_request_and_reply():
    seen: list[httpx.Request] = []
    body = {"candidates": [{"content": {"parts": [{"text": '{"results": []}'}]}}]}
    assert gemini(200, body, seen).complete_json("sys", "user") == '{"results": []}'
    request = seen[0]
    assert request.url.path == "/v1beta/models/test-model:generateContent"
    assert request.headers["x-goog-api-key"] == "k"
    assert "key=" not in str(request.url)  # never in the URL
    sent = json.loads(request.content)
    assert sent["generationConfig"]["responseMimeType"] == "application/json"
    assert sent["systemInstruction"]["parts"][0]["text"] == "sys"


def test_gemini_without_candidates_is_an_error():
    with pytest.raises(LLMError):
        gemini(200, {"candidates": []}, []).complete_json("s", "u")


def test_gemini_requires_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        GeminiClient()


def test_make_client_rejects_unknown_provider():
    with pytest.raises(ValueError):
        make_client("openai-ish")
