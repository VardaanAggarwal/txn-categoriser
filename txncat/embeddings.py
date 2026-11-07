"""Text embeddings for retrieval: one interface, three implementations.

- HashingEmbedder: offline and deterministic, no model download or API key. It hashes character
  n-grams, so it is lexical, not semantic: it finds "THEOBROMA BAKERY" from "THEOBROMA", but it
  can't know that "CHAAYOS" is a cafe. It's the default so the system runs anywhere.
- OllamaEmbedder: a local embedding model such as nomic-embed-text. Free, semantic.
- GeminiEmbedder: the Gemini embedding API, which has a free tier. Semantic.

Every implementation returns float32 rows with unit length, so a dot product is cosine similarity.
Vectors from different embedders are not comparable; the vector store refuses to mix them.
"""

import os
import time
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from .llm.base import LLMError, post_json
from .retry import with_retries

Kind = Literal["query", "document"]


class Embedder(Protocol):
    name: str  # identifies the model, e.g. "ollama:nomic-embed-text"

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        """One unit-length float32 row per text."""
        ...


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1. All-zero rows stay zero instead of dividing by zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _batched(
    texts: Sequence[str], size: int, embed_batch: Callable[[list[str]], list[list[float]]]
) -> np.ndarray:
    rows: list[list[float]] = []
    for start in range(0, len(texts), size):
        batch = list(texts[start : start + size])
        vectors = embed_batch(batch)
        if len(vectors) != len(batch):
            raise LLMError(f"asked for {len(batch)} embeddings, got {len(vectors)}")
        rows.extend(vectors)
    return unit_rows(np.array(rows, dtype=np.float32))


class HashingEmbedder:
    def __init__(self, dim: int = 1024, ngram_range: tuple[int, int] = (2, 4)) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}:{ngram_range[0]}-{ngram_range[1]}"
        # Stateless: nothing to fit, so documents and queries map to the same space from day one.
        self._vectoriser = HashingVectorizer(
            analyzer="char_wb", ngram_range=ngram_range, n_features=dim,
            alternate_sign=False, norm="l2", lowercase=True,
        )

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return unit_rows(self._vectoriser.transform(list(texts)).toarray())


class OllamaEmbedder:
    DEFAULT_MODEL = "nomic-embed-text"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 64,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.batch_size = batch_size
        self._http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self._sleep = sleep

    def _prefix(self, kind: Kind) -> str:
        # nomic-embed-text is trained with task prefixes and retrieves noticeably worse without them.
        if self.model.startswith("nomic-embed"):
            return "search_query: " if kind == "query" else "search_document: "
        return ""

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        prefix = self._prefix(kind)

        def call(batch: list[str]) -> list[list[float]]:
            payload = {"model": self.model, "input": [prefix + t for t in batch]}
            response = with_retries(
                lambda: post_json(self._http, "/api/embed", json=payload),
                sleep=self._sleep,
            )
            try:
                return response.json()["embeddings"]
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError(f"unexpected Ollama embed response shape: {exc}") from exc

        return _batched(texts, self.batch_size, call)


class GeminiEmbedder:
    DEFAULT_MODEL = "gemini-embedding-001"
    MAX_BATCH = 100  # the API's limit for batchEmbedContents

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        dimensions: int = 768,
        base_url: str = "https://generativelanguage.googleapis.com",
        timeout: float = 60.0,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError("GEMINI_API_KEY is not set")
        self.model = model
        self.dimensions = dimensions
        self.name = f"gemini:{model}:{dimensions}"
        # As with the chat client, the key goes in a header so it never appears in a URL or log.
        self._http = http or httpx.Client(
            base_url=base_url, timeout=timeout, headers={"x-goog-api-key": api_key}
        )
        self._sleep = sleep

    def embed(self, texts: Sequence[str], *, kind: Kind = "document") -> np.ndarray:
        # Gemini optimises the vector for its role, so queries and documents are embedded differently.
        task = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"

        def call(batch: list[str]) -> list[list[float]]:
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.model}",
                        "content": {"parts": [{"text": t}]},
                        "taskType": task,
                        "outputDimensionality": self.dimensions,
                    }
                    for t in batch
                ]
            }
            response = with_retries(
                lambda: post_json(
                    self._http, f"/v1beta/models/{self.model}:batchEmbedContents", json=payload
                ),
                sleep=self._sleep,
            )
            try:
                return [e["values"] for e in response.json()["embeddings"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise LLMError(f"unexpected Gemini embed response shape: {exc}") from exc

        # Reduced-dimension Gemini vectors are not unit length; _batched normalises them.
        return _batched(texts, self.MAX_BATCH, call)


EMBEDDERS = ("hashing", "ollama", "gemini")


def make_embedder(name: str, model: str | None = None) -> Embedder:
    if name == "hashing":
        return HashingEmbedder()
    if name == "ollama":
        return OllamaEmbedder(model=model) if model else OllamaEmbedder()
    if name == "gemini":
        return GeminiEmbedder(model=model) if model else GeminiEmbedder()
    raise ValueError(f"unknown embedder {name!r}; choose one of {EMBEDDERS}")
