"""A small vector store: documents and their embeddings in SQLite, searched in memory with numpy.

Search is exact (brute-force cosine over every vector). At this size, a few thousand merchants and
labelled examples, one matrix multiply takes well under a millisecond, and exact search has no
recall loss and no index to tune. Past roughly a million vectors an approximate index (HNSW via
FAISS, or pgvector in Postgres) would be the next step; the interface here would not change.

The store records which embedder produced its vectors and refuses vectors from any other one,
because similarity between vectors from two different models is meaningless.
"""

import sqlite3
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .categories import Category

DocKind = Literal["example", "merchant"]


@dataclass(frozen=True)
class Document:
    id: str  # stable, so re-indexing the same thing replaces it instead of duplicating it
    kind: DocKind
    text: str  # what was embedded
    category: Category
    source: str  # "train", "kb" or "user"
    direction: str | None = None
    note: str | None = None  # shown to the LLM, not embedded


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float  # cosine similarity, from -1 to 1


class EmbedderMismatchError(ValueError):
    pass


class VectorStore:
    def __init__(self, path: str = ":memory:") -> None:
        # Same pattern as the cache: one connection shared by FastAPI's worker threads, behind a lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                " id TEXT PRIMARY KEY,"
                " kind TEXT NOT NULL,"
                " text TEXT NOT NULL,"
                " category TEXT NOT NULL,"
                " source TEXT NOT NULL,"
                " direction TEXT,"
                " note TEXT,"
                " vector BLOB NOT NULL)"
            )
        self._docs: list[Document] = []
        self._position: dict[str, int] = {}
        self._vectors: list[np.ndarray] = []
        self._matrix: np.ndarray | None = None  # rebuilt lazily after writes
        self._load()

    # --- metadata -------------------------------------------------------------------------

    @property
    def embedder_name(self) -> str | None:
        return self._meta("embedder")

    @property
    def dim(self) -> int | None:
        value = self._meta("dim")
        return int(value) if value is not None else None

    def _meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def _check_embedder(self, embedder_name: str, dim: int) -> None:
        stored_name, stored_dim = self.embedder_name, self.dim
        if stored_name is None:
            with self._lock, self._conn:
                self._conn.executemany(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                    [("embedder", embedder_name), ("dim", str(dim))],
                )
            return
        if stored_name != embedder_name or stored_dim != dim:
            raise EmbedderMismatchError(
                f"this index was built with {stored_name} ({stored_dim} dims), not "
                f"{embedder_name} ({dim} dims); use a new index path"
            )

    # --- reads and writes -----------------------------------------------------------------

    def _load(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, text, category, source, direction, note, vector FROM documents"
                " ORDER BY rowid"
            ).fetchall()
        for doc_id, kind, text, category, source, direction, note, blob in rows:
            self._append(
                Document(doc_id, kind, text, Category(category), source, direction, note),
                np.frombuffer(blob, dtype=np.float32),
            )

    def _append(self, doc: Document, vector: np.ndarray) -> None:
        if doc.id in self._position:
            self._docs[self._position[doc.id]] = doc
            self._vectors[self._position[doc.id]] = vector
        else:
            self._position[doc.id] = len(self._docs)
            self._docs.append(doc)
            self._vectors.append(vector)
        self._matrix = None

    def upsert(self, docs: Sequence[Document], vectors: np.ndarray, *, embedder_name: str) -> None:
        """Insert documents, replacing any with the same id. Vectors must be unit length."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(docs) != vectors.shape[0]:
            raise ValueError("need exactly one vector row per document")
        if not docs:
            return
        self._check_embedder(embedder_name, vectors.shape[1])
        with self._lock, self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO documents"
                " (id, kind, text, category, source, direction, note, vector)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (d.id, d.kind, d.text, d.category.value, d.source, d.direction, d.note,
                     v.tobytes())
                    for d, v in zip(docs, vectors)
                ],
            )
            for doc, vector in zip(docs, vectors):
                self._append(doc, vector.copy())

    def search(
        self,
        queries: np.ndarray,
        k: int,
        *,
        kind: DocKind | None = None,
        min_score: float = -1.0,
    ) -> list[list[Hit]]:
        """The k most similar documents for each query row, best first."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim != 2:
            raise ValueError("queries must be a 2-D matrix")
        with self._lock:
            if self._matrix is None and self._vectors:
                self._matrix = np.vstack(self._vectors)
            matrix, docs = self._matrix, list(self._docs)
        if matrix is None or k <= 0 or queries.shape[0] == 0:
            return [[] for _ in range(queries.shape[0])]
        if queries.shape[1] != matrix.shape[1]:
            raise EmbedderMismatchError(
                f"query vectors have {queries.shape[1]} dims, index has {matrix.shape[1]}"
            )

        scores = queries @ matrix.T  # unit vectors, so this is cosine similarity
        if kind is not None:
            mask = np.array([d.kind == kind for d in docs])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(docs))
        # argpartition finds the top k in linear time; only those k are then sorted.
        top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        results: list[list[Hit]] = []
        for row, candidates in enumerate(top):
            ordered = candidates[np.argsort(-scores[row, candidates])]
            results.append([
                Hit(docs[j], float(scores[row, j]))
                for j in ordered
                if np.isfinite(scores[row, j]) and scores[row, j] >= min_score
            ])
        return results

    def __len__(self) -> int:
        with self._lock:
            return len(self._docs)

    def count(self, *, kind: DocKind | None = None, source: str | None = None) -> int:
        with self._lock:
            return sum(
                1 for d in self._docs
                if (kind is None or d.kind == kind) and (source is None or d.source == source)
            )

    def delete_source(self, source: str) -> int:
        """Remove every document from one source, e.g. to re-index training data while keeping
        user corrections. Returns the number removed."""
        with self._lock, self._conn:
            removed = self._conn.execute(
                "DELETE FROM documents WHERE source = ?", (source,)
            ).rowcount
            kept = [(d, v) for d, v in zip(self._docs, self._vectors) if d.source != source]
            self._docs = [d for d, _ in kept]
            self._vectors = [v for _, v in kept]
            self._position = {d.id: i for i, d in enumerate(self._docs)}
            self._matrix = None
        return removed

    def close(self) -> None:
        with self._lock:
            self._conn.close()
