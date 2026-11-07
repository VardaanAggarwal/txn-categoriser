"""SQLite cache of model decisions, keyed on the normalised description.

Only ML and LLM results are cached; rules are already cheaper than a cache lookup. Because the key
has reference numbers and dates stripped, a merchant you pay every week is a model call once and a
cache hit after that.
"""

import sqlite3
import threading
from dataclasses import dataclass

from .categories import Category


@dataclass(frozen=True)
class CachedPrediction:
    category: Category
    source: str
    confidence: float


class Cache:
    def __init__(self, path: str = ":memory:") -> None:
        # FastAPI runs sync endpoints in a thread pool, so share one connection behind a lock.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock, self._conn:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS predictions ("
                " key TEXT PRIMARY KEY,"
                " category TEXT NOT NULL,"
                " source TEXT NOT NULL,"
                " confidence REAL NOT NULL)"
            )
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> CachedPrediction | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT category, source, confidence FROM predictions WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                self.misses += 1
                return None
            self.hits += 1
        return CachedPrediction(Category(row[0]), row[1], row[2])

    def put(self, key: str, category: Category, source: str, confidence: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO predictions (key, category, source, confidence)"
                " VALUES (?, ?, ?, ?)",
                (key, category.value, source, confidence),
            )

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
