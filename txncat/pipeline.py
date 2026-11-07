"""The cascade: cache -> rules -> ML -> LLM -> review.

Each layer only sees what the cheaper layers before it couldn't decide confidently. Anything no
layer is confident about is marked needs_review instead of guessed, because in a finance app a
confident wrong answer is worse than "we're not sure".
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, Field

from .cache import Cache
from .categories import Category
from .llm.base import LLMError
from .llm_categoriser import LLMCategoriser
from .ml import MLCategoriser
from .normalise import Direction, Normalised, normalise
from .rag import Retriever
from .rules import match_rules

Source = Literal["cache", "rule", "ml", "llm", "none"]


class Transaction(BaseModel):
    description: str = Field(min_length=1, max_length=500)
    direction: Direction | None = None


class CategorisedTransaction(BaseModel):
    description: str
    category: Category
    confidence: float
    source: Source
    needs_review: bool
    reason: str
    normalised_key: str


def _dedupe_key(n: Normalised) -> str:
    # Descriptions with no usable words all normalise to the same empty key, so two different
    # transfers to phone-number UPI IDs would be merged. Keep those apart by their raw text.
    return n.cache_key if n.tokens else f"raw|{n.raw}"


def _result(
    n: Normalised, category: Category, confidence: float, source: Source, reason: str,
    needs_review: bool = False,
) -> CategorisedTransaction:
    return CategorisedTransaction(
        description=n.raw,
        category=category,
        confidence=round(confidence, 4),
        source=source,
        needs_review=needs_review,
        reason=reason,
        normalised_key=n.key,
    )


class Pipeline:
    def __init__(
        self,
        *,
        ml: MLCategoriser | None = None,
        llm: LLMCategoriser | None = None,
        cache: Cache | None = None,
        use_rules: bool = True,
        ml_threshold: float = 0.6,
        retriever: Retriever | None = None,
    ) -> None:
        self.ml = ml
        self.llm = llm
        self.cache = cache
        self.use_rules = use_rules
        self.ml_threshold = ml_threshold
        # Only used to index corrections here; the LLM layer reads from it via its examples_fn
        # and facts_fn. Pass the same Retriever to both.
        self.retriever = retriever

    @property
    def layers(self) -> list[str]:
        names = ["cache"] if self.cache is not None else []
        names += ["rules"] if self.use_rules else []
        names += ["ml"] if self.ml is not None else []
        names += [f"llm:{self.llm.client.name}"] if self.llm is not None else []
        names += [f"rag:{self.retriever.embedder.name}"] if self.retriever is not None else []
        return names

    def categorise(self, transactions: Sequence[Transaction]) -> list[CategorisedTransaction]:
        norms = [normalise(t.description, t.direction) for t in transactions]
        out: list[CategorisedTransaction | None] = [None] * len(norms)
        todo: list[int] = []

        for i, n in enumerate(norms):
            if self.cache is not None and (hit := self.cache.get(n.cache_key)):
                out[i] = _result(n, hit.category, hit.confidence, "cache", f"cache:{hit.source}")
            elif self.use_rules and (match := match_rules(n)):
                out[i] = _result(n, match.category, 1.0, "rule", match.reason)
            else:
                todo.append(i)

        if self.ml is not None and todo:
            remaining = []
            for i, (category, prob) in zip(todo, self.ml.predict([norms[i] for i in todo])):
                if prob >= self.ml_threshold:
                    out[i] = _result(norms[i], category, prob, "ml", f"ml:p={prob:.2f}")
                    self._remember(norms[i], category, "ml", prob)
                else:
                    remaining.append(i)
            todo = remaining

        if self.llm is not None and todo:
            todo = self._run_llm(norms, todo, out)

        for i in todo:
            out[i] = _result(
                norms[i], Category.OTHER, 0.0, "none", "no layer was confident", needs_review=True
            )
        return [r for r in out if r is not None]

    def record_correction(self, transaction: Transaction, category: Category) -> bool:
        """Remember a user's correction, so this merchant gets this category from now on.

        Corrections go into the cache, which is checked before every other layer, so they override
        rules and models for this exact merchant. With RAG on, they're also indexed as labelled
        examples, so the LLM sees them when it meets a similar merchant.

        Returns False if there's nowhere to store it (no cache and no retriever), or if the
        description has no usable words to key on (for example a transfer to a phone-number UPI ID).
        """
        n = normalise(transaction.description, transaction.direction)
        if not n.tokens:
            return False
        stored = False
        if self.cache is not None:
            self.cache.put(n.cache_key, category, "user", 1.0)
            stored = True
        if self.retriever is not None:
            try:
                stored = self.retriever.add_correction(n, category) or stored
            except LLMError:
                pass  # the embedding API failed; the cache entry above still holds the correction
        return stored

    def _run_llm(
        self, norms: list[Normalised], todo: list[int], out: list[CategorisedTransaction | None]
    ) -> list[int]:
        # Ask about each distinct description once, however many times it appears in the batch.
        first_index: dict[str, int] = {}
        for i in todo:
            first_index.setdefault(_dedupe_key(norms[i]), i)
        unique = list(first_index.values())
        predictions = dict(
            zip((_dedupe_key(norms[i]) for i in unique), self.llm.categorise([norms[i] for i in unique]))
        )

        remaining = []
        for i in todo:
            prediction = predictions[_dedupe_key(norms[i])]
            if prediction is not None and prediction.confidence >= self.llm.min_confidence:
                out[i] = _result(
                    norms[i], prediction.category, prediction.confidence, "llm",
                    f"llm:{self.llm.client.name}",
                )
                self._remember(norms[i], prediction.category, "llm", prediction.confidence)
            else:
                remaining.append(i)
        return remaining

    def _remember(self, n: Normalised, category: Category, source: str, confidence: float) -> None:
        if self.cache is not None and n.tokens:  # an empty key would make unrelated transactions collide
            self.cache.put(n.cache_key, category, source, confidence)
