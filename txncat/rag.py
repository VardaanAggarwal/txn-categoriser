"""Retrieval-augmented generation for the LLM layer.

Before the LLM sees a transaction, the retriever looks up two kinds of context:

1. **Labelled examples**: past transactions with known categories (the training data plus every
   user correction), used as few-shot examples.
2. **Merchant facts** from the knowledge base, such as "THEOBROMA: bakery and dessert chain".

Examples come from hybrid search: dense (embedding) search and the ML layer's TF-IDF search each
produce a ranking, and reciprocal rank fusion merges them. Dense search can match by meaning when a
semantic embedder is used; TF-IDF is better at exact, odd strings such as UPI handles. Fusing ranks
instead of scores avoids comparing two incomparable similarity scales.

Anything below a similarity floor is dropped, because irrelevant context is worse than none: a
small model given a wrong "similar" example tends to copy its label.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from .categories import Category
from .embeddings import Embedder
from .knowledge import DEFAULT_KB, MerchantFact, MerchantNote, load_kb
from .ml import Example
from .normalise import Normalised, normalise
from .vectorstore import Document, EmbedderMismatchError, VectorStore

T = TypeVar("T")

LexicalFn = Callable[[Sequence[Normalised], int], list[list[Example]]]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]], *, key: Callable[[T], Hashable], k: int = 60
) -> list[T]:
    """Merge ranked lists: each item scores sum(1 / (k + rank)) over the lists it appears in.

    k = 60 is the constant from the original RRF paper (Cormack et al., 2009). It stops the top
    item of a single list from dominating items that rank well in several lists.
    """
    scores: dict[Hashable, float] = defaultdict(float)
    first: dict[Hashable, T] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            item_key = key(item)
            scores[item_key] += 1.0 / (k + rank)
            first.setdefault(item_key, item)
    return [first[item_key] for item_key in sorted(scores, key=lambda x: -scores[x])]


def ensure_indexed(
    retriever: "Retriever",
    *,
    examples: Iterable[tuple[Normalised, Category]] | None = None,
    facts: Iterable[MerchantFact] | None = None,
    refresh: bool = False,
) -> dict[str, int]:
    """Index training examples and knowledge-base facts unless the store already has them.

    Embedding costs time (and API quota with Gemini), so a persisted index is reused across
    restarts. refresh=True re-indexes both sources; user corrections are never touched.
    """
    store = retriever.store
    written = {"train": 0, "kb": 0}
    if examples is not None and (refresh or store.count(source="train") == 0):
        store.delete_source("train")
        written["train"] = retriever.index_examples(examples, source="train")
    if facts is not None and (refresh or store.count(source="kb") == 0):
        store.delete_source("kb")
        written["kb"] = retriever.index_facts(facts)
    return written


def labelled(rows: Iterable) -> list[tuple[Normalised, Category]]:
    """(normalised description, category) pairs from dataset rows."""
    return [(normalise(r.description, r.direction), r.category) for r in rows]


def build_retriever(
    embedder: Embedder,
    *,
    index_path: str = ":memory:",
    lexical_fn: LexicalFn | None = None,
    rows: Iterable | None = None,
    kb_path: str | Path | None = DEFAULT_KB,
    refresh: bool = False,
) -> "Retriever":
    """The one place the API, the CLI and the evaluation build their retriever."""
    retriever = Retriever(VectorStore(index_path), embedder, lexical_fn=lexical_fn)
    ensure_indexed(
        retriever,
        examples=labelled(rows) if rows is not None else None,
        facts=load_kb(kb_path) if kb_path is not None else None,
        refresh=refresh,
    )
    return retriever


def query_text(n: Normalised) -> str:
    """Search on the cleaned payee words: reference numbers and bank codes only add noise."""
    return n.key or n.raw


def _example_id(source: str, n: Normalised) -> str:
    return f"{source}|{n.cache_key}"


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        lexical_fn: LexicalFn | None = None,
        min_example_score: float = 0.35,
        min_fact_score: float = 0.5,
        candidates: int = 10,
    ) -> None:
        # Fail at startup, not at the first search: two models can share a dimension, and then
        # the mismatch would silently return nonsense instead of raising.
        if store.embedder_name is not None and store.embedder_name != embedder.name:
            raise EmbedderMismatchError(
                f"the index was built with {store.embedder_name}, not {embedder.name}; use a new "
                "index path, or delete the old index and run `python -m txncat index` again"
            )
        self.store = store
        self.embedder = embedder
        self.lexical_fn = lexical_fn
        self.min_example_score = min_example_score
        self.min_fact_score = min_fact_score
        self.candidates = candidates
        self._last_queries: tuple[tuple[str, ...], object] | None = None

    # --- indexing -------------------------------------------------------------------------

    def index_examples(
        self, items: Iterable[tuple[Normalised, Category]], *, source: str = "train"
    ) -> int:
        """Index labelled transactions. Returns the number of documents written.

        Statements repeat the same merchant with different reference numbers, so rows are grouped
        by normalised key and each key is indexed once, with its most common label.
        """
        labels: dict[str, Counter] = defaultdict(Counter)
        representative: dict[str, Normalised] = {}
        for n, category in items:
            if not n.tokens:  # no usable words: every such row would look identical
                continue
            labels[n.cache_key][category] += 1
            representative.setdefault(n.cache_key, n)

        docs = [
            Document(
                id=_example_id(source, n),
                kind="example",
                text=query_text(n),
                category=labels[key].most_common(1)[0][0],
                source=source,
                direction=n.direction,
                note=n.raw,  # the original description, shown to the LLM as the example
            )
            for key, n in representative.items()
        ]
        return self._write(docs)

    def index_facts(self, facts: Iterable[MerchantFact]) -> int:
        docs = [
            Document(
                id=f"kb|{fact.name}",
                kind="merchant",
                text=fact.search_text,
                category=fact.category,
                source="kb",
                note=fact.note,
            )
            for fact in facts
        ]
        return self._write(docs)

    def add_correction(self, n: Normalised, category: Category) -> bool:
        """Index a user's correction, so similar merchants, not only this exact one, benefit.

        The cache already makes the exact merchant correct from now on. This makes the correction
        show up as a few-shot example for lookalikes: correcting "SHARMA SWEETS" helps with
        "SHARMA SWEETS AND NAMKEEN" too.
        """
        if not n.tokens:
            return False
        self._write([Document(
            id=_example_id("user", n), kind="example", text=query_text(n), category=category,
            source="user", direction=n.direction, note=n.raw,
        )])
        return True

    def _write(self, docs: list[Document]) -> int:
        if not docs:
            return 0
        vectors = self.embedder.embed([d.text for d in docs], kind="document")
        self.store.upsert(docs, vectors, embedder_name=self.embedder.name)
        return len(docs)

    # --- retrieval ------------------------------------------------------------------------

    def _embed_queries(self, items: Sequence[Normalised]):
        # The LLM layer asks for examples and then facts for the same batch. Remember the last
        # batch's vectors so a paid embedding API is called once per batch, not twice. The tuple
        # is replaced in one assignment, so concurrent requests can't see a half-written entry.
        texts = tuple(query_text(n) for n in items)
        last = self._last_queries
        if last is not None and last[0] == texts:
            return last[1]
        vectors = self.embedder.embed(list(texts), kind="query")
        self._last_queries = (texts, vectors)
        return vectors

    def examples(self, items: Sequence[Normalised], k: int) -> list[list[Example]]:
        """Up to k labelled examples per item. Same signature as MLCategoriser.nearest, so it can
        be passed straight to LLMCategoriser as examples_fn."""
        if not items or k <= 0:
            return [[] for _ in items]
        dense = self.store.search(
            self._embed_queries(items), max(k, self.candidates), kind="example",
            min_score=self.min_example_score,
        )
        lexical = (self.lexical_fn(items, max(k, self.candidates)) if self.lexical_fn
                   else [[] for _ in items])

        out: list[list[Example]] = []
        for hits, lex in zip(dense, lexical):
            # User corrections are the most trustworthy labels, so they go first on a tie.
            dense_examples = [
                Example(hit.document.note or hit.document.text, hit.document.direction,
                        hit.document.category)
                for hit in sorted(hits, key=lambda h: (-h.score, h.document.source != "user"))
            ]
            # Key on the normalised form: the two searches may return different statement lines
            # for the same merchant, and those should count as agreement, not as two results.
            fused = reciprocal_rank_fusion(
                [dense_examples, lex], key=lambda e: normalise(e.description, e.direction).cache_key
            )
            out.append(fused[:k])
        return out

    def facts(self, items: Sequence[Normalised], k: int) -> list[list[MerchantNote]]:
        """Up to k knowledge-base facts per item, only when they're a close match."""
        if not items or k <= 0:
            return [[] for _ in items]
        results = self.store.search(
            self._embed_queries(items), k, kind="merchant", min_score=self.min_fact_score
        )
        return [
            [
                MerchantNote(
                    name=hit.document.id.removeprefix("kb|"),
                    category=hit.document.category,
                    note=hit.document.note or "",
                    score=round(hit.score, 3),
                )
                for hit in hits
            ]
            for hits in results
        ]
