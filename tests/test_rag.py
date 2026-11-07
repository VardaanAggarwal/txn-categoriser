"""Retrieval: embedders, the vector store, hybrid search, the knowledge base and the RAG prompt.

Everything here runs offline: the hashing embedder needs no model, and the Ollama and Gemini
embedders are tested against httpx's MockTransport.
"""

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from txncat.cache import Cache
from txncat.categories import Category
from txncat.embeddings import (
    GeminiEmbedder, HashingEmbedder, OllamaEmbedder, make_embedder, unit_rows,
)
from txncat.knowledge import MerchantFact, load_kb
from txncat.llm.base import LLMError
from txncat.llm.fake import FakeLLM
from txncat.llm_categoriser import LLMCategoriser
from txncat.normalise import normalise
from txncat.pipeline import Pipeline, Transaction
from txncat.rag import Retriever, ensure_indexed, reciprocal_rank_fusion
from txncat.vectorstore import Document, EmbedderMismatchError, VectorStore

KB_FILE = Path(__file__).resolve().parent.parent / "data" / "merchant_kb.jsonl"

FACTS = [
    MerchantFact("THEOBROMA", Category.FOOD_DINING, "Bakery and dessert cafe chain"),
    MerchantFact("CROMA", Category.SHOPPING, "Electronics retail chain"),
    MerchantFact("CULT FIT", Category.HEALTH, "Gyms and fitness memberships", ("CULTFIT",)),
]

DHABA = "UPI/DR/111122223333/SHARMA DHABA/YESB/sharmadhaba@okaxis/NA"
TRAIN = [
    (DHABA, Category.FOOD_DINING),
    ("UPI/DR/222233334444/RAJ KIRANA STORE/HDFC/rajkirana@oksbi/NA", Category.GROCERIES),
    ("UPI/DR/999988887777/SHARMA DHABA/YESB/sharmadhaba@okaxis/NA", Category.FOOD_DINING),
]


def train_items():
    return [(normalise(d, "debit"), c) for d, c in TRAIN]


def doc(doc_id: str, text: str, *, kind="example", source="train",
        category=Category.OTHER) -> Document:
    return Document(id=doc_id, kind=kind, text=text, category=category, source=source)


def reply(*results: tuple[int, str, float]) -> str:
    return json.dumps({"results": [{"id": i, "category": c, "confidence": p} for i, c, p in results]})


def no_sleep(_: float) -> None:
    pass


class CountingEmbedder(HashingEmbedder):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def embed(self, texts, *, kind="document"):
        self.calls.append(kind)
        return super().embed(texts, kind=kind)


# --- embedders ----------------------------------------------------------------------------


def test_hashing_embedder_is_unit_length_and_deterministic():
    e = HashingEmbedder()
    a = e.embed(["THEOBROMA", "CHAAYOS"])
    assert a.shape == (2, e.dim) and a.dtype == np.float32
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert np.array_equal(a, e.embed(["THEOBROMA", "CHAAYOS"]))
    assert e.embed([]).shape == (0, e.dim)


def test_unit_rows_leaves_zero_rows_alone():
    out = unit_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
    assert np.allclose(out, [[0.0, 0.0], [0.6, 0.8]])


def test_ollama_embedder_request_prefixes_and_normalisation():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        n = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * n})

    http = httpx.Client(base_url="http://localhost:11434", transport=httpx.MockTransport(handler))
    vectors = OllamaEmbedder(http=http).embed(["THEOBROMA"], kind="query")
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/api/embed"
    assert body == {"model": "nomic-embed-text", "input": ["search_query: THEOBROMA"]}
    assert np.allclose(vectors, [[0.6, 0.8]])


def test_gemini_embedder_batches_at_the_api_limit_and_sets_task_type():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        n = len(json.loads(request.content)["requests"])
        return httpx.Response(200, json={"embeddings": [{"values": [1.0, 0.0]}] * n})

    http = httpx.Client(base_url="https://example.test", headers={"x-goog-api-key": "k"},
                        transport=httpx.MockTransport(handler))
    out = GeminiEmbedder(api_key="k", http=http).embed([f"M{i}" for i in range(150)])
    assert out.shape == (150, 2)
    assert len(seen) == 2  # 100 + 50
    assert seen[0].url.path == "/v1beta/models/gemini-embedding-001:batchEmbedContents"
    assert "key=" not in str(seen[0].url)
    first = json.loads(seen[0].content)["requests"][0]
    assert first["taskType"] == "RETRIEVAL_DOCUMENT" and first["outputDimensionality"] == 768


def test_embedder_rejects_a_reply_with_the_wrong_number_of_vectors():
    http = httpx.Client(
        base_url="http://localhost:11434",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"embeddings": []})),
    )
    with pytest.raises(LLMError):
        OllamaEmbedder(http=http).embed(["A"])


def test_make_embedder_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_embedder("word2vec")


# --- vector store -------------------------------------------------------------------------


def test_store_returns_the_best_match_first():
    e, store = HashingEmbedder(), VectorStore()
    docs = [doc("a", "THEOBROMA"), doc("b", "CROMA"), doc("c", "RAJ KIRANA STORE")]
    store.upsert(docs, e.embed([d.text for d in docs]), embedder_name=e.name)
    [hits] = store.search(e.embed(["THEOBROMA BAKERY"], kind="query"), 2)
    assert hits[0].document.id == "a"
    assert hits[0].score > hits[1].score


def test_upsert_replaces_a_document_with_the_same_id():
    e, store = HashingEmbedder(), VectorStore()
    store.upsert([doc("a", "CROMA")], e.embed(["CROMA"]), embedder_name=e.name)
    store.upsert([doc("a", "CROMA", category=Category.SHOPPING)], e.embed(["CROMA"]),
                 embedder_name=e.name)
    assert len(store) == 1
    [[hit]] = store.search(e.embed(["CROMA"]), 5)
    assert hit.document.category is Category.SHOPPING


def test_store_persists_and_refuses_vectors_from_another_embedder(tmp_path):
    path, e = str(tmp_path / "index.db"), HashingEmbedder()
    store = VectorStore(path)
    store.upsert([doc("a", "THEOBROMA")], e.embed(["THEOBROMA"]), embedder_name=e.name)
    store.close()

    reopened = VectorStore(path)
    assert len(reopened) == 1 and reopened.embedder_name == e.name
    [[hit]] = reopened.search(e.embed(["THEOBROMA"]), 1)
    assert hit.document.id == "a" and hit.score == pytest.approx(1.0, abs=1e-5)

    other = HashingEmbedder(dim=256)
    with pytest.raises(EmbedderMismatchError):
        reopened.upsert([doc("b", "CROMA")], other.embed(["CROMA"]), embedder_name=other.name)


def test_retriever_refuses_an_index_built_by_another_embedder():
    e, store = HashingEmbedder(), VectorStore()
    store.upsert([doc("a", "CROMA")], e.embed(["CROMA"]), embedder_name=e.name)
    with pytest.raises(EmbedderMismatchError):
        Retriever(store, HashingEmbedder(ngram_range=(2, 5)))  # same dimension, different model


def test_search_filters_by_kind_and_similarity_floor():
    e, store = HashingEmbedder(), VectorStore()
    docs = [doc("fact", "THEOBROMA", kind="merchant", source="kb"), doc("ex", "THEOBROMA")]
    store.upsert(docs, e.embed([d.text for d in docs]), embedder_name=e.name)
    [hits] = store.search(e.embed(["THEOBROMA"]), 5, kind="merchant")
    assert [h.document.id for h in hits] == ["fact"]
    assert store.search(e.embed(["QWXZ PLKJ"]), 5, min_score=0.9) == [[]]


def test_empty_store_returns_no_hits():
    assert VectorStore().search(HashingEmbedder().embed(["A B", "C D"]), 3) == [[], []]


def test_delete_source_keeps_other_sources():
    e, store = HashingEmbedder(), VectorStore()
    docs = [doc("t", "CROMA", source="train"), doc("u", "CROMA STORE", source="user")]
    store.upsert(docs, e.embed([d.text for d in docs]), embedder_name=e.name)
    assert store.delete_source("train") == 1
    [hits] = store.search(e.embed(["CROMA"]), 5)
    assert [h.document.id for h in hits] == ["u"]


# --- hybrid retrieval ---------------------------------------------------------------------


def test_rrf_rewards_agreement_between_rankings():
    assert reciprocal_rank_fusion([["a", "b"], ["b", "c"]], key=lambda x: x) == ["b", "a", "c"]


def test_examples_are_indexed_once_per_merchant():
    r = Retriever(VectorStore(), HashingEmbedder())
    assert r.index_examples(train_items()) == 2  # the two SHARMA DHABA rows share a key


def test_examples_find_a_similar_merchant():
    r = Retriever(VectorStore(), HashingEmbedder(), min_example_score=0.0)
    r.index_examples(train_items())
    [[top, *_]] = r.examples([normalise("UPI/DR/1/VERMA DHABA/YESB/vermadhaba@ybl/NA", "debit")], 2)
    assert top.category is Category.FOOD_DINING


def test_hybrid_examples_merge_dense_and_lexical_results(trained_ml):
    r = Retriever(VectorStore(), HashingEmbedder(), lexical_fn=trained_ml.nearest)
    r.index_examples(train_items())
    [examples] = r.examples([normalise("POS 4XXXXXXXXXXX1234 BLINKIT", "debit")], 3)
    # BLINKIT is only in the ML model's training data, so it can only come from lexical search.
    assert any("BLINKIT" in e.description for e in examples)


def test_facts_are_returned_only_for_close_matches():
    r = Retriever(VectorStore(), HashingEmbedder())
    r.index_facts(FACTS)
    notes = r.facts([
        normalise("POS 4XXXXXXXXXXX1234 THEOBROMA", "debit"),
        normalise("UPI/DR/5/QWXZ PLKJ/YESB/qwxz@ybl/NA", "debit"),
    ], 1)
    assert notes[0][0].name == "THEOBROMA" and notes[0][0].category is Category.FOOD_DINING
    assert notes[1] == []


def test_a_correction_outranks_the_training_label_for_the_same_merchant():
    r = Retriever(VectorStore(), HashingEmbedder(), min_example_score=0.0)
    n = normalise(DHABA, "debit")
    r.index_examples([(n, Category.FOOD_DINING)])
    assert r.add_correction(n, Category.GROCERIES)
    [[top]] = r.examples([n], 1)
    assert top.category is Category.GROCERIES
    assert r.store.count(source="user") == 1


def test_a_correction_with_no_usable_words_is_rejected():
    r = Retriever(VectorStore(), HashingEmbedder())
    assert not r.add_correction(normalise("UPI/DR/123456789012/9876543210@ybl/NA", "debit"),
                                Category.RENT)


def test_queries_are_embedded_once_per_batch():
    embedder = CountingEmbedder()
    r = Retriever(VectorStore(), embedder)
    r.index_examples(train_items())
    r.index_facts(FACTS)
    embedder.calls.clear()
    batch = [normalise("POS 4XXXXXXXXXXX1234 THEOBROMA", "debit")]
    r.examples(batch, 2)
    r.facts(batch, 2)
    assert embedder.calls == ["query"]


def test_ensure_indexed_reuses_an_index_and_refresh_keeps_corrections():
    r = Retriever(VectorStore(), HashingEmbedder())
    items = train_items()
    assert ensure_indexed(r, examples=items, facts=FACTS) == {"train": 2, "kb": 3}
    assert ensure_indexed(r, examples=items, facts=FACTS) == {"train": 0, "kb": 0}
    r.add_correction(items[0][0], Category.GROCERIES)
    ensure_indexed(r, examples=items, facts=FACTS, refresh=True)
    assert r.store.count(source="user") == 1
    assert r.store.count(source="train") == 2 and r.store.count(source="kb") == 3


# --- knowledge base -----------------------------------------------------------------------


def test_the_shipped_knowledge_base_is_valid():
    facts = load_kb(KB_FILE)
    assert len(facts) >= 50
    assert all(f.name == f.name.upper() for f in facts)


def test_load_kb_rejects_an_unknown_category(tmp_path):
    bad = tmp_path / "kb.jsonl"
    bad.write_text('{"name": "ACME", "category": "luxury", "note": "Not a real category"}\n')
    with pytest.raises(ValueError, match="kb.jsonl:1"):
        load_kb(bad)


def test_load_kb_rejects_duplicates(tmp_path):
    dup = tmp_path / "kb.jsonl"
    line = '{"name": "CROMA", "category": "shopping", "note": "Electronics"}\n'
    dup.write_text(line + line)
    with pytest.raises(ValueError, match="duplicate"):
        load_kb(dup)


# --- the LLM prompt and the pipeline ------------------------------------------------------


def test_prompt_includes_retrieved_merchant_facts():
    r = Retriever(VectorStore(), HashingEmbedder())
    r.index_facts(FACTS)
    llm = FakeLLM([reply((0, "food_dining", 0.9))])
    LLMCategoriser(llm, examples_fn=r.examples, facts_fn=r.facts, sleep=no_sleep).categorise(
        [normalise("POS 4XXXXXXXXXXX1234 THEOBROMA", "debit")]
    )
    prompt = llm.calls[0][1]
    assert "Merchant facts from the knowledge base:" in prompt
    assert '"THEOBROMA": Bakery and dessert cafe chain -> food_dining' in prompt


def test_failed_retrieval_still_sends_the_batch():
    def broken(_items, _k):
        raise LLMError("embedding API is down")

    llm = FakeLLM([reply((0, "food_dining", 0.9))])
    [prediction] = LLMCategoriser(
        llm, examples_fn=broken, facts_fn=broken, sleep=no_sleep
    ).categorise([normalise("THEOBROMA", "debit")])
    assert prediction.category is Category.FOOD_DINING
    assert "Merchant facts" not in llm.calls[0][1]


def test_pipeline_corrections_reach_the_retriever():
    r = Retriever(VectorStore(), HashingEmbedder())
    t = Transaction(description=DHABA, direction="debit")
    pipeline = Pipeline(cache=Cache(), retriever=r)
    assert pipeline.record_correction(t, Category.GROCERIES)
    assert r.store.count(source="user") == 1
    assert pipeline.layers[-1] == f"rag:{r.embedder.name}"
    # A retriever alone is somewhere to store a correction, even without a cache.
    assert Pipeline(retriever=Retriever(VectorStore(), HashingEmbedder())).record_correction(
        t, Category.GROCERIES
    )
