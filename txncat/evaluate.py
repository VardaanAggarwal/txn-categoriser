"""Measures each layer and each combination of layers on held-out data.

Every number in the README comes from here. Two definitions matter:
- auto-resolved: the share of transactions the system categorised without sending them to review.
- precision on auto-resolved: of those, the share that were correct.
End-to-end accuracy counts every transaction sent to review as wrong.
"""

import json
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sklearn.metrics import f1_score, precision_recall_fscore_support

from .cache import Cache
from .dataset import Row, merchant_disjoint_split, random_split
from .embeddings import Embedder
from .llm.base import LLMClient
from .llm_categoriser import LLMCategoriser
from .ml import MLCategoriser
from .normalise import normalise
from .pipeline import CategorisedTransaction, Pipeline, Transaction
from .rag import build_retriever

REVIEW = "needs_review"
THRESHOLDS = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


@dataclass
class ModeScore:
    mode: str
    n: int
    auto_resolved: float
    precision_auto: float
    accuracy: float
    macro_f1: float
    sources: dict[str, int]
    seconds: float
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    top_confusions: list[tuple[str, str, int]] = field(default_factory=list)
    sent_to_review: dict[str, int] = field(default_factory=dict)
    llm: dict[str, float] | None = None


class CountingClient:
    """Wraps an LLM client to count calls and time them."""

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.name = inner.name
        self.latencies: list[float] = []

    def complete_json(self, system: str, user: str) -> str:
        start = time.perf_counter()
        try:
            return self.inner.complete_json(system, user)
        finally:
            self.latencies.append(time.perf_counter() - start)


def _transactions(rows: Sequence[Row]) -> list[Transaction]:
    return [Transaction(description=r.description, direction=r.direction) for r in rows]


def score(
    mode: str, rows: Sequence[Row], results: Sequence[CategorisedTransaction], seconds: float
) -> ModeScore:
    y_true = [r.category.value for r in rows]
    y_pred = [REVIEW if res.needs_review else res.category.value for res in results]
    n = len(rows)
    auto = [i for i in range(n) if y_pred[i] != REVIEW]
    correct = sum(y_true[i] == y_pred[i] for i in range(n))
    labels = sorted(set(y_true))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    # Confident wrong answers and "not sure" are different failures, so they're counted apart.
    confusions = Counter((t, p) for t, p in zip(y_true, y_pred) if t != p and p != REVIEW)
    reviewed = Counter(t for t, p in zip(y_true, y_pred) if p == REVIEW)
    return ModeScore(
        mode=mode,
        n=n,
        auto_resolved=len(auto) / n,
        precision_auto=(sum(y_true[i] == y_pred[i] for i in auto) / len(auto)) if auto else 0.0,
        accuracy=correct / n,
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        sources=dict(Counter(res.source for res in results)),
        seconds=seconds,
        per_category={
            label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for label, p, r, f, s in zip(labels, precision, recall, f1, support)
        },
        top_confusions=[(t, p, c) for (t, p), c in confusions.most_common(8)],
        sent_to_review=dict(reviewed.most_common()),
    )


def _run(mode: str, pipeline: Pipeline, rows: Sequence[Row]) -> ModeScore:
    start = time.perf_counter()
    results = pipeline.categorise(_transactions(rows))
    return score(mode, rows, results, time.perf_counter() - start)


def train_ml(rows: Sequence[Row]) -> MLCategoriser:
    return MLCategoriser().fit([normalise(r.description, r.direction) for r in rows],
                               [r.category for r in rows])


def evaluate_split(
    train: Sequence[Row],
    test: Sequence[Row],
    *,
    ml_threshold: float = 0.6,
    llm_client: LLMClient | None = None,
    embedder: Embedder | None = None,
    kb_path: str | Path | None = None,
) -> dict:
    ml = train_ml(train)
    retrieval: dict | None = None
    modes = [
        _run("rules", Pipeline(), test),
        _run("ml", Pipeline(ml=ml, use_rules=False, ml_threshold=0.0), test),
        _run(f"rules+ml (threshold {ml_threshold})", Pipeline(ml=ml, ml_threshold=ml_threshold), test),
    ]
    if llm_client is not None:
        counting = CountingClient(llm_client)
        llm = LLMCategoriser(counting, examples_fn=ml.nearest)
        result = _run(
            f"rules+ml+llm ({llm_client.name})",
            Pipeline(ml=ml, llm=llm, ml_threshold=ml_threshold),
            test,
        )
        result.llm = _latency(counting)
        modes.append(result)

    if llm_client is not None and embedder is not None:
        # The index holds training rows only, never test rows, so retrieval can't leak labels.
        # The knowledge base is separate, hand-written data; see the README on what that means.
        retriever = build_retriever(embedder, lexical_fn=ml.nearest, rows=train, kb_path=kb_path)
        counting = CountingClient(llm_client)
        llm = LLMCategoriser(counting, examples_fn=retriever.examples, facts_fn=retriever.facts)
        result = _run(
            f"rules+ml+llm+rag ({llm_client.name}, {embedder.name})",
            Pipeline(ml=ml, llm=llm, ml_threshold=ml_threshold),
            test,
        )
        result.llm = _latency(counting)
        modes.append(result)
        norms = [normalise(r.description, r.direction) for r in test]
        with_fact = sum(1 for notes in retriever.facts(norms, 1) if notes)
        retrieval = {
            "embedder": embedder.name,
            "indexed_examples": retriever.store.count(source="train"),
            "indexed_facts": retriever.store.count(source="kb"),
            "test_rows_with_a_fact_retrieved": with_fact / len(test) if test else 0.0,
        }

    sweep = []
    for t in THRESHOLDS:
        s = _run(f"t={t}", Pipeline(ml=ml, ml_threshold=t), test)
        sweep.append({"threshold": t, "auto_resolved": s.auto_resolved,
                      "precision_auto": s.precision_auto})
    out = {"train_size": len(train), "test_size": len(test),
           "modes": [asdict(m) for m in modes], "threshold_sweep": sweep}
    if retrieval is not None:
        out["retrieval"] = retrieval
    return out


def _latency(counting: CountingClient) -> dict[str, float]:
    lat = sorted(counting.latencies)
    return {
        "calls": len(lat),
        "p50_seconds": statistics.median(lat) if lat else 0.0,
        "p95_seconds": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else 0.0,
    }


def measure_speed(rows: Sequence[Row], repeats: int = 5) -> dict:
    """Throughput of rules + ML, without cache or LLM."""
    pipeline = Pipeline(ml=train_ml(rows))
    txns = _transactions(rows)
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        pipeline.categorise(txns)
        timings.append(time.perf_counter() - start)
    seconds = statistics.median(timings)
    return {"transactions": len(rows), "median_seconds": seconds,
            "transactions_per_second": len(rows) / seconds}


def measure_cache(rows: Sequence[Row]) -> dict:
    """Process transactions one at a time, as they'd arrive, and count cache hits.

    Only transactions that reach the model layers are counted: rules never touch the cache.
    """
    train, _ = random_split(rows)
    pipeline = Pipeline(ml=train_ml(train), cache=Cache(), ml_threshold=0.0)
    model_path = Counter()
    for txn in _transactions(rows):
        model_path[pipeline.categorise([txn])[0].source] += 1
    looked_up = model_path["cache"] + model_path["ml"]
    return {
        "transactions": len(rows),
        "resolved_by_rules": model_path["rule"],
        "reached_model_layers": looked_up,
        "served_from_cache": model_path["cache"],
        "cache_hit_rate": model_path["cache"] / looked_up if looked_up else 0.0,
    }


def run_all(
    rows: Sequence[Row], *, ml_threshold: float = 0.6, llm_client: LLMClient | None = None,
    embedder: Embedder | None = None, kb_path: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    llm_args = {"llm_client": llm_client, "embedder": embedder, "kb_path": kb_path}
    log("Evaluating on unseen merchants...")
    unseen = evaluate_split(*merchant_disjoint_split(rows), ml_threshold=ml_threshold, **llm_args)
    log("Evaluating on a random split...")
    seen = evaluate_split(*random_split(rows), ml_threshold=ml_threshold, **llm_args)
    log("Measuring speed and cache...")
    return {
        "dataset_size": len(rows),
        "ml_threshold": ml_threshold,
        "unseen_merchants": unseen,
        "random_split": seen,
        "speed": measure_speed(rows),
        "cache": measure_cache(rows),
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _modes_table(split: dict) -> list[str]:
    lines = [
        f"Train: {split['train_size']} transactions · Test: {split['test_size']} transactions",
        "",
        "| Mode | Auto-resolved | Precision on auto-resolved | End-to-end accuracy | Macro-F1 |",
        "|---|---|---|---|---|",
    ]
    for m in split["modes"]:
        lines.append(f"| {m['mode']} | {_pct(m['auto_resolved'])} | {_pct(m['precision_auto'])} "
                     f"| {_pct(m['accuracy'])} | {m['macro_f1']:.3f} |")
    return lines


def to_markdown(report: dict) -> str:
    unseen, seen = report["unseen_merchants"], report["random_split"]
    cascade = unseen["modes"][2]
    lines = [
        "# Evaluation results",
        "",
        "Generated by `python -m txncat eval`. Do not edit by hand.",
        "",
        f"Dataset: {report['dataset_size']} synthetic transactions (see README for its limits). "
        f"ML confidence threshold: {report['ml_threshold']}.",
        "",
        "## Unseen merchants (headline)",
        "",
        "No merchant or person in the test set appears in training.",
        "",
        *_modes_table(unseen),
        "",
        "## Random split (merchants mostly seen before)",
        "",
        *_modes_table(seen),
        "",
        "## Coverage vs precision, rules + ML, unseen merchants",
        "",
        "| ML threshold | Auto-resolved | Precision on auto-resolved |",
        "|---|---|---|",
        *[f"| {s['threshold']} | {_pct(s['auto_resolved'])} | {_pct(s['precision_auto'])} |"
          for s in unseen["threshold_sweep"]],
        "",
        f"## Per category: {cascade['mode']}, unseen merchants",
        "",
        "| Category | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|",
        *[f"| {c} | {_pct(v['precision'])} | {_pct(v['recall'])} | {v['f1']:.3f} | {v['support']} |"
          for c, v in cascade["per_category"].items()],
        "",
        "## Confident wrong answers (true → predicted)",
        "",
        *([f"- {t} → {p}: {c}" for t, p, c in cascade["top_confusions"]] or ["- None"]),
        "",
        "## Sent to review, by true category",
        "",
        *[f"- {t}: {c}" for t, c in cascade["sent_to_review"].items()],
        "",
        "## Speed and cache",
        "",
        f"- Rules + ML: {report['speed']['transactions_per_second']:,.0f} transactions/second "
        f"on an Apple M1 laptop ({report['speed']['transactions']} transactions, median of 5 runs, "
        "no cache, no LLM).",
        f"- Cache: {_pct(report['cache']['cache_hit_rate'])} of the "
        f"{report['cache']['reached_model_layers']} transactions that got past the rules were "
        "served from cache when processed one at a time.",
    ]
    llm_modes = [m for m in unseen["modes"] if m.get("llm")]
    for m in llm_modes:
        lines += ["", f"## LLM calls: {m['mode']}", "",
                  f"- Calls: {m['llm']['calls']} · p50 {m['llm']['p50_seconds']:.2f}s · "
                  f"p95 {m['llm']['p95_seconds']:.2f}s"]
    if (r := unseen.get("retrieval")) is not None:
        lines += ["", "## Retrieval (RAG), unseen merchants", "",
                  f"- Embedder: {r['embedder']}",
                  f"- Indexed: {r['indexed_examples']} training examples, "
                  f"{r['indexed_facts']} merchant facts",
                  f"- Test transactions with at least one merchant fact retrieved: "
                  f"{_pct(r['test_rows_with_a_fact_retrieved'])}. The knowledge base is "
                  "hand-written and covers many merchants in this synthetic data, so this mode "
                  "measures the LLM *with* a good merchant directory, not unaided generalisation."]
    if not llm_modes:
        lines += ["", "## LLM layer", "",
                  "Not measured in this run. Run `python -m txncat eval --llm ollama` with a local "
                  "model to add it."]
    return "\n".join(lines) + "\n"


def write_report(report: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    (out / "RESULTS.md").write_text(to_markdown(report))
