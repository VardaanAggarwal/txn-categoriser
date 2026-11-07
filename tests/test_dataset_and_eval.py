from txncat.categories import Category
from txncat.dataset import generate, load_csv, merchant_disjoint_split, random_split, write_csv
from txncat.evaluate import evaluate_split, score
from txncat.llm.fake import FakeLLM
from txncat.pipeline import CategorisedTransaction


def test_generation_is_deterministic():
    assert generate(seed=1) == generate(seed=1)
    assert generate(seed=1) != generate(seed=2)


def test_every_category_except_other_is_present():
    present = {r.category for r in generate()}
    assert present == set(Category) - {Category.OTHER}


def test_csv_roundtrip(tmp_path):
    rows = generate()[:20]
    write_csv(rows, tmp_path / "d.csv")
    assert load_csv(tmp_path / "d.csv") == rows


def test_merchant_disjoint_split_shares_no_merchants_and_keeps_every_category():
    rows = generate()
    train, test = merchant_disjoint_split(rows)
    assert not {r.merchant_id for r in train} & {r.merchant_id for r in test}
    assert {r.category for r in train} == {r.category for r in test}
    assert len(train) + len(test) == len(rows)


def test_random_split_sizes():
    train, test = random_split(generate(), test_fraction=0.25)
    assert abs(len(test) / (len(train) + len(test)) - 0.25) < 0.01


def _result(category: Category, needs_review: bool = False) -> CategorisedTransaction:
    return CategorisedTransaction(
        description="x", category=category, confidence=1.0, source="rule",
        needs_review=needs_review, reason="", normalised_key="x",
    )


def test_score_counts_review_as_not_auto_resolved_and_not_correct():
    rows = generate()[:4]
    results = [_result(rows[0].category), _result(rows[1].category),
               _result(Category.OTHER, needs_review=True), _result(Category.OTHER)]
    s = score("m", rows, results, seconds=1.0)
    assert s.auto_resolved == 0.75
    assert s.accuracy == 0.5
    assert abs(s.precision_auto - 2 / 3) < 1e-9


def test_evaluate_split_reports_every_mode_including_llm():
    rows = generate()
    train, test = merchant_disjoint_split(rows)
    llm = FakeLLM(lambda _s, _u: '{"results": []}')  # always unhelpful
    report = evaluate_split(train, test[:50], llm_client=llm)
    modes = [m["mode"] for m in report["modes"]]
    assert modes[0] == "rules" and modes[1] == "ml" and modes[-1] == "rules+ml+llm (fake)"
    assert report["modes"][-1]["llm"]["calls"] > 0
    assert len(report["threshold_sweep"]) == 8
