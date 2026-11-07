import json

from txncat.cache import Cache
from txncat.categories import Category
from txncat.llm.fake import FakeLLM
from txncat.llm_categoriser import LLMCategoriser
from txncat.pipeline import Pipeline, Transaction


def txn(description: str, direction: str = "debit") -> Transaction:
    return Transaction(description=description, direction=direction)


def answer_all(category: str, confidence: float):
    def respond(_system: str, user: str) -> str:
        ids = [int(line.split(".")[0]) for line in user.splitlines() if line[:1].isdigit()]
        return json.dumps({"results": [{"id": i, "category": category, "confidence": confidence}
                                       for i in ids]})
    return respond


def test_rules_resolve_known_merchants():
    [r] = Pipeline().categorise([txn("UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA")])
    assert (r.category, r.source, r.needs_review) == (Category.FOOD_DINING, "rule", False)


def test_nothing_confident_goes_to_review():
    [r] = Pipeline().categorise([txn("POS 4XXXXXXXXXXX1234 THEOBROMA")])
    assert r.needs_review and r.source == "none" and r.category is Category.OTHER


def test_ml_threshold_decides_between_ml_and_review(trained_ml):
    t = txn("POS 4XXXXXXXXXXX9999 VERMA SWEETS")
    [accepted] = Pipeline(ml=trained_ml, use_rules=False, ml_threshold=0.0).categorise([t])
    assert accepted.source == "ml"
    [rejected] = Pipeline(ml=trained_ml, use_rules=False, ml_threshold=1.01).categorise([t])
    assert rejected.needs_review


def test_llm_handles_what_ml_was_unsure_about(trained_ml):
    llm = LLMCategoriser(FakeLLM(answer_all("food_dining", 0.9)), sleep=lambda _: None)
    [r] = Pipeline(ml=trained_ml, llm=llm, ml_threshold=1.01).categorise(
        [txn("POS 4XXXXXXXXXXX1234 THEOBROMA")]
    )
    assert (r.category, r.source) == (Category.FOOD_DINING, "llm")


def test_low_confidence_llm_answer_goes_to_review():
    llm = LLMCategoriser(FakeLLM(answer_all("food_dining", 0.3)), sleep=lambda _: None)
    [r] = Pipeline(llm=llm).categorise([txn("POS 4XXXXXXXXXXX1234 THEOBROMA")])
    assert r.needs_review


def test_duplicate_descriptions_are_sent_to_the_llm_once():
    fake = FakeLLM(answer_all("food_dining", 0.9))
    llm = LLMCategoriser(fake, sleep=lambda _: None)
    results = Pipeline(llm=llm).categorise([
        txn("UPI/DR/111111111111/THEOBROMA/YESB/theobroma@ybl/NA"),
        txn("UPI/DR/222222222222/THEOBROMA/YESB/theobroma@ybl/NA"),
    ])
    assert all(r.source == "llm" for r in results)
    assert len(fake.calls) == 1
    assert fake.calls[0][1].count("THEOBROMA/YESB") == 1


def test_cached_answers_skip_the_llm_next_time():
    fake = FakeLLM(answer_all("food_dining", 0.9))
    pipeline = Pipeline(llm=LLMCategoriser(fake, sleep=lambda _: None), cache=Cache())
    pipeline.categorise([txn("UPI/DR/111111111111/THEOBROMA/YESB/theobroma@ybl/NA")])
    [again] = pipeline.categorise([txn("UPI/DR/999999999999/THEOBROMA/YESB/theobroma@ybl/NA")])
    assert again.source == "cache" and again.category is Category.FOOD_DINING
    assert len(fake.calls) == 1


def test_an_empty_cache_is_still_used():
    # Regression: an empty Cache has len() == 0, so `if self.cache:` skipped it entirely.
    cache = Cache()
    assert len(cache) == 0
    llm = LLMCategoriser(FakeLLM(answer_all("food_dining", 0.9)), sleep=lambda _: None)
    Pipeline(llm=llm, cache=cache).categorise([txn("POS 4XXXXXXXXXXX1234 THEOBROMA")])
    assert len(cache) == 1


def test_descriptions_with_no_usable_words_are_never_cached():
    cache = Cache()
    llm = LLMCategoriser(FakeLLM(answer_all("p2p_transfer", 0.9)), sleep=lambda _: None)
    Pipeline(llm=llm, cache=cache).categorise([txn("UPI/DR/123456789012/9876543210@ybl")])
    assert len(cache) == 0


def test_transfers_with_no_usable_words_are_not_merged():
    # Regression: both normalise to an empty key, so deduplication used to send only the first.
    fake = FakeLLM(answer_all("p2p_transfer", 0.9))
    first = "UPI/DR/111111111111/9876543210@ybl/NA"
    second = "UPI/DR/222222222222/9123456780@ybl/NA"
    results = Pipeline(llm=LLMCategoriser(fake, sleep=lambda _: None)).categorise(
        [txn(first), txn(second)]
    )
    prompt = fake.calls[0][1]
    assert json.dumps(first) in prompt and json.dumps(second) in prompt
    assert all(r.source == "llm" for r in results)


def test_correction_overrides_rules_for_that_merchant():
    pipeline = Pipeline(cache=Cache())
    assert pipeline.record_correction(
        txn("UPI/DR/111111111111/SWIGGY/YESB/swiggy@ybl/NA"), Category.GROCERIES
    )
    [r] = pipeline.categorise([txn("UPI/DR/999999999999/SWIGGY/YESB/swiggy@ybl/NA")])
    assert (r.category, r.source, r.reason) == (Category.GROCERIES, "cache", "cache:user")


def test_correction_needs_a_cache_and_usable_words():
    t = txn("UPI/DR/111111111111/SWIGGY/YESB/swiggy@ybl/NA")
    assert not Pipeline().record_correction(t, Category.GROCERIES)
    nameless = txn("UPI/DR/123456789012/9876543210@ybl/NA")
    assert not Pipeline(cache=Cache()).record_correction(nameless, Category.RENT)


def test_output_order_matches_input_order(trained_ml):
    llm = LLMCategoriser(FakeLLM(answer_all("shopping", 0.9)), sleep=lambda _: None)
    descriptions = [
        "POS 4XXXXXXXXXXX1234 UNIQLO",
        "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA",
        "ATM WDL/123456/SAKET NEW DELHI",
    ]
    results = Pipeline(ml=trained_ml, llm=llm, ml_threshold=1.01).categorise(
        [txn(d) for d in descriptions]
    )
    assert [r.description for r in results] == descriptions
    assert [r.source for r in results] == ["llm", "rule", "rule"]


def test_layers_report_what_is_enabled(trained_ml):
    llm = LLMCategoriser(FakeLLM([]), sleep=lambda _: None)
    assert Pipeline(ml=trained_ml, llm=llm, cache=Cache()).layers == ["cache", "rules", "ml", "llm:fake"]
