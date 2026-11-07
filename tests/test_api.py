from fastapi.testclient import TestClient

from txncat.api import MAX_BATCH, create_app
from txncat.cache import Cache
from txncat.pipeline import Pipeline


def client(trained_ml) -> TestClient:
    return TestClient(create_app(Pipeline(ml=trained_ml, cache=Cache())))


def test_healthz(trained_ml):
    with client(trained_ml) as c:
        response = c.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "layers": ["cache", "rules", "ml"]}


def test_categorise(trained_ml):
    body = {"transactions": [
        {"description": "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA", "direction": "debit"},
        {"description": "ATM WDL/123456/SAKET NEW DELHI"},
    ]}
    with client(trained_ml) as c:
        response = c.post("/v1/categorise", json=body)
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["category"] for r in results] == ["food_dining", "cash_withdrawal"]
    assert results[0]["reason"] == "merchant:SWIGGY"


def test_rejects_empty_and_oversized_batches(trained_ml):
    one = {"description": "POS 4XXXXXXXXXXX1234 CROMA"}
    with client(trained_ml) as c:
        assert c.post("/v1/categorise", json={"transactions": []}).status_code == 422
        too_many = {"transactions": [one] * (MAX_BATCH + 1)}
        assert c.post("/v1/categorise", json=too_many).status_code == 422


def test_correction_changes_the_next_answer(trained_ml):
    swiggy = {"description": "UPI/DR/412345678901/SWIGGY/YESB/swiggy@ybl/NA", "direction": "debit"}
    with client(trained_ml) as c:
        stored = c.post("/v1/corrections", json={"transaction": swiggy, "category": "groceries"})
        assert stored.status_code == 200 and stored.json() == {"stored": True}
        [result] = c.post("/v1/categorise", json={"transactions": [swiggy]}).json()["results"]
    assert (result["category"], result["reason"]) == ("groceries", "cache:user")


def test_correction_without_usable_words_is_rejected(trained_ml):
    nameless = {"description": "UPI/DR/123456789012/9876543210@ybl/NA", "direction": "debit"}
    with client(trained_ml) as c:
        response = c.post("/v1/corrections", json={"transaction": nameless, "category": "rent"})
    assert response.status_code == 422


def test_correction_with_unknown_category_is_rejected(trained_ml):
    body = {"transaction": {"description": "POS 4XXXXXXXXXXX1234 CROMA"}, "category": "luxury"}
    with client(trained_ml) as c:
        assert c.post("/v1/corrections", json=body).status_code == 422


def test_rejects_bad_direction(trained_ml):
    body = {"transactions": [{"description": "X", "direction": "sideways"}]}
    with client(trained_ml) as c:
        assert c.post("/v1/categorise", json=body).status_code == 422
