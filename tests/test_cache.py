from txncat.cache import Cache
from txncat.categories import Category


def test_put_and_get():
    cache = Cache()
    assert cache.get("debit|upi|THEOBROMA") is None
    cache.put("debit|upi|THEOBROMA", Category.FOOD_DINING, "llm", 0.9)
    hit = cache.get("debit|upi|THEOBROMA")
    assert (hit.category, hit.source, hit.confidence) == (Category.FOOD_DINING, "llm", 0.9)
    assert (cache.hits, cache.misses) == (1, 1)


def test_put_overwrites():
    cache = Cache()
    cache.put("k", Category.OTHER, "ml", 0.6)
    cache.put("k", Category.RENT, "llm", 0.8)
    assert cache.get("k").category is Category.RENT
    assert len(cache) == 1


def test_persists_to_disk(tmp_path):
    path = str(tmp_path / "cache.db")
    first = Cache(path)
    first.put("k", Category.RENT, "llm", 0.8)
    first.close()
    assert Cache(path).get("k").category is Category.RENT
