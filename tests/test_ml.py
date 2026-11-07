import pytest

from txncat.categories import Category
from txncat.ml import MLCategoriser
from txncat.normalise import normalise


def test_predicts_a_category_with_a_probability(trained_ml):
    [(category, prob)] = trained_ml.predict([normalise("NEFT CR-SBIN0001111-ORBIT LLP-SALARY JAN 2026", "credit")])
    assert category is Category.INCOME
    assert 0.0 < prob <= 1.0


def test_generalises_from_character_ngrams(trained_ml):
    [(category, _)] = trained_ml.predict([normalise("POS 4XXXXXXXXXXX9999 VERMA DHABA", "debit")])
    assert category is Category.FOOD_DINING


def test_nearest_returns_similar_labelled_examples(trained_ml):
    [neighbours] = trained_ml.nearest([normalise("UPI/DR/1/SWIGGY/YESB/swiggy@ybl/NA", "debit")], k=2)
    assert len(neighbours) == 2
    assert neighbours[0].category is Category.FOOD_DINING
    assert "SWIGGY" in neighbours[0].description


def test_empty_input():
    model = MLCategoriser()
    assert model.predict([]) == []
    assert model.nearest([]) == []


def test_fit_rejects_mismatched_input():
    with pytest.raises(ValueError):
        MLCategoriser().fit([normalise("X")], [])


def test_save_and_load_roundtrip(trained_ml, tmp_path):
    path = tmp_path / "model.joblib"
    trained_ml.save(path)
    loaded = MLCategoriser.load(path)
    item = [normalise("UPI/DR/1/RAJ KIRANA STORE/HDFC/rajkirana@oksbi/NA", "debit")]
    assert loaded.predict(item) == trained_ml.predict(item)
