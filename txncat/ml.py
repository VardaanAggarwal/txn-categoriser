"""Classical ML layer: character n-gram TF-IDF with logistic regression.

Character n-grams cope with what bank descriptions actually look like: truncated names
("SWIGGY INSTAMA"), names glued together in UPI IDs ("rajkirana12") and spelling variants
("MEDICOS", "MEDICAL"). The same vectors are reused to retrieve similar labelled examples for the
LLM prompt.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .categories import Category
from .normalise import Normalised


@dataclass(frozen=True)
class Example:
    description: str
    direction: str | None
    category: Category


class MLCategoriser:
    def __init__(self, c: float = 10.0) -> None:
        self.vectoriser = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, lowercase=False
        )
        self.model = LogisticRegression(C=c, max_iter=3000)
        self._train_matrix = None
        self._examples: list[Example] = []

    @property
    def is_fitted(self) -> bool:
        return self._train_matrix is not None

    def fit(self, items: Sequence[Normalised], labels: Sequence[Category]) -> "MLCategoriser":
        if len(items) != len(labels) or not items:
            raise ValueError("need the same, non-zero number of items and labels")
        self._train_matrix = self.vectoriser.fit_transform([n.features for n in items])
        self.model.fit(self._train_matrix, [str(label) for label in labels])
        self._examples = [Example(n.raw, n.direction, Category(lbl)) for n, lbl in zip(items, labels)]
        return self

    def predict(self, items: Sequence[Normalised]) -> list[tuple[Category, float]]:
        """Most likely category and its probability for each item."""
        if not items:
            return []
        probs = self.model.predict_proba(self.vectoriser.transform([n.features for n in items]))
        best = probs.argmax(axis=1)
        return [
            (Category(self.model.classes_[j]), float(probs[i, j])) for i, j in enumerate(best)
        ]

    def nearest(self, items: Sequence[Normalised], k: int = 3) -> list[list[Example]]:
        """The k most similar labelled training examples for each item (cosine similarity)."""
        if not items or self._train_matrix is None:
            return [[] for _ in items]
        # TF-IDF rows are L2-normalised, so the dot product is the cosine similarity.
        sims = (self.vectoriser.transform([n.features for n in items]) @ self._train_matrix.T).toarray()
        k = min(k, sims.shape[1])
        top = np.argsort(-sims, axis=1)[:, :k]
        return [[self._examples[j] for j in row] for row in top]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "MLCategoriser":
        """Load a model saved by `save`. Only load files you created: joblib uses pickle."""
        model = joblib.load(path)
        if not isinstance(model, MLCategoriser):
            raise TypeError(f"{path} does not contain an MLCategoriser")
        return model
