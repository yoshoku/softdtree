import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin
from sklearn.datasets import load_digits, make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.multiclass import unique_labels

from softdtree._softdtree import BaseSoftDecisionTreeClassifier


class Classifier(BaseEstimator, MultiOutputMixin, ClassifierMixin):
    def __init__(self) -> None:
        self.clf = BaseSoftDecisionTreeClassifier(
            max_depth=8, batch_size=10, eta=0.1, max_epoch=50, random_seed=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier":
        self.classes_ = unique_labels(y)
        if len(self.classes_) > 2:
            y = self._one_hot_encode(y)
        if y.ndim == 1:
            y = np.array([y], dtype=np.float64).T
        self.clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        r = np.zeros(n_samples, dtype=self.classes_.dtype)
        pred = self.clf.decision_function(X)
        for i in range(n_samples):
            if pred.shape[1] > 1:
                r[i] = self.classes_[pred[i].argmax()]
            else:
                r[i] = self.classes_[1] if pred[i][0] > 0.5 else self.classes_[0]
        return r

    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        return label_binarize(y, classes=self.classes_).astype(np.float64)

def test_base_soft_decision_tree_classifier() -> None:
    X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Classifier()),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95

def test_base_soft_decision_tree_multi_classifier() -> None:
    X, y = load_digits(n_class=3, return_X_y=True)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Classifier()),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95
