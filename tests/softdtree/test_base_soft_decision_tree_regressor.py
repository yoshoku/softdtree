import numpy as np
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

from softdtree import BaseSoftDecisionTreeRegressor


class Regressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    def __init__(self) -> None:
        self.reg = BaseSoftDecisionTreeRegressor(max_depth=4, eta=0.1, random_seed=42)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor":
        if y.ndim == 1:
            y = np.array([y]).T
        self.reg.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y = self.reg.decision_function(X)
        if y.shape[1] == 1:
            y = y[:, 0]
        return y

    def size(self) -> int:
        return self.reg.size()

def test_soft_decision_tree_regressor() -> None:
    X, y = make_regression(n_samples=100, n_targets=1, n_features=2, bias=20.0, noise=2.0, random_state=42)
    y = scale(y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Regressor()),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95

def test_soft_decision_tree_multi_regressor() -> None:
    X, y = make_regression(n_samples=100, n_targets=2, n_features=2, bias=20.0, noise=2.0, random_state=42)
    y = scale(y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Regressor()),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95
