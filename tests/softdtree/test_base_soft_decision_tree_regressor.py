from softdtree import BaseSoftDecisionTreeRegressor
from sklearn.base import BaseEstimator, MultiOutputMixin, RegressorMixin
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

import numpy as np


class Regressor(BaseEstimator, MultiOutputMixin, RegressorMixin):
    def __init__(self):
        self.reg = BaseSoftDecisionTreeRegressor(max_depth=4, eta=0.1, random_seed=42)

    def fit(self, X, y):
        if y.ndim == 1:
            y = np.array([y]).T
        self.reg.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        y = self.reg.decision_function(X)
        if y.shape[1] == 1:
            y = y[:, 0]
        return y

    def size(self):
        return self.reg.size()

def test_soft_decision_tree_regressor():
    X, y = make_regression(n_samples=100, n_targets=1, n_features=2, bias=20.0, noise=2.0, random_state=42)
    y = scale(y)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Regressor())
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95

def test_soft_decision_tree_multi_regressor():
    X, y = make_regression(n_samples=100, n_targets=2, n_features=2, bias=20.0, noise=2.0, random_state=42)
    y = scale(y)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Regressor())
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95
