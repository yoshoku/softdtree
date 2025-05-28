from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, scale

from softdtree import SoftDecisionTreeRegressor


def test_soft_decision_tree_regressor() -> None:
    X, y = make_regression(n_samples=100, n_targets=1, n_features=2, bias=20.0, noise=2.0, random_state=1984)
    y = scale(y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SoftDecisionTreeRegressor(max_depth=8, batch_size=10, random_seed=42)),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95

def test_soft_decision_tree_regressor_multi() -> None:
    X, y = make_regression(n_samples=100, n_targets=3, n_features=2, bias=20.0, noise=2.0, random_state=1984)
    y = scale(y)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SoftDecisionTreeRegressor(max_depth=8, batch_size=10, random_seed=42)),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95
