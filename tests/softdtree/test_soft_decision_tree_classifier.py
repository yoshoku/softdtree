from sklearn.datasets import load_digits, make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from softdtree import SoftDecisionTreeClassifier


def test_soft_decision_tree_classifier() -> None:
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SoftDecisionTreeClassifier(max_depth=8, eta=0.1, max_epoch=50, random_seed=42)),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95

def test_soft_decision_tree_multi_classifier() -> None:
    X, y = load_digits(n_class=3, return_X_y=True)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SoftDecisionTreeClassifier(max_depth=8, eta=0.1, max_epoch=50, random_seed=42)),
    ])
    model.fit(X, y)
    assert model.score(X, y) >= 0.95
