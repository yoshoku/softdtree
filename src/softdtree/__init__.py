"""
Soft Decision Tree for Classification and Regression.

This module provides classifier and regressor based on soft decision tree.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MultiOutputMixin, RegressorMixin
from sklearn.preprocessing import label_binarize
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from ._softdtree import BaseSoftDecisionTreeClassifier, BaseSoftDecisionTreeRegressor

__verison__: str = "0.1.0"

class SoftDecisionTreeClassifier(BaseEstimator, MultiOutputMixin, ClassifierMixin):

    """
    Soft Decision Tree Classifier.

    This class implements a classifier based on a soft decision tree.
    """

    max_depth: int
    max_features: float
    max_epoch: int
    batch_size: int
    eta: float
    beta1: float
    beta2: float
    epsilon: float
    tol: float
    verbose: int
    random_seed: int
    classes_: np.ndarray
    tree_: BaseSoftDecisionTreeClassifier

    def __init__(self, max_depth: int = 8, max_features: float = 1.0,
                 max_epoch: int = 100, batch_size: int = 5,
                 eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 tol: float = 1e-4, verbose: int = 0, random_seed: int = -1) -> None:
        """Initialize the SoftDecisionTreeClassifier."""
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tol = tol
        self.verbose = verbose
        self.random_seed = random_seed
        self.tree_ = BaseSoftDecisionTreeClassifier(
            max_depth, max_features, max_epoch, batch_size,
            eta,beta1, beta2, epsilon, tol, verbose, random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray, **params: dict) -> "SoftDecisionTreeClassifier":
        """Fit the SoftDecisionTreeClassifier to the training data."""
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        if len(self.classes_) > 2:
            y = self._one_hot_encode(y)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1).astype(np.float64)
        self.tree_.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the input data X."""
        check_is_fitted(self)
        X = check_array(X)
        df_values = self.tree_.decision_function(X)
        n_samples = X.shape[0]
        if df_values.shape[1] > 1:
            result = np.array(
                    [self.classes_[df_values[i].argmax()] for i in range(n_samples)],
                    dtype=self.classes_.dtype)
        else:
            result = np.array(
                    [(self.classes_[1] if df_values[i][0] > 0.5 else self.classes_[0]) for i in range(n_samples)],
                    dtype=self.classes_.dtype)
        return result

    def decision_function(self, X: np.ndarray) -> np.typing.NDArray[np.float64]:
        """Compute the decision function for the input data X."""
        X = X[:, self.random_ids] if hasattr(self, "random_ids") else X
        return self.tree_.decision_function(X)

    def _one_hot_encode(self, y: np.ndarray) -> np.typing.NDArray[np.float64]:
        return label_binarize(y, classes=self.classes_).astype(np.float64)


class SoftDecisionTreeRegressor(BaseEstimator, MultiOutputMixin, RegressorMixin):

    """
    Soft Decision Tree Regressor.

    This class implements a regressor based on a soft decision tree.
    """

    max_depth: int
    max_features: float
    max_epoch: int
    batch_size: int
    eta: float
    beta1: float
    beta2: float
    epsilon: float
    tol: float
    verbose: int
    random_seed: int
    tree_: BaseSoftDecisionTreeRegressor

    def __init__(self, max_depth: int = 8, max_features: float = 1.0,
                 max_epoch: int = 100, batch_size: int = 5,
                 eta: float = 0.1, beta1: float = 0.9, beta2: float = 0.999,
                 epsilon: float = 1e-8,
                 tol: float = 1e-4, verbose: int = 0, random_seed: int = -1) -> None:
        """Initialize the SoftDecisionTreeRegressor."""
        self.max_depth = max_depth
        self.max_features = max_features
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.tol = tol
        self.verbose = verbose
        self.random_seed = random_seed
        self.tree_ = BaseSoftDecisionTreeRegressor(
            max_depth, max_features, max_epoch, batch_size,
            eta, beta1, beta2, epsilon, tol, verbose, random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray, **params: dict) -> "SoftDecisionTreeRegressor":
        """Fit the SoftDecisionTreeRegressor to the training data."""
        X, y = check_X_y(X, y, multi_output=True)
        if y.ndim == 1:
            y = np.expand_dims(y, axis=1)
        self.tree_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.typing.NDArray[np.float64]:
        """Predict target values for the input data X."""
        X = check_array(X)
        X = X[:, self.random_ids] if hasattr(self, "random_ids") else X
        df_values = self.tree_.decision_function(X)
        if df_values.ndim > 1 and df_values.shape[1] == 1:
            return df_values[:,0]
        return df_values


__all__ = [
    "SoftDecisionTreeClassifier",
    "SoftDecisionTreeRegressor",
    "__version__",
]
