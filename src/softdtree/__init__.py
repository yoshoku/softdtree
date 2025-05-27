"""
Soft Decision Tree for Classification and Regression.

This module provides classifier and regressor based on soft decision tree.
"""

from ._softdtree import BaseSoftDecisionTreeClassifier, BaseSoftDecisionTreeRegressor, Node

__all__ = [
    "BaseSoftDecisionTreeClassifier",
    "BaseSoftDecisionTreeRegressor",
    "Node",
]
