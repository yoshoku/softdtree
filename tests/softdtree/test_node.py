import numpy as np
import pytest

from softdtree import Node


def test_node() -> None:
    node = Node()
    assert node is not None
    assert node.id == -1
    assert node.parent_id == -1
    assert node.left_child_id == -1
    assert node.right_child_id == -1
    assert node.depth == 0
    assert node.is_leaf is True
    assert node.is_left is True
    assert node.has_parent is False
    node.id = 1
    node.parent_id = 0
    node.left_child_id = 2
    node.right_child_id = 3
    node.is_leaf = False
    assert node.id == 1
    assert node.parent_id == 0
    assert node.left_child_id == 2
    assert node.right_child_id == 3
    assert node.is_leaf is False
    assert node.has_parent is True
    node.weight = np.array([0.1, 0.2, 0.3])
    node.bias = 0.5
    node.response = np.array([0.4, 0.5])
    assert node.weight.shape == (3,)
    assert node.weight[0] == pytest.approx(0.1)
    assert node.weight[1] == pytest.approx(0.2)
    assert node.weight[2] == pytest.approx(0.3)
    assert node.bias == pytest.approx(0.5)
    assert node.response.shape == (2,)
    assert node.response[0] == pytest.approx(0.4)
    assert node.response[1] == pytest.approx(0.5)
