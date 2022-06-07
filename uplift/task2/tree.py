from typing import Iterable, Tuple

import numpy as np


class Node:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t
        self.num_samples = len(y)
        self.num_samples_control = sum(t == 0)
        self.num_samples_treated = sum(t == 1)
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

    def calc_tau(self) -> float:
        return np.mean(self.y[self.t == 1]) - np.mean(self.y[self.t == 0])


class UpliftTreeRegressor:
    """
    Parameters:
        max_depth - максимальная глубина дерева.
        min_samples_leaf - минимальное необходимое число обучающих объектов в листе дерева.
        min_samples_leaf_treated - минимальное необходимое число обучающих объектов с T=1 в листе дерева.
        min_samples_leaf_control - минимальное необходимое число обучающих объектов с T=0 в листе дерева.

    """

    def __init__(self,
                 max_depth: int = 3,
                 min_samples_leaf: int = 1000,
                 min_samples_leaf_treated: int = 300,
                 min_samples_leaf_control: int = 300):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_leaf_treated = min_samples_leaf_treated
        self.min_samples_leaf_control = min_samples_leaf_control
        self.tree_ = None

    def _get_thresholds(self, data) -> np.array:
        unique_values = np.unique(data)
        if len(unique_values) > 10:
            percentiles = np.percentile(data, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
        else:
            percentiles = np.percentile(unique_values, [10, 50, 90])

        return np.unique(percentiles)

    def _calc_delta_p(self, left: Node, right: Node) -> float:
        return abs(left.calc_tau() - right.calc_tau())

    def _node_is_valid(self, node):
        check_ = (
                (node.num_samples >= self.min_samples_leaf)
                & (node.num_samples_treated >= self.min_samples_leaf_treated)
                & (node.num_samples_control >= self.min_samples_leaf_control)
        )
        return check_

    def _best_split(self, node: Node) -> Tuple[Node, Node, int, float]:
        delta_p_opt = -1
        best_split = (None, None, -1, -1.0)
        for feat_ind in range(node.x.shape[1]):
            for threshold in self._get_thresholds(node.x[:, feat_ind]):
                left_ind = (node.x[:, feat_ind] <= threshold)
                node_l = Node(node.x[left_ind], node.y[left_ind], node.t[left_ind])
                node_r = Node(node.x[~left_ind], node.y[~left_ind], node.t[~left_ind])
                if not self._node_is_valid(node_l) or not self._node_is_valid(node_r):
                    continue
                delta_p = self._calc_delta_p(node_l, node_r)
                if delta_p > delta_p_opt:
                    delta_p_opt = delta_p
                    best_split = (node_l, node_r, feat_ind, threshold)

        return best_split

    def _grow_tree(self, node: Node, depth: int = 0) -> Node:
        if depth < self.max_depth:
            node_l, node_r, feat_ind, threshold = self._best_split(node)
            if feat_ind >= 0:
                node.feature_index = feat_ind
                node.threshold = threshold
                node.left = self._grow_tree(node_l, depth+1)
                node.right = self._grow_tree(node_r, depth+1)

        return node

    def _predict(self, inputs: np.array) -> float:
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.calc_tau()

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """
        Parameters:
            X - массив (n * k) с признаками.
            treatment - массив (n) с флагом воздействия.
            y - массив (n) с целевой переменной.

        """

        base_node = Node(X, y, treatment)
        self.tree_ = self._grow_tree(base_node)

    def predict(self, X: np.ndarray) -> Iterable[float]:
        if not self.tree_:
            raise ValueError(f'Error. {self.__class__.__name__} is not fitted. Please use .fit() first.')

        return [self._predict(inputs) for inputs in X]
