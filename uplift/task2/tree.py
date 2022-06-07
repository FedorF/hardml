from typing import Iterable, Tuple

import numpy as np


class Node:
    def __init__(self, tau):
        self.tau = tau
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None


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

    def _calc_tau(self, y, t) -> float:
        return y[t == 1].mean() - y[t == 0].mean()

    def _split_is_valid(self, t):
        check_ = (
                (len(t) >= self.min_samples_leaf)
                & (sum(t == 1) >= self.min_samples_leaf_treated)
                & (sum(t == 0) >= self.min_samples_leaf_control)
        )
        return check_

    def _best_split(self, x, y, t) -> Tuple[Node, Node, int, float, np.array]:
        delta_p_opt = -1
        best_split = (None, None, -1, None, None)
        for feat_ind in range(x.shape[1]):
            for threshold in self._get_thresholds(x[:, feat_ind]):
                mask_left = (x[:, feat_ind] <= threshold)
                if not self._split_is_valid(t[mask_left]) or not self._split_is_valid(t[~mask_left]):
                    continue

                tau_l = self._calc_tau(y[mask_left], t[mask_left])
                tau_r = self._calc_tau(y[~mask_left], t[~mask_left])
                delta_p = abs(tau_l - tau_r)
                if delta_p > delta_p_opt:
                    delta_p_opt = delta_p
                    best_split = (Node(tau_l), Node(tau_r), feat_ind, threshold, mask_left)

        return best_split

    def _grow_tree(self, x, y, t, depth: int = 0) -> Node:
        node = Node(self._calc_tau(y, t))

        if depth < self.max_depth:
            node_l, node_r, feat_ind, threshold, mask_left = self._best_split(x, y, t)
            if feat_ind >= 0:
                node.feature_index = feat_ind
                node.threshold = threshold
                node.left = self._grow_tree(x[mask_left], y[mask_left], t[mask_left], depth+1)
                node.right = self._grow_tree(x[~mask_left], y[~mask_left], t[~mask_left], depth+1)

        return node

    def _predict(self, inputs: np.array) -> float:
        """Predict class for a single sample."""
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.tau

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray) -> None:
        """
        Parameters:
            X - массив (n * k) с признаками.
            treatment - массив (n) с флагом воздействия.
            y - массив (n) с целевой переменной.

        """
        self.tree_ = self._grow_tree(X, y, treatment)

    def predict(self, X: np.ndarray) -> Iterable[float]:
        if not self.tree_:
            raise ValueError(f'Error. {self.__class__.__name__} is not fitted. Please use .fit() first.')

        return [self._predict(inputs) for inputs in X]
