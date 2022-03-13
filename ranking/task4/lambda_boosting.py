import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self._prepare_data()

        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.tree_feat = []

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()
        X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)
        self.n_samples = round(self.subsample * X_train.shape[0])
        self.n_features = round(self.colsample_bytree * X_train.shape[1])
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train).unsqueeze(1)
        self.ys_test = torch.FloatTensor(y_test).unsqueeze(1)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for qid in set(inp_query_ids):
            mask = (inp_query_ids == qid)
            inp_feat_array[mask] = StandardScaler().fit_transform(inp_feat_array[mask])

        return inp_feat_array

    def _train_one_tree(self, cur_tree_idx: int,
                        train_preds: torch.FloatTensor
                        ) -> Tuple[DecisionTreeRegressor, np.ndarray]:
        random.seed(cur_tree_idx)
        samples = random.sample(np.arange(self.X_train.shape[0]).tolist(), self.n_samples)
        features = random.sample(np.arange(self.X_train.shape[1]).tolist(), self.n_features)
        X_train = self.X_train[samples, features]
        y_train = self._compute_lambdas(self.ys_train, train_preds)[samples]
        tree = DecisionTreeRegressor(random_state=cur_tree_idx,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     )
        tree.fit(X_train, y_train)

        return tree, features

    def _calc_data_ndcg(self, queries_list: np.ndarray,
                        true_labels: torch.FloatTensor, preds: torch.FloatTensor) -> float:
        ndcgs = []
        for qid in set(queries_list):
            mask = (queries_list == qid)
            batch_pred = preds[mask].flatten()
            batch_true = true_labels[mask].flatten()
            ndcg = self._ndcg_k(batch_true, batch_pred, self.ndcg_top_k)
            ndcgs.append(ndcg)

        return float(np.mean(ndcgs))

    def fit(self):
        np.random.seed(0)
        best_ndcg, cut_ind = 0, 0
        y_train_pred, y_test_pred = 0 * self.ys_train, 0 * self.ys_test
        for i in tqdm(range(self.n_estimators)):
            tree, features = self._train_one_tree(i, y_train_pred)
            y_train_pred += self.lr * tree.predict(self.X_train[:, features])
            y_test_pred += self.lr * tree.predict(self.X_test[:, features])
            ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, y_test_pred)
            if ndcg > best_ndcg:
                best_ndcg, cut_ind = ndcg, i

            self.trees.append(tree)
            self.tree_feat.append(features)

        self.trees = self.trees[:cut_ind]
        self.tree_feat = self.tree_feat[:cut_ind]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        y_pred = torch.FloatTensor(torch.zeros(list(data.shape)[0]))
        for i in tqdm(range(len(self.trees))):
            y_pred += self.lr * self.trees[i].predict(data[:, self.tree_feat[i]])

        return y_pred

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # todo
        lambdas = 0

        return lambdas

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        ideal_dcg = self._dcg_k(ys_true, ys_true, ndcg_top_k)
        dcg = self._dcg_k(ys_true, ys_pred, ndcg_top_k)
        if ideal_dcg == 0:
            return 0
        else:
            return dcg / ideal_dcg

    def _dcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
               top_k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        ind = min((len(ys_true), top_k))
        ys_true_sorted = ys_true[indices][:ind]
        gain = 0
        for i, y in enumerate(ys_true_sorted, start=1):
            gain += (2 ** y.item() - 1) / math.log2(i + 1)

        return gain

    def save_model(self, path: str):
        state = {self.tree_feat, self.trees}
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            self.tree_feat, self.trees = pickle.load(f)
