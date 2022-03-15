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
        self.best_ndcg = 0

        self.n_samples = round(self.subsample * self.X_train.shape[0])
        self.n_features = round(self.colsample_bytree * self.X_train.shape[1])

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()
        # train_df = self._filter_zero_relevance(train_df)

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
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train).unsqueeze(1)
        self.ys_test = torch.FloatTensor(y_test).unsqueeze(1)

    def _filter_zero_relevance(self, data):
        return data.loc[data[0] > 0]

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

        # calculate lambda separately for each query
        lambdas = torch.zeros(list(self.ys_train.shape)[0]).unsqueeze(1)
        for qid in set(self.query_ids_train):
            mask = (self.query_ids_train == qid)
            lambdas[mask] = self._compute_lambdas(self.ys_train[mask], train_preds[mask])
        lambdas = lambdas[samples]
        X_train = self.X_train[samples][:, features]
        tree = DecisionTreeRegressor(random_state=cur_tree_idx,
                                     max_depth=self.max_depth,
                                     min_samples_leaf=self.min_samples_leaf,
                                     )
        tree.fit(X_train, -lambdas)

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

        y_train_pred, y_test_pred = 0 * self.ys_train, 0 * self.ys_test
        prune_ind = -1
        for i in tqdm(range(self.n_estimators)):
            tree, features = self._train_one_tree(i, y_train_pred)
            y_train_pred += self.lr * torch.FloatTensor(tree.predict(self.X_train[:, features])).unsqueeze(1)
            y_test_pred += self.lr * torch.FloatTensor(tree.predict(self.X_test[:, features])).unsqueeze(1)

            ndcg = self._calc_data_ndcg(self.query_ids_test, self.ys_test, y_test_pred)
            if ndcg > self.best_ndcg:
                self.best_ndcg, prune_ind = ndcg, i+1

            self.trees.append(tree)
            self.tree_feat.append(features)

            # print(f"\nndcg: {round(ndcg, 4)}\tbest ndcg: {round(self.best_ndcg, 4)}")

        self.trees = self.trees[:prune_ind]
        self.tree_feat = self.tree_feat[:prune_ind]

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        y_pred = torch.FloatTensor(torch.zeros(list(data.shape)[0])).unsqueeze(1)
        for i in tqdm(range(len(self.trees))):
            y_pred += self.lr * torch.FloatTensor(self.trees[i].predict(data[:, self.tree_feat[i]])).unsqueeze(1)

        return y_pred

    def _compute_lambdas(self, y_true: torch.FloatTensor, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        # calculate normalization coefficient
        ideal_dcg = self._dcg_k(y_true, y_true, -1)
        if ideal_dcg == 0:
            N = 0
        else:
            N = 1 / ideal_dcg
        # calculate document relevance order
        _, rank_order = torch.sort(y_true, descending=True, dim=0)
        rank_order += 1

        with torch.no_grad():
            # calc pairwise scores differences in batch
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            # assign 1 if first document more relevant, otherwise -1
            Sij = self._compute_labels_in_batch(y_true)
            # calc gain boost in case of permutation
            gain_diff = self._compute_gain_diff(y_true)

            # calc denominator
            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            # calc NDCG score
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            # calc lambdas
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

            return lambda_update

    def _compute_labels_in_batch(self, y_true):
        # pairwise difference of relevance scores
        rel_diff = y_true - y_true.t()

        # first document in pair is more relevant
        pos_pairs = (rel_diff > 0).type(torch.float32)

        # first document in pair is less relevant
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs

        return Sij

    def _compute_gain_diff(self, y_true, gain_scheme='exp2'):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff

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
        state = {'tree_feat': self.tree_feat, 'trees': self.trees, 'lr': self.lr}
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)

        self.tree_feat = state['tree_feat']
        self.trees = state['trees']
        self.lr = state['lr']
