import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.first_layer_dim = round((num_input_features - hidden_dim) / 4)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.first_layer_dim),
            torch.nn.Dropout(),
            torch.nn.Tanh(),
            torch.nn.Linear(self.first_layer_dim, hidden_dim),
            torch.nn.Dropout(),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs

        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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
        self.X_train = torch.FloatTensor(X_train)
        self.X_test = torch.FloatTensor(X_test)
        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        for qid in set(inp_query_ids):
            mask = (inp_query_ids == qid)
            inp_feat_array[mask] = StandardScaler().fit_transform(inp_feat_array[mask])

        return inp_feat_array

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        ndcgs = []
        for i in range(self.n_epochs):
            self._train_one_epoch()
            ndcgs.append(self._eval_test_set())
        return ndcgs

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        # loss = torch.nn.functional.cross_entropy(batch_pred, batch_ys)
        pred_soft = torch.nn.functional.softmax(batch_pred)
        true_soft = torch.nn.functional.softmax(batch_ys)
        loss = -(true_soft * torch.log(pred_soft)).sum()
        return loss

    def _train_one_epoch(self) -> None:
        self.model.train()
        for qid in set(self.query_ids_train):
            mask = (self.query_ids_train == qid)
            batch_pred = self.model(self.X_train[mask])
            loss = self._calc_loss(self.ys_train[mask], batch_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            for qid in set(self.query_ids_test):
                mask = (self.query_ids_test == qid)
                batch_pred = self.model(self.X_test[mask])
                ndcg = self._ndcg_k(self.ys_test[mask], batch_pred, self.ndcg_top_k)
                ndcgs.append(ndcg)
            return np.mean(ndcgs)

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
