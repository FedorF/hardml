from hyperopt import hp
import numpy as np

PATH_TO_MODEL = './model.bin'

PARAM_GRID = {
    'max_depth': hp.choice('max_depth', np.arange(2, 16, 1)),
    'min_samples_leaf': hp.choice('min_samples_leaf', [1, 3, 5, 7, 10, 15, 20, 30, 50]),
    'subsample': hp.choice('subsample', np.arange(0.6, 1.0, 0.05)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.6, 1.0, 0.05)),
    'lr': hp.choice('lr', [0.01, 0.5]),
}

PARAMS = {
    'lr': 0.9,
    'max_depth': 5,
    'min_samples_leaf': 100,
    'subsample': 0.99,
    'colsample_bytree': 0.99,
    'n_estimators': 100,
    'ndcg_top_k': 10,
}
