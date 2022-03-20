from hyperopt import hp
import numpy as np

PATH_TO_MODEL = './model.bin'

PARAM_GRID = {
    'ndcg_top_k': 10,
    'max_depth': 1 + hp.randint('max_depth', 8),
    'n_estimators': 10 + hp.randint('n_estimators', 200),
    'lr': hp.loguniform('lr', np.log(0.01), np.log(0.9)),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.2, 0.9),
    'subsample': hp.uniform('subsample', 0.1, 0.7),
    'min_samples_leaf': 5 + hp.randint('min_samples_leaf', 50),
}
PARAMS = {
    'lr': 0.3767074163111257,
    'max_depth': 15.569075684303327,
    'min_samples_leaf': 93,
    'subsample': 0.9158298390219355,
    'colsample_bytree': 0.9172341375456129,
    'n_estimators': 100,
    'ndcg_top_k': 10,
}
# best loss: 0.4217832161179494
