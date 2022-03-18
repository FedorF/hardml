from hyperopt import hp

PATH_TO_MODEL = './model.bin'

PARAM_GRID = {
    'max_depth': hp.uniform('max_depth', 3, 21),
    'min_samples_leaf': hp.randint('min_samples_leaf', 262) + 2,
    'subsample': hp.uniform('subsample', .9, .99),
    'colsample_bytree': hp.uniform('colsample_bytree', .85, .99),
    'lr': hp.uniform('lr', .05, .999),
    'n_estimators': 100,
    'ndcg_top_k': 10,
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
