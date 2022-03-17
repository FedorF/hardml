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
    'lr': .96,
    'max_depth': 7,
    'min_samples_leaf': 195,
    'subsample': 0.99,
    'colsample_bytree': 0.99,
    'n_estimators': 100,
    'ndcg_top_k': 10,
}
# 0.40788

# PARAMS = {
#     'lr': .95,
#     'max_depth': 7,
#     'min_samples_leaf': 190,
#     'subsample': 0.99,
#     'colsample_bytree': 0.99,
#     'n_estimators': 100,
#     'ndcg_top_k': 10,
# }
# иest ndcg: 0.40579

# 0.405