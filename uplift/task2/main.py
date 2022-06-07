import numpy as np

from config import DevConfig
from tree import UpliftTreeRegressor
from utils import read_np_array


def read_data(cfg):
    features = read_np_array(cfg.features)
    target = read_np_array(cfg.target)
    treatment = read_np_array(cfg.treatment)
    forecast = read_np_array(cfg.forecast)
    print(f'features shape: {features.shape}')
    print(f'treatment unique cnt: {len(np.unique(treatment))}')
    return features, target, treatment, forecast


def _check(model_constructor, model_params, X, treatment, y, X_test, pred_right, eps=1e-5) -> bool:
    model = model_constructor(**model_params)
    print('Start fitting')
    model.fit(X, treatment, y)
    print('Start inference')
    pred = np.array(model.predict(X_test)).reshape(len(X_test))
    print(f'predict mean: {pred.mean()}')
    print(f'predict right: {pred_right.mean()}')
    score = np.max(np.abs(pred - pred_right))
    print(score)
    passed = (score < eps)

    return passed


if __name__ == '__main__':
    cfg = DevConfig()
    features, target, treatment, forecast = read_data(cfg)
    assert _check(UpliftTreeRegressor, cfg.model_params, features, treatment, target, features, forecast)
