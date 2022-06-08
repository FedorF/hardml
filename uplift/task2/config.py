from pathlib import Path
from typing import Dict

from pydantic import BaseSettings


class DevConfig(BaseSettings):
    features: Path = Path('./data/example_X.npy')
    target: Path = Path('./data/example_y.npy')
    treatment: Path = Path('./data/example_treatment.npy')
    forecast: Path = Path('./data/example_preds.npy')
    model_params: Dict = {'max_depth': 3,
                          'min_samples_leaf': 6000,
                          'min_samples_leaf_treated': 2500,
                          'min_samples_leaf_control': 2500,
                          }
