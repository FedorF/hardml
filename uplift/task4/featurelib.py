import sklearn.base as skbase
import sklearn.pipeline as skpipe
import functools
import dask.dataframe as dd
import datetime

from abc import ABC, abstractmethod
from typing import List, Dict


class Engine:
    def __init__(self, tables: Dict[str, dd.DataFrame]):
        self.tables = tables

    def register_table(self, table: dd.DataFrame, name: str) -> None:
        self.tables[name] = table

    def get_table(self, name: str) -> dd.DataFrame:
        return self.tables[name]


class FeatureCalcer(ABC):
    name = '_base'
    keys = None

    def __init__(self, engine: Engine):
        self.engine = engine

    @abstractmethod
    def compute(self):
        pass


class DateFeatureCalcer(FeatureCalcer):
    def __init__(self, date_to: datetime.date, **kwargs):
        self.date_to = date_to
        super().__init__(**kwargs)


CALCER_REFERENCE = {}


def register_calcer(calcer_class) -> None:
    CALCER_REFERENCE[calcer_class.name] = calcer_class


def create_calcer(name: str, **kwargs) -> FeatureCalcer:
    return CALCER_REFERENCE[name](**kwargs)


def join_tables(tables: List[dd.DataFrame], on: List[str], how: str) -> dd.DataFrame:
    result = tables[0]
    for table in tables[1: ]:
        result = result.merge(table, on=on, how=how)
    return result


def compute_features(engine: Engine, features_config: dict) -> dd.DataFrame:
    calcers = list()
    keys = None

    for feature_config in features_config:
        calcer_args = feature_config["args"]
        calcer_args["engine"] = engine

        calcer = create_calcer(feature_config["name"], **calcer_args)
        if keys is None:
            keys = set(calcer.keys)
        elif set(calcer.keys) != keys:
            raise KeyError(f"{calcer.keys}")

        calcers.append(calcer)

    computation_results = []
    for calcer in calcers:
        computation_results.append(calcer.compute())
    result = join_tables(computation_results, on=list(keys), how='outer')

    return result


class FunctionalTransformer(skbase.BaseEstimator):
    def __init__(self, function, **params):
        self.function = functools.partial(function, **params)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def functional_transformer(function):
    def builder(**params):
        return FunctionalTransformer(function, **params)
    return builder


TRANSFORMER_REFERENCE = {}


def register_transformer(transformer_class, name: str) -> None:
    TRANSFORMER_REFERENCE[name] = transformer_class


def create_transformer(name: str, **kwargs) -> skbase.BaseEstimator:
    return TRANSFORMER_REFERENCE[name](**kwargs)


def build_pipeline(transform_config: dict) -> skpipe.Pipeline:
    transformers = list()

    for i, transformer_config in enumerate(transform_config):
        transformer_args = transformer_config["args"]

        transformer = create_transformer(transformer_config["name"], **transformer_args)
        uname = transformer_config.get("uname", f'stage_{i}')

        transformers.append((uname, transformer))

    pipeline = skpipe.Pipeline(transformers)
    return pipeline
