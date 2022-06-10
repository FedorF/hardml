
import dask.dataframe as dd
import pandas as pd
import datetime
import sklearn.base as skbase
import sklearn.preprocessing as skpreprocessing
import featurelib as fl

from typing import List, Dict, Union


def dask_groupby(
    data: dd.DataFrame,
    by: List[str],
    config: Dict[str, Union[str, List[str]]]
) -> dd.DataFrame:
    data_ = data.copy()
    dask_agg_config = dict()

    for col, aggs in config.items():
        aggs = aggs if isinstance(aggs, list) else [aggs]
        for agg in aggs:
            fictious_col = f'{col}_{agg}'
            data_ = data_.assign(**{fictious_col: lambda d: d[col]})
            dask_agg_config[fictious_col] = agg

    result = data_.groupby(by=by).agg(dask_agg_config)
    return result


class UniqueCategoriesCalcer(fl.DateFeatureCalcer):
    name = 'unique_categories'
    keys = ['client_id']

    def __init__(self, delta: int, col_category: str = 'level_3', **kwargs):
        self.delta = delta
        self.col_category = col_category
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        purchases = self.engine.get_table('purchases')
        products = self.engine.get_table('products')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (purchases['transaction_datetime'] >= date_from) & (purchases['transaction_datetime'] < date_to)

        purchases = (
            purchases
            .loc[date_mask]
            .merge(
                products[['product_id', self.col_category]],
                on=['product_id'],
                how='inner'
            )
        )
        result = purchases.groupby(by=['client_id'])[self.col_category].nunique().reset_index()
        
        result = result.rename(columns={self.col_category: f'unique_{self.col_category}__{self.delta}d'})
        return result


class AgeGenderCalcer(fl.FeatureCalcer):
    name = 'age_gender'
    keys = ['client_id']

    def compute(self) -> dd.DataFrame:
        client_profile = self.engine.get_table('client_profile')
        return client_profile[self.keys + ['age', 'gender']]


class ReceiptsBasicFeatureCalcer(fl.DateFeatureCalcer):
    name = 'receipts_basic'
    keys = ['client_id']

    def __init__(self, delta: int, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        date_mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        features = (
            receipts
            .loc[date_mask]
            .assign(points_spent=lambda d: d['regular_points_spent'] + d['express_points_spent'])
            .assign(points_spent_flag=lambda d: (d['points_spent'] < 0).astype(int))
            .assign(express_points_spent_flag=lambda d: (d['express_points_spent'] < 0).astype(int))
        )
        features = dask_groupby(
            features,
            by=['client_id'],
            config={
                "transaction_id": "count",
                "purchase_sum": ["sum", "max", "min", "mean"],
                "regular_points_spent": ["sum", "max"],
                "express_points_spent": ["sum", "max"],
                "transaction_datetime": ["min", "max"],
                "trn_sum_from_red": ["sum", "max", "mean"],
                "points_spent_flag": ["sum"],
                "express_points_spent_flag": ["sum"],
            }
        )
        features = (
            features
            .assign(
                mean_time_interval=lambda d: (
                    (d['transaction_datetime_max'] - d['transaction_datetime_min'])
                    / (d['transaction_id_count'] - 1)
                ).apply(lambda delta: delta.total_seconds() / (24 * 3600))
            )
            .assign(
                time_since_last=lambda d: (
                    date_to - d['transaction_datetime_max']
                ).apply(lambda delta: delta.total_seconds() / (24 * 3600))
            )
        )

        features = features.reset_index()
        features = features.rename(columns={
            col: col + f'__{self.delta}d' for col in features.columns if col not in self.keys
        })

        return features


class TargetFromCampaignsCalcer(fl.DateFeatureCalcer):
    name = 'target_from_campaigns'
    keys = ['client_id']
    
    def compute(self) -> dd.DataFrame:
        campaigns = self.engine.get_table('campaigns')
        date_mask = (dd.to_datetime(campaigns['treatment_date'], format='%Y-%m-%d').dt.date == self.date_to)

        result = (
            self.engine.get_table('campaigns')
            .loc[date_mask]
            [[
                'client_id', 'treatment_flg',
                'target_purchases_sum', 'target_purchases_count', 'target_campaign_points_spent'
            ]]
        )
        return result


class OneHotEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, cols: List[str], prefix: str = 'ohe', **ohe_params):
        self.cols = cols
        self.prefix = prefix
        self.encoder_ = skpreprocessing.OneHotEncoder(**(ohe_params or {}))

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        self.encoder_.fit(data[self.cols])
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        result_column_names = []
        for col_idx, col in enumerate(self.cols):
            result_column_names += [
                f'{self.prefix}__{col}__{value}'
                for i, value in enumerate(self.encoder_.categories_[col_idx])
                if self.encoder_.drop_idx_ is None or i != self.encoder_.drop_idx_[col_idx]
            ]

        encoded = pd.DataFrame(
            self.encoder_.transform(data[self.cols]).todense(),
            columns=result_column_names
        )

        for col in encoded.columns:
            data[col] = encoded[col]
        return data


@fl.functional_transformer
def divide_cols(data: pd.DataFrame, col_numerator: str, col_denominator: str, col_result: str = None):
    col_result = col_result or f'ratio__{col_numerator}__{col_denominator}'
    data[col_result] = data[col_numerator] / data[col_denominator]
    return data