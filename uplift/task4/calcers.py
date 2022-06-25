import datetime

import dask.dataframe as dd

import featurelib as fl


class DayOfWeekReceiptsCalcer(fl.DateFeatureCalcer):
    name = 'day_of_week_receipts'
    keys = ['client_id']

    def __init__(self, delta, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')
        receipts['day_of_week'] = receipts.transaction_datetime.dt.weekday.astype('category').cat.as_known()

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)

        result = (
            receipts
            .loc[mask, ['client_id', 'transaction_id', 'day_of_week']]
            .drop_duplicates()
            .pivot_table(index='client_id', columns='day_of_week', values='transaction_id', aggfunc='count')
        )
        result.columns = [f'purchases_count_dw{col}__{self.delta}d' for col in result.columns]

        return result.reset_index()


class FavouriteStoreCalcer(fl.DateFeatureCalcer):
    name = 'favourite_store'
    keys = ['client_id']

    def __init__(self, delta, **kwargs):
        self.delta = delta
        super().__init__(**kwargs)

    def compute(self) -> dd.DataFrame:
        receipts = self.engine.get_table('receipts')

        date_to = datetime.datetime.combine(self.date_to, datetime.datetime.min.time())
        date_from = date_to - datetime.timedelta(days=self.delta)
        mask = (receipts['transaction_datetime'] >= date_from) & (receipts['transaction_datetime'] < date_to)
        receipts = receipts[mask]

        trans_cnt = receipts.groupby(['client_id', 'store_id'])['transaction_id'].count().reset_index()
        trans_cnt_max = trans_cnt.groupby('client_id')['transaction_id'].max().reset_index()
        trans_filtered = trans_cnt.merge(trans_cnt_max)
        client_best_store = trans_filtered.groupby('client_id')['store_id'].max().reset_index()

        return client_best_store.rename(columns={'store_id': f'favourite_store_id__{self.delta}d'})
