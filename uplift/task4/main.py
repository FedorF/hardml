import datetime

import dask.dataframe as dd

import featurelib as ff
import calcers
import transformers


def init_engine():
    receipts = dd.read_parquet('data/receipts.parquet')
    campaigns = dd.read_csv('data/campaigns.csv')
    products = dd.read_csv('data/products.csv')
    client_profile = dd.read_csv('data/client_profile.csv')
    purchases = dd.read_parquet('data/purchases.parquet/')

    tables = {
        'receipts': receipts,
        'campaigns': campaigns,
        'client_profile': client_profile,
        'products': products,
        'purchases': purchases,
    }

    return ff.Engine(tables=tables)


if __name__ == '__main__':
    engine = init_engine()
