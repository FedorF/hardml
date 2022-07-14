import pandas as pd
import numpy as np
from scipy import stats

from abt.task6 import ratio, config as cfg


def get_user_receipts(n_days, n_visits):
    has_visited = stats.bernoulli.rvs(p=n_visits/n_days, size=n_days)
    receipt = 1000 + 500*np.random.randn(n_days)
    return receipt * has_visited


def generate_data(value_name, user_id_name, date_name, user_num, n_days):
    df = []
    for i in range(user_num):
        df_user = pd.DataFrame(data={
            date_name: [f'2022-07-{10+day}' for day in range(n_days)],
            user_id_name: (i+np.zeros(n_days)),
            value_name: get_user_receipts(n_days, np.random.randint(0, n_days, 1)[0]),
        })
        df.append(df_user)
    df = pd.concat(df)
    list_user_id = df.sample(frac=.5)[user_id_name].unique().tolist()
    periods = {'begin': '2022-07-10', 'end': f'2022-07-{10+n_days-1}'}

    return df, list_user_id, periods


if __name__ == '__main__':
    df, list_user_id, period = generate_data(cfg.VALUE_COL, cfg.USER_COL, cfg.TS_COL, cfg.USER_NUM, cfg.N_DAYS)
    result = ratio.calculate_linearized_metric(
        df, cfg.VALUE_COL, cfg.USER_COL, list_user_id, cfg.TS_COL, period, cfg.METRIC_COL
    )
    print(df)
    print('\n')
    print(result.head())
