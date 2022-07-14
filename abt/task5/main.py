import pandas as pd
import numpy as np

from abt.task5 import cuped as cpd, config as cfg


def generate_data(size, user_num, value_name, user_id_name, date_name):
    df = pd.DataFrame(data={
        date_name: [f'2022-07-{day}' for day in np.random.randint(low=10, high=31, size=size)],
        user_id_name: np.random.randint(user_num, size=size),
        value_name: 5000 + 2000*np.random.randn(size),
    })
    list_user_id = df.sample(frac=.2)[user_id_name].unique().tolist()
    periods = {
            'prepilot': {'begin': '2022-07-10', 'end': '2022-07-21'},
            'pilot': {'begin': '2022-07-21', 'end': '2022-07-31'}
        }

    return df, list_user_id, periods


if __name__ == '__main__':
    df, list_user_id, periods = generate_data(cfg.DATA_SIZE, cfg.USER_NUM, cfg.VALUE_COL, cfg.USER_COL, cfg.TS_COL)
    df_cuped = (
        cpd.calculate_metric_cuped(df, cfg.VALUE_COL, cfg.USER_COL, list_user_id, cfg.TS_COL, periods, cfg.METRIC_COL)
    )
    print(df.head())
    print('\n')
    print(df_cuped.head())
