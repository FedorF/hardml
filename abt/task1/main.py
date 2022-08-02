import pandas as pd
import numpy as np

from abt.task1 import error as er


def generate_data(n_pilot, n_control, metric_name='revenue', n_effects=10):
    df_pilot = pd.DataFrame({metric_name: 300 + 150*np.random.randn(n_pilot)})
    df_control = pd.DataFrame({metric_name: 300 + 150*np.random.randn(n_control)})
    effects = 1 + np.random.uniform(size=n_effects) / 10

    return df_pilot, df_control, metric_name, effects


if __name__ == '__main__':
    df_pilot_group, df_control_group, metric_name, effects = generate_data(1000, 1000)
    er1 = er.estimate_first_type_error(df_pilot_group, df_control_group, metric_name, n_iter=1000)
    er2 = er.estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, n_iter=1000)
    print(er1)
    print(sorted(er2.items(), key=lambda x: x[0]))
