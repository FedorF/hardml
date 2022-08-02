import pandas as pd
import numpy as np

from abt.task4 import stratification, config as cfg


def calc_strat_weights(df, strat_cols):
    df_size = df.shape[0]
    strats = {}
    for strat_name, strat in df.groupby(strat_cols):
        weight = strat.shape[0]/df_size
        strats[strat_name] = weight
        print(f'{strat_name}: {round(weight, 4)}')

    return strats


def generate_data(df_size, strat_cols):
    df = pd.DataFrame(
        data=[[i,
               'ios' if (np.random.uniform(size=1)[0] > .75) else 'andr',
               'man' if (np.random.uniform(size=1)[0] > .6) else 'women',
               np.random.randint(1960, 2012, 1)[0],
               'msk' if (np.random.uniform(size=1)[0] < .7) else 'spb',
               100 * np.random.randn(1)[0] + 300,
               ] for i in range(df_size)],
        columns=['id', 'os', 'gender', 'birth_year', 'city', 'revenue'],
    )
    print('General Distridution Strats Weights:')
    strat_weights = calc_strat_weights(df, strat_cols)

    return df, strat_weights


def check_result(pilot, control, strat_cols, weights):
    print('\npilot:')
    print(pilot.head())
    print(f'{pilot.shape}\n')
    print('Pilot Strats Weights:')
    pilot_weights = calc_strat_weights(pilot, strat_cols)
    for strat_name, w in pilot_weights.items():
        assert abs(w - weights[strat_name]) < 2/cfg.GROUP_SIZE

    print('\ncontrol:')
    print(control.head())
    print(f'{control.shape}\n')
    print('Control Strats Weights:')
    control_weights = calc_strat_weights(control, strat_cols)
    for strat_name, w in control_weights.items():
        assert abs(w - weights[strat_name]) < 2 / cfg.GROUP_SIZE


if __name__ == '__main__':
    df, general_strat_weights = generate_data(cfg.DATA_SIZE, cfg.STRAT_COLS)
    pilot, control = stratification.select_stratified_groups(df, cfg.STRAT_COLS, cfg.GROUP_SIZE, cfg.WEIGHTS, cfg.SEED)
    check_result(pilot, control, cfg.STRAT_COLS, cfg.WEIGHTS or general_strat_weights)
