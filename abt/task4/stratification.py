import numpy as np
import pandas as pd


def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    """Подбирает стратифицированные группы для эксперимента.

    data - pd.DataFrame, датафрейм с описанием объектов, содержит атрибуты для стратификации.
    strat_columns - List[str], список названий столбцов, по которым нужно стратифицировать.
    group_size - int, размеры групп.
    weights - dict, словарь весов страт {strat: weight}, где strat - либо tuple значений элементов страт,
        например, для strat_columns=['os', 'gender', 'birth_year'] будет ('ios', 'man', 1992), либо просто строка/число.
        Если None, определить веса пропорционально доле страт в датафрейме data.
    seed - int, исходное состояние генератора случайных чисел для воспроизводимости
        результатов. Если None, то состояние генератора не устанавливается.

    return (data_pilot, data_control) - два датафрейма того же формата, что и data
        c пилотной и контрольной группами.
    """
    np.random.seed(seed)

    data_size = data.shape[0]
    data_pilot, data_control = [], []
    for strat_name, strat in data.groupby(strat_columns):
        if weights:
            # если не хотим учитывать какую-то определенную страту
            if strat_name not in weights:
                continue
            else:
                sample_size = group_size * weights[strat_name]
        else:
            sample_size = group_size * strat.shape[0] / data_size

        sample_size = round(sample_size)
        sample = strat.sample(n=2*sample_size)
        pilot, control = sample.iloc[:sample_size], sample.iloc[sample_size:]
        data_pilot.append(pilot)
        data_control.append(control)

    data_pilot = pd.concat(data_pilot).reset_index(drop=True)
    data_control = pd.concat(data_control).reset_index(drop=True)

    return data_pilot, data_control
