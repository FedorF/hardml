import numpy as np
import pandas as pd


def _calc_cuped(y, y_cov):
    theta = np.cov(y, y_cov)[0, 1] / np.var(y_cov)
    y_cup = y - theta*y_cov

    return y_cup


def calculate_metric(
        df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    """Вычисляет значение метрики для списка пользователей в определённый период.

    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    period - dict, словарь с датами начала и конца периода, за который нужно посчитать метрики.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}. Дата начала периода входит нужный
        полуинтервал, а дата окончание нет, то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами [user_id_name, metric_name], кол-во строк должно быть равно
        кол-ву элементов в списке list_user_id.
    """
    users = pd.DataFrame({user_id_name: list_user_id})
    metric = (
        df
        .loc[df[user_id_name].isin(list_user_id)
             & (df[date_name] >= period['begin'])
             & (df[date_name] < period['end'])]
        .groupby(user_id_name)[value_name]
        .sum()
        .reset_index()
        .rename(columns={value_name: metric_name})
    )

    return users.merge(metric, how='left').fillna(0)


def calculate_metric_cuped(
        df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    """Вычисляет метрики во время пилота, коварианту и преобразованную метрику cuped.

    df - pd.DataFrame, датафрейм с данными
    value_name - str, название столбца со значениями для вычисления целевой метрики
    user_id_name - str, название столбца с идентификаторами пользователей
    list_user_id - List[int], список идентификаторов пользователей, для которых нужно посчитать метрики
    date_name - str, название столбца с датами
    periods - dict, словарь с датами начала и конца периода пилота и препилота.
        Пример, {
            'prepilot': {'begin': '2020-01-01', 'end': '2020-01-08'},
            'pilot': {'begin': '2020-01-08', 'end': '2020-01-15'}
        }.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    metric_name - str, название полученной метрики

    return - pd.DataFrame, со столбцами
        [user_id_name, metric_name, f'{metric_name}_prepilot', f'{metric_name}_cuped'],
        кол-во строк должно быть равно кол-ву элементов в списке list_user_id.
    """

    df_cov = calculate_metric(df, value_name, user_id_name, list_user_id, date_name, periods['prepilot'], metric_name)
    df_pilot = calculate_metric(df, value_name, user_id_name, list_user_id, date_name, periods['pilot'], metric_name)
    df_pilot = df_pilot.merge(df_cov.rename(columns={metric_name: f'{metric_name}_prepilot'}))
    df_pilot[f'{metric_name}_cuped'] = _calc_cuped(df_pilot[metric_name].values, df_pilot[f'{metric_name}_prepilot'])

    return df_pilot
