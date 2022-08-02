import numpy as np
import pandas as pd


def make_full_dates(first, last, date_name):
    dates = pd.date_range(first, last, freq=pd.DateOffset(days=1), closed='left')
    return pd.DataFrame(index=dates).reset_index().rename(columns={'index': date_name})


def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    """Вычисляет метрики по продажам.

    df - pd.DataFrame, датафрейм с данными. Пример
        pd.DataFrame(
            [[820, '2021-04-03', 1, 213]],
            columns=['cost', 'date', 'sale_id', 'shop_id']
        )
    cost_name - str, название столбца с стоимостью товара
    date_name - str, название столбца с датой покупки
    sale_id_name - str, название столбца с идентификатором покупки (в одной покупке может быть несколько товаров)
    period - dict, словарь с датами начала и конца периода пилота.
        Пример, {'begin': '2020-01-01', 'end': '2020-01-08'}.
        Дата начала периода входит в полуинтервал, а дата окончания нет,
        то есть '2020-01-01' <= date < '2020-01-08'.
    filters - dict, словарь с фильтрами. Ключ - название поля, по которому фильтруем, значение - список значений,
        которые нужно оставить. Например, {'user_id': [111, 123, 943]}.
        Если None, то фильтровать не нужно.

    return - pd.DataFrame, в индексах все даты из указанного периода отсортированные по возрастанию,
        столбцы - метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items'].
        Формат данных столбцов - float, формат данных индекса - datetime64[ns].
    """

    first, last = period['begin'], period['end']
    df = df.loc[(df[date_name] >= first) & (df[date_name] < last)]

    if filters:
        for col, vals in filters.items():
            df = df.loc[df[col].isin(vals)]

    df[date_name] = pd.to_datetime(df[date_name])
    df_revenue = df.groupby(date_name)[cost_name].sum().reset_index()
    df_revenue.columns = [date_name, 'revenue']

    df_num_purch = df.groupby(date_name)[sale_id_name].unique().map(len).reset_index()
    df_num_purch.columns = [date_name, 'number_purchases']

    df_av_check = (
        df
        .groupby([date_name, sale_id_name])[cost_name]
        .sum()
        .reset_index()
        .groupby(date_name)[cost_name]
        .mean()
        .reset_index()
    )
    df_av_check.columns = [date_name, 'average_check']

    df_av_num_items = (
        df
        .groupby([date_name, sale_id_name])[cost_name]
        .count()
        .reset_index()
        .groupby(date_name)[cost_name]
        .mean()
        .reset_index()
    )
    df_av_num_items.columns = [date_name, 'average_number_items']

    df = (
        df_revenue
        .merge(df_num_purch)
        .merge(df_av_check)
        .merge(df_av_num_items)
    )
    df_result = make_full_dates(first, last, date_name).merge(df, how='outer')
    df_result.index = df_result[date_name]

    return df_result.drop(columns=[date_name]).fillna(0)
