import numpy as np
import pandas as pd


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
    # Todo
    pass
