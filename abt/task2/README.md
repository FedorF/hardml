# Вычисление метрик для мониторинга.

<br>Суммарная выручка;
<br>Количество покупок;
<br>Средний чек;
<br>Среднее количество товаров в покупке.


Дополнительно добавим возможность фильтровать данные по различным параметрам. Это может быть полезно для того, чтобы посмотреть, как меняются продажи в пилотной и контрольной группах или в какой-то отдельной категории товаров.

На вход функции будет подаваться датафрейм с данными о продажах, словарь с фильтрами и период, за который нужно посчитать метрики.

Функция должна вернуть датафрейм, в индексах которого будут все даты из указанного периода, отсортированные по возрастанию, а в столбцах — метрики ['revenue', 'number_purchases', 'average_check', 'average_number_items']. Формат данных столбцов — float, формат данных индекса — datetime64[ns].

Если в какие-то дни не было продаж, то пропуск нужно заполнить нулём.
