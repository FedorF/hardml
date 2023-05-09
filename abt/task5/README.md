# Задача 1. Вычисление метрик для CUPED.
Наша цель — научиться получать преобразованную с помощью CUPED метрику.

Для этого нужно будет написать две функции.

Первая функция вспомогательная, она будет вычислять значения целевой метрики для пользователей на произвольном промежутке времени. Первую функцию можно будет использовать, чтобы посчитать значения метрики пользователей во время пилота и до пилота.

Вторая функция будет вычислять непосредственно преобразованную cuped-метрику. В качестве ковариаты будем использовать значение метрики, посчитанное на периоде до начала пилота.

Целевая метрика — суммарная стоимость покупок пользователя за определённый период времени.