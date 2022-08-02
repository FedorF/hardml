import pandas as pd
import numpy as np


class SequentialTester:
    def __init__(
            self, metric_name, time_column_name,
            alpha, beta, pdf_one, pdf_two
    ):
        """Создаём класс для проверки гипотезы о равенстве средних тестом Вальда.

        Предполагается, что среднее значение метрики у распределения альтернативной
        гипотезы с плотность pdf_two больше.

        :param metric_name: str, название стобца со значениями измерений.
        :param time_column_name: str, названия столбца с датой и временем измерения.
        :param alpha: float, допустимая ошибка первого рода.
        :param beta: float, допустимая ошибка второго рода.
        :param pdf_one: function, функция плотности распределения метрики при H0.
        :param pdf_two: function, функция плотности распределения метрики при H1.
        """
        self.metric_name = metric_name
        self.time_column_name = time_column_name
        self.alpha = alpha
        self.beta = beta
        self.pdf_one = pdf_one
        self.pdf_two = pdf_two
        self.control = None
        self.pilot = None
        self.upper_bound = np.log((1-beta)/alpha)
        self.lower_bound = np.log(beta/(1-alpha))

    def run_test(self, data_control, data_pilot):
        """Запускаем новый тест, проверяет гипотезу о равенстве средних.

        :param data_control: pd.DataFrame, данные контрольной группы.
        :param data_pilot: pd.DataFrame, данные пилотной группы.

        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        self.control = data_control
        self.pilot = data_pilot

        data_control = data_control.sort_values(self.time_column_name)[self.metric_name].values
        data_pilot = data_pilot.sort_values(self.time_column_name)[self.metric_name].values

        min_len = min([len(data_pilot), len(data_control)])

        data_pilot = data_pilot[:min_len]
        data_control = data_control[:min_len]

        xs = data_pilot - data_control
        t = np.cumsum(np.log(self.pdf_two(xs) / self.pdf_one(xs)))

        indexes_lower = np.arange(min_len)[t < self.lower_bound]
        indexes_upper = np.arange(min_len)[t > self.upper_bound]

        first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else min_len + 1
        first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else min_len + 1

        if first_index_lower < first_index_upper:
            return 0, first_index_lower + 1
        elif first_index_lower > first_index_upper:
            return 1, first_index_upper + 1
        else:
            return 0.5, min_len

    def add_data(self, data_control, data_pilot):
        """Добавляет новые данные, проверяет гипотезу о равенстве средних.

        Гарантируется, что данные новые и не дублируют ранее добавленные.

        :param data_control: pd.DataFrame, новые данные контрольной группы.
        :param data_pilot: pd.DataFrame, новые данные пилотной группы.

        :return (result, length):
            result: float,
                0 - отклоняем H1,
                1 - отклоняем H0,
                0.5 - недостаточно данных для принятия решения
            length: int, сколько потребовалось данных для принятия решения. Если данных
                недостаточно, то возвращает текущее кол-во данных. Кол-во данных - это
                кол-во элементов в одном из наборов data_control или data_pilot.
                Гарантируется, что они равны.
        """
        pilot = pd.concat([self.pilot, data_pilot])
        control = pd.concat([self.control, data_control])
        return self.run_test(control, pilot)
