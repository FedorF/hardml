import numpy as np


def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.

    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    """

    mean = np.mean(values)
    gap = 1.96 * (mean * (1 - mean) / values.shape[0]) ** 0.5

    return np.clip(mean - gap, 0, 1), np.clip(mean + gap, 0, 1)
