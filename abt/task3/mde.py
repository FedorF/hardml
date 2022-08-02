import numpy as np
import pandas as pd
from scipy.stats import norm


def calc_min_sample_size(alpha, beta, eff, var1, var2):
    ppf_a, ppf_b = norm.ppf(1-alpha), norm.ppf(1-beta)
    sample_size = (var1 + var2) * (ppf_a + ppf_b)**2 / eff**2
    return int(np.ceil(sample_size))


def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    """Оцениваем sample size для списка эффектов.

    df - pd.DataFrame, датафрейм с данными
    metric_name - str, название столбца с целевой метрикой
    effects - List[float], список ожидаемых эффектов. Например, [1.03] - увеличение на 3%
    alpha - float, ошибка первого рода
    beta - float, ошибка второго рода

    return - pd.DataFrame со столбцами ['effect', 'sample_size']
    """
    alpha /= 2
    metric = df[metric_name].values
    mean, var = np.mean(metric), np.var(metric)

    sample_sizes = [calc_min_sample_size(alpha, beta, (eff-1)*mean, var, var) for eff in effects]

    return pd.DataFrame(data={'effect': effects, 'sample_size': sample_sizes})
