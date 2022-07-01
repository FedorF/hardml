import numpy as np
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    np.random.seed(seed)

    control, target = df_control_group[metric_name].values, df_pilot_group[metric_name].values
    n_control, n_target = control.shape[0], target.shape[0]
    p_vals = np.zeros(n_iter)
    for i in range(n_iter):
        control_cur = control[np.random.randint(low=0, high=n_control, size=n_control)]
        target_cur = target[np.random.randint(low=0, high=n_target, size=n_target)]
        p_vals[i] = ttest_ind(control_cur, target_cur)[1]

    return np.mean(p_vals <= alpha)


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000,
                               seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.

    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    np.random.seed(seed)

    control, target = df_control_group[metric_name].values, df_pilot_group[metric_name].values
    n_control, n_target = control.shape[0], target.shape[0]
    p_vals = {boost: np.zeros(n_iter) for boost in effects}
    for i in range(n_iter):
        control_cur = control[np.random.randint(low=0, high=n_control, size=n_control)]
        for boost in effects:
            target_boosted = target * boost
            target_cur = target_boosted[np.random.randint(low=0, high=n_target, size=n_target)]
            p_vals[boost][i] = ttest_ind(control_cur, target_cur)[1]

    p_vals = {boost: np.mean(p_val > alpha) for boost, p_val in p_vals.items()}

    return p_vals
