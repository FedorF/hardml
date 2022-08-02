import os
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from flask import Flask, jsonify, request


def make_strats(df):
    df['age_bins'] = None
    df.loc[(df['age'] <= 20) | (df['age'] > 40), 'age_bins'] = 1
    df.loc[(df['age'] > 20) & (df['age'] <= 25), 'age_bins'] = 2
    df.loc[(df['age'] > 25) & (df['age'] <= 35), 'age_bins'] = 3
    df.loc[(df['age'] > 35) & (df['age'] <= 40), 'age_bins'] = 2

    df['strat'] = df.groupby(['age_bins', 'gender']).ngroup()
    weights = (df['strat'].value_counts() / df.shape[0]).to_dict()

    return df.drop(columns=['age_bins']), weights


def preprocess(df):
    df_all_users = pd.DataFrame(data={'user_id': df['user_id'].unique()})
    df = df.groupby(['day', 'user_id'])['sales'].sum().reset_index()

    df['week_num'] = np.floor(df['day'] / 7)
    df_prepilot = (
        df
        .loc[(df['week_num'] == 6)]
        .groupby('user_id')
        .agg(
            sales_sum_cov=('sales', 'sum'),
            sales_cnt_cov=('sales', 'count'),
        )
        .reset_index()
    )
    df_pilot = (
        df
        .loc[df['week_num'] == 7]
        .groupby('user_id')
        .agg(
            sales_sum=('sales', 'sum'),
            sales_cnt=('sales', 'count'),
        )
        .reset_index()
    )
    out = (
        df_all_users
        .merge(df_prepilot, how='left')
        .merge(df_pilot, how='left')
        .fillna(0)
    )

    return out


def calculate_linearized_metric(df, control_users):
    mask_control = df['user_id'].isin(control_users)
    kappa = df.loc[mask_control, 'sales_sum'].values.sum() / df.loc[mask_control, 'sales_cnt'].values.sum()

    df['metric_lin'] = (df['sales_sum'] - kappa*df['sales_cnt']).fillna(0)
    df['covar_lin'] = (df['sales_sum_cov'] - kappa * df['sales_cnt_cov']).fillna(0)
    return df


def _calc_cuped(y, y_cov):
    theta = np.cov(y, y_cov)[0, 1] / np.var(y_cov)
    y_cup = y - theta*y_cov
    return y_cup


def calculate_cuped_metric(df, metric_name, covar_name):
    df[f'{metric_name}_cuped'] = _calc_cuped(df[metric_name].values, df[covar_name].values)
    return df


def remove_outliers(df, y_name, q=.99):
    return df[df[y_name] < np.quantile(df[y_name].values, q)]


def calc_alpha(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=1000):
    control, target = df_control_group[metric_name].values, df_pilot_group[metric_name].values
    n_control, n_target = control.shape[0], target.shape[0]
    p_vals = np.zeros(n_iter)
    for i in range(n_iter):
        control_cur = control[np.random.randint(low=0, high=n_control, size=n_control)]
        target_cur = target[np.random.randint(low=0, high=n_target, size=n_target)]
        p_vals[i] = ttest_ind(control_cur, target_cur)[1]

    return np.mean(p_vals <= alpha)


def _calc_strat_mean(df: pd.DataFrame, strat_column: str, target_name: str, weights: dict):
    strat_mean = df.groupby(strat_column)[target_name].mean()
    return (strat_mean * pd.Series(weights)).sum()


def _calc_strat_var(df: pd.DataFrame, strat_column: str, target_name: str, weights: dict):
    strat_var = df.groupby(strat_column)[target_name].var()
    return (strat_var * pd.Series(weights)).sum()


DF_USERS, WEIGHTS = make_strats(pd.read_csv(os.environ['PATH_DF_USERS']))
DF_PILOT = preprocess(pd.read_csv(os.environ['PATH_DF_SALES']))

app = Flask(__name__)


@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))


def _check_test(test, run_aa_test=False):
    group_a_one, group_a_two, group_b = test['group_a_one'], test['group_a_two'], test['group_b']
    user_a = group_a_one + group_a_two
    user_b = group_b

    df = DF_PILOT.loc[DF_PILOT['user_id'].isin(user_a + user_b)]
    df = calculate_linearized_metric(df, control_users=user_a)
    df = calculate_cuped_metric(df, metric_name='metric_lin', covar_name='covar_lin')
    df = remove_outliers(df, y_name='metric_lin_cuped')

    df = df.merge(DF_USERS)

    if run_aa_test:
        aa_has_effect = run_post_strat_test(df, group_a_one, group_a_two, 'strat', 'metric_lin_cuped', WEIGHTS)
        if aa_has_effect:
            return 0

    return run_post_strat_test(df, user_a, user_b, 'strat', 'metric_lin_cuped', WEIGHTS)


def run_student_test(df, user_a, user_b, metric_name, alpha=0.05):
    sales_a = df[df['user_id'].isin(user_a)][metric_name].values
    sales_b = df[df['user_id'].isin(user_b)][metric_name].values
    return ttest_ind(sales_a, sales_b)[1] < alpha


def run_post_strat_test(df, user_a, user_b, strat_column, target_name, weights):
    df_control = df[df['user_id'].isin(user_a)]
    df_pilot = df[df['user_id'].isin(user_b)]

    mean_strat_pilot = _calc_strat_mean(df_pilot, strat_column, target_name, weights)
    mean_strat_control = _calc_strat_mean(df_control, strat_column, target_name, weights)
    var_strat_pilot = _calc_strat_var(df_pilot, strat_column, target_name, weights)
    var_strat_control = _calc_strat_var(df_control, strat_column, target_name, weights)

    delta_mean_strat = mean_strat_pilot - mean_strat_control
    std_mean_strat = (var_strat_pilot / len(df_pilot) + var_strat_control / len(df_control)) ** 0.5

    if delta_mean_strat - 1.96*std_mean_strat > 0 and delta_mean_strat + 1.96*std_mean_strat > 0:
        return 1
    elif delta_mean_strat - 1.96*std_mean_strat < 0 and delta_mean_strat + 1.96*std_mean_strat < 0:
        return 1
    return 0
