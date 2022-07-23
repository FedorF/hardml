import os
import json
import time

from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# получить данные о пользователях и их покупках
df_users = pd.read_csv(os.environ['PATH_DF_USERS'])
df_sales = pd.read_csv(os.environ['PATH_DF_SALES'])

# эксперимент проводился с 49 до 55 день включительно
df_sales = df_sales[
    df_sales['day'].isin(np.arange(49, 56))
]


app = Flask(__name__)

@app.route('/ping')
def ping():
    return jsonify(status='ok')


@app.route('/check_test', methods=['POST'])
def check_test():
    test = json.loads(request.json)['test']
    has_effect = _check_test(test)
    return jsonify(has_effect=int(has_effect))

def _check_test(test):
    group_a_one = test['group_a_one']
    group_a_two = test['group_a_two']
    group_b = test['group_b']

    user_a = group_a_one + group_a_two
    user_b = group_b

    sales_a = df_sales[
        df_sales['user_id'].isin(user_a)
    ][
        'sales'
    ].values

    sales_b = df_sales[
        df_sales['user_id'].isin(user_b)
    ][
        'sales'
    ].values

    return ttest_ind(sales_a, sales_b)[1] < 0.05
