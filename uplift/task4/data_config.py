import datetime


data_config = {
    'calcers': [
        {
            'name': 'day_of_week_receipts',
            'args': {'delta': 30, 'date_to': datetime.date(2019, 3, 19)}
        },
        {
            'name': 'favourite_store',
            'args': {'delta': 30, 'date_to': datetime.date(2019, 3, 19)}
        },
        {
            'name': 'age_gender',
            'args': {}
        },
        {
            'name': 'target_from_campaigns',
            'args': {'date_to': datetime.date(2019, 3, 21)}
        }
    ],
    'transforms': [
        {
            'name': 'expression',
            'args': {
                'expression': "({d}['purchases_count_dw5__30d'] + {d}['purchases_count_dw6__30d']) / ({d}['purchases_count_dw0__30d'] + {d}['purchases_count_dw1__30d'] + {d}['purchases_count_dw2__30d'] + {d}['purchases_count_dw3__30d'] + {d}['purchases_count_dw4__30d'] + {d}['purchases_count_dw5__30d'] + {d}['purchases_count_dw6__30d'])",
                'col_result': "weekend_purchases_ratio__30d"
            }
        },
        {
            'name': 'expression',
            'args': {
                'expression': "{d}['target_purchases_sum'] * 0.2 - {d}['target_campaign_points_spent'] * 0.1 - {d}['treatment_flg'] * 1.5",
                'col_result': "target_profit"
            }
        },
        {
            'name': 'loo_mean_target_encoder',
            'args': {
                'col_categorical': 'gender',
                'col_target': 'target_profit',
                'col_result': "gender__mte__target_profit"
            }
        },
        {
            'name': 'loo_mean_target_encoder',
            'args': {
                'col_categorical': 'gender',
                'col_target': 'target_purchases_count',
                'col_result': "gender__mte__target_purchases_count"
            }
        }
    ]
}