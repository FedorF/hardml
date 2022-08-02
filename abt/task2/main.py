import pandas as pd

from abt.task2 import metrics


if __name__ == '__main__':
    df = pd.DataFrame(
        [[800, '2021-04-03', 1], [500, '2021-04-03', 1], [1000, '2021-04-03', 2],
         [600, '2021-04-05', 1], [300, '2021-04-05', 10],
         [10000, '2021-04-10', 1],
         [100, '2021-04-11', 1],
         [550, '2021-04-12', 1]],
        columns=['cost', 'date', 'sale_id']
    )
    df = metrics.calculate_sales_metrics(df, 'cost', 'date', 'sale_id', {'begin': '2021-04-03', 'end': '2021-04-12'})
    print(df)
