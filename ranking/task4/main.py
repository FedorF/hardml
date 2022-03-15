from ranking.task4.lambda_boosting import Solution


if __name__ == '__main__':
    print('Start Training')
    solution = Solution(n_estimators=100,
                        lr=0.09,
                        subsample=0.99,
                        colsample_bytree=0.99,
                        max_depth=5,
                        min_samples_leaf=15,
                        )
    solution.fit()
    solution.save_model('./model.bin')
    print('Finish Training')

    print('Start Predict')
    estimator = Solution()
    estimator.load_model('./model.bin')
    y_pred = estimator.predict(estimator.X_test)
    ndcg = estimator._calc_data_ndcg(estimator.query_ids_test, estimator.ys_test, y_pred)
    print(round(ndcg, 4))
    print('Finish Predict')


'''
best ndcg: 0.3254
best ndcg: 0.3518

'''