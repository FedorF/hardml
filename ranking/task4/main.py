from ranking.task4.lambda_boosting import Solution


if __name__ == '__main__':
    solution = Solution(n_estimators=100,
                        lr=0.01,
                        subsample=0.7,
                        colsample_bytree=0.95,
                        max_depth=3,
                        min_samples_leaf=5,
                        )
    print('Start Training')
    solution.fit()
    solution.save_model('./model.bin')
    print('Finish Training')
    print('Start Predict')
    estimator = Solution()
    estimator.load_model('./model.bin')
    y_pred = estimator.predict(estimator.X_test)
    print(y_pred.shape)
    ndcg = estimator._calc_data_ndcg(estimator.query_ids_test, estimator.ys_test, y_pred)
    print(round(ndcg, 4))

'''
best ndcg: 0.3254
best ndcg: 0.3518

'''