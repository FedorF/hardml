from ranking.task4.lambda_boosting import Solution
from ranking.task4.params import PATH_TO_MODEL, PARAM_GRID, PARAMS
from ranking.task4.utils import ModelTrainer


if __name__ == '__main__':
    print('Start Training')
    trainer = ModelTrainer(PATH_TO_MODEL, PARAMS, PARAM_GRID, -1)
    trainer.train()
    print('Finish Training')

    print('Start Predict')
    estimator = Solution()
    estimator.load_model(PATH_TO_MODEL)
    y_pred = estimator.predict(estimator.X_test)
    ndcg = estimator._calc_data_ndcg(estimator.query_ids_test, estimator.ys_test, y_pred)
    print(round(ndcg, 4))
    print('Finish Predict')
