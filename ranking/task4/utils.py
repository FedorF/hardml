import hyperopt as hopt

from ranking.task4.lambda_boosting import Solution


class ModelTrainer:
    def __init__(self, path_to_model, params, param_grid, num_it=-1):
        self.path_to_model = path_to_model
        self.param_grid = param_grid
        self.params = params
        self.num_it = num_it

    def train(self):
        if self.num_it > 0:
            params = self._find_best_params()
        else:
            params = self.params

        print(params)
        model = Solution(**params)
        model.fit()
        model.save_model(self.path_to_model)

    def _find_best_params(self) -> dict:
        """Use hyperopt to find optimal model hyper-parameters."""

        def objective(pars):
            estimator = Solution(**pars)
            estimator.fit()

            return estimator.best_ndcg

        best_params = hopt.fmin(fn=objective, space=self.param_grid, algo=hopt.tpe.suggest, max_evals=self.num_it)

        return best_params
