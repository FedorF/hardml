import category_encoders as cat
import pandas as pd
import sklearn.base as skbase


class ExpressionTransformer(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, expression, col_result):
        self.expression = expression
        self.col_result = col_result

    def fit(self, *args, **kwargs):
        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        data[self.col_result] = eval(self.expression.format(d='data'))

        return data


class LOOMeanTargetEncoder(skbase.BaseEstimator, skbase.TransformerMixin):

    def __init__(self, col_categorical, col_target, col_result):
        self.col_categorical = col_categorical
        self.col_target = col_target
        self.col_result = col_result
        self.encoder = cat.LeaveOneOutEncoder()

    def fit(self, data: pd.DataFrame, *args, **kwargs):
        X_train = data[[self.col_categorical]]
        y_train = data[self.col_target] if self.col_target in data.columns else None
        self.encoder.fit(X_train, y_train)

        return self

    def transform(self, data: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        X_test = data[[self.col_categorical]]
        y_test = data[self.col_target] if self.col_target in data.columns else None
        data[self.col_result] = self.encoder.transform(X_test, y_test)[self.col_categorical]

        return data
