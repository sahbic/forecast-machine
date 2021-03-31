import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class Model(object):
    def __init__(self, id_col, time_col, dependent_var, log):
        self.id_col = id_col
        self.time_col = time_col
        self.dependent_var = dependent_var
        self.log = log

    def fit(self, train):
        pass

    def predict(self, test):
        return np.zeros(len(test))


class Mean(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.mean_value = None

    def fit(self, train):
        self.mean_value = train[self.dependent_var].mean()

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res["prediction"] = self.mean_value
        return res


class MeanTS(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.means = None

    def fit(self, train):
        self.means = pd.DataFrame(
            train.groupby(self.id_col)[self.dependent_var].mean()
        ).reset_index()
        self.means.rename(columns={self.dependent_var: "prediction"}, inplace=True)

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res = res.merge(self.means, on=self.id_col, how="left")
        return res


class Last(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "Last"
        self.last = None
        self.best_params = None
        self.best_index = None
        self.cv_results = None

    def fit(self, train):
        pass

    def fit_with_params(self, df):
        pass

    def tune_fit(self, train, splitter, n_iter):
        cv_results = {}
        i = 0
        scores = []
        for train_idx, test_idx in splitter:
            test = train.iloc[test_idx]
            preds = self.predict(test)
            preds = preds.fillna(0)

            preds = preds.merge(
                test[[self.id_col, self.time_col, self.dependent_var]],
                on=[self.id_col, self.time_col],
                how="left",
            )
            res = mean_squared_error(preds["prediction"], preds[self.dependent_var])

            cv_results["split" + str(i) + "_test_score"] = [res]
            scores.append(res)
            i = i + 1

        cv_results["split" + str(i) + "_test_score"] = [res]
        cv_results["mean_test_score"] = [np.mean(scores)]
        self.cv_results = cv_results
        self.best_index = 0

    def predict(self, test):
        var_name = "last"
        res = test.loc[:, [self.id_col, self.time_col, var_name]]
        res[var_name] = res[var_name].fillna(0)
        res.rename(columns={var_name: "prediction"}, inplace=True)
        return res


class RandomForest(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "RandomForest"
        self.model = None
        self.best_params = None
        self.best_index = None
        self.parameters_space = {
            "model__n_estimators": [50],
            "model__max_features": ["auto", "sqrt", "log2"],
            "model__min_samples_split": [2, 5, 10],
            "model__max_depth": [3, 5, 10, None],
            "model__criterion": ["mse", "mae"],
        }

    def get_pipeline(self):
        # Define data pipelines
        numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, make_column_selector(dtype_include=np.number)),
                ("cat", categorical_transformer, make_column_selector(dtype_include=object)),
            ]
        )

        rf_model = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", RandomForestRegressor(random_state=3))]
        )

        return rf_model

    def fit(self, df):

        X = df.drop(columns=[self.id_col, self.time_col, self.dependent_var])
        y = df[self.dependent_var]

        rf_model = self.get_pipeline()

        rf_model.fit(X, y)
        self.model = rf_model

    def fit_with_params(self, df):

        X = df.drop(columns=[self.id_col, self.time_col, self.dependent_var])
        y = df[self.dependent_var]

        rf_model = self.get_pipeline()
        rf_model.set_params(**self.best_params)

        rf_model.fit(X, y)
        self.model = rf_model

    def tune_fit(self, df, splitter, n_iter):

        X = df.drop(columns=[self.id_col, self.time_col, self.dependent_var])
        y = df[self.dependent_var]

        rf_model = self.get_pipeline()

        search = RandomizedSearchCV(
            rf_model,
            param_distributions=self.parameters_space,
            cv=splitter,
            scoring=make_scorer(mean_squared_error),
            n_iter=n_iter,
            n_jobs=-1,
        )
        search.fit(X, y)

        self.model = search.best_estimator_
        self.best_params = search.best_params_
        self.cv_results = search.cv_results_
        self.best_index = search.best_index_

    def predict(self, test):
        preds = self.model.predict(
            test.drop(columns=[self.id_col, self.time_col, self.dependent_var])
        )
        res = test.loc[:, [self.id_col, self.time_col]]
        res["prediction"] = preds
        return res
