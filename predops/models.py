import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import swat

import pipefitter
from pipefitter.estimator import DecisionTree, DecisionForest, GBTree
from pipefitter.transformer import Imputer
from pipefitter.model_selection import HyperParameterTuning

from pathlib import Path
from dotenv import load_dotenv

import os
import uuid
import json
import random
import string
import requests

import mlflow


class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, id_col, time_col, dependent_var, log):
        self.id_col = id_col
        self.time_col = time_col
        self.dependent_var = dependent_var
        self.log = log
        self.cv_results = {}
        self.parameters_space = None
        self.conda_env = None
        self.name = ""

    def fit(self, train):
        pass

    def predict(self, test):
        return np.zeros(len(test))

    def track(self, exp_id, tags, test_mode, n_periods):
        for i in range(len(self.cv_results["mean_test_score"])):
            # Start run
            with mlflow.start_run(run_name=self.name, nested=True):
                # Track parameters
                if self.parameters_space:
                    params = list(self.parameters_space.keys())
                    for param in params:
                        mlflow.log_param(param, self.cv_results["param_%s" % param][i])

                # Track metrics
                mlflow.log_metric("average_mse", self.cv_results["mean_test_score"][i])
                mlflow.log_metric("std_mse", self.cv_results["std_test_score"][i])

                if test_mode == "cv":
                    for j in range(n_periods):
                        mlflow.log_metric(
                            "split" + str(j) + "_test_score",
                            self.cv_results["split" + str(j) + "_test_score"][i],
                        )

                # Track extra data related to the experiment
                mlflow.set_tags(tags)

                # Log model
                if self.conda_env:
                    mlflow.pyfunc.log_model(
                        artifact_path="model", python_model=self, conda_env=self.conda_env
                    )


class Mean(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.mean_value = None
        self.name = "Mean"
        self.best_params = None
        self.best_index = None
        self.conda_env = {
            "name": "pandas-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
            ],
        }

    def get_name(self):
        return self.name

    def fit_with_params(self, df):
        self.fit(df)

    def fit(self, train):
        self.mean_value = train[self.dependent_var].mean()

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res["prediction"] = self.mean_value
        return res

    def tune_fit(self, df, splitter, n_iter):
        cv_results = {}
        i = 0
        scores = []
        for train_idx, test_idx in splitter:
            train = df.iloc[train_idx]
            self.fit(train)
            test = df.iloc[test_idx]
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

        # cv_results["split" + str(i) + "_test_score"] = [res]
        cv_results["mean_test_score"] = [np.mean(scores)]
        cv_results["std_test_score"] = [np.std(scores)]
        self.cv_results = cv_results
        self.best_index = 0
        self.best_params = {}


class MeanTS(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.means = None
        self.name = "MeanTS"
        self.best_params = None
        self.best_index = None
        self.conda_env = {
            "name": "pandas-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
            ],
        }

    def get_name(self):
        return self.name

    def fit(self, train):
        self.means = pd.DataFrame(
            train.groupby(self.id_col)[self.dependent_var].mean()
        ).reset_index()
        self.means.rename(columns={self.dependent_var: "prediction"}, inplace=True)

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res = res.merge(self.means, on=self.id_col, how="left")
        return res

    def fit_with_params(self, df):
        self.fit(df)

    def tune_fit(self, df, splitter, n_iter):
        cv_results = {}
        i = 0
        scores = []
        for train_idx, test_idx in splitter:
            train = df.iloc[train_idx]
            self.fit(train)
            test = df.iloc[test_idx]
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

        # cv_results["split" + str(i) + "_test_score"] = [res]
        cv_results["mean_test_score"] = [np.mean(scores)]
        cv_results["std_test_score"] = [np.std(scores)]
        self.cv_results = cv_results
        self.best_index = 0
        self.best_params = {}


class Last(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "Last"
        self.last = None
        self.best_params = None
        self.best_index = None
        self.conda_env = {
            "name": "pandas-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
            ],
        }

    def get_name(self):
        return self.name

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

        # cv_results["split" + str(i) + "_test_score"] = [res]
        cv_results["mean_test_score"] = [np.mean(scores)]
        cv_results["std_test_score"] = [np.std(scores)]
        self.cv_results = cv_results
        self.best_index = 0
        self.best_params = {}

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
        # TODO
        # Change this requirements dynamically ?
        self.parameters_space = {
            "model__n_estimators": [100],
            "model__max_features": ["sqrt", "log2", 1.0],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 5, 10],
            "model__max_depth": [6, 12, None],
        }
        self.conda_env = {
            "name": "sklearn-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "scikit-learn==0.24.1",
            ],
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


class ViyaModel(Model):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "ViyaModel"
        self.session = None
        self.model = None
        self.best_params = None
        self.best_index = None
        self.parameters_space = {}
        self.conda_env = {
            "name": "swat-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
                "swat==1.6.1",
                "pipefitter==1.0.0",
            ],
        }

    def get_session(self):
        env_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path=env_path)

        HOST_IP = os.getenv("HOST_IP")
        USER_NAME = os.getenv("USER_NAME")
        PASSWORD = os.getenv("PASSWORD")
        HOST_PORT = os.getenv("HOST_PORT")

        if self.session:
            s = swat.CAS(HOST_IP, int(HOST_PORT), USER_NAME, PASSWORD, session=self.session)
        else:
            s = swat.CAS(HOST_IP, int(HOST_PORT), USER_NAME, PASSWORD)
            # self.session = s.session.sessionId()

        return s

    def get_inputs(self, df):
        varnames = list(df.columns.values) + ["index"]
        rejected = [self.id_col, self.time_col, self.dependent_var, "index"]
        inputs = [item for item in varnames if item not in rejected]

        return inputs

    def to_cv_results(self, df):
        columns = json.loads(df.to_json(orient="columns"))
        res = {}
        res["mean_test_score"] = df["MeanScore"].values
        res["std_test_score"] = df["ScoreStd"].values
        res["params"] = list(columns["Parameters"].values())
        for parameter in res["params"][0]:
            res["param_" + str(parameter)] = []
            for run in res["params"]:
                res["param_" + str(parameter)].append(run[parameter])
        for i in range(len(list(df["FoldScores"][0]))):
            res["split" + str(i) + "_test_score"] = [
                df["FoldScores"][j][i] for j in range(df["FoldScores"].shape[0])
            ]
        return res

    def getToken(self):
        env_path = Path.cwd() / ".env"
        load_dotenv(dotenv_path=env_path)

        HOST_IP = os.getenv("HOST_IP")
        USER_NAME = os.getenv("USER_NAME")
        PASSWORD = os.getenv("PASSWORD")

        url = "http://{}/SASLogon/oauth/token?grant_type=password&username={}&password={}".format(
            HOST_IP, USER_NAME, PASSWORD
        )

        payload = {}
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic c2FzLmVjOg==",
        }

        response = requests.request("GET", url, headers=headers, data=payload)
        dict_res = json.loads(response.text)

        return "Bearer " + dict_res["access_token"]

    def createMLPAProject(self, oauthToken, project_name, table_name, target):

        HOST_IP = os.getenv("HOST_IP")
        USER_NAME = os.getenv("USER_NAME")

        url = "http://{}/mlPipelineAutomation/projects".format(HOST_IP)

        headers = {
            "Authorization": oauthToken,
            "Accept": "application/vnd.sas.analytics.ml.pipeline.automation.project+json",
            "Content-Type": "application/json",
        }
        payload = {
            "dataTableUri": "/dataTables/dataSources/cas~fs~cas-shared-default~fs~CASUSER({})/tables/".format(
                USER_NAME
            )
            + table_name,
            "type": "predictive",
            "name": project_name,
            "description": "Project generated for test",
            "settings": {"autoRun": True, "modelingMode": "Standard", "maxModelingTime": 10},
            "analyticsProjectAttributes": {"targetVariable": target},
        }

        payload_data = json.dumps(payload, indent=4)

        response = requests.request("POST", url, headers=headers, data=payload_data)
        dict_res = json.loads(response.text)

        return dict_res


class ViyaGradientBoosting(ViyaModel):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "ViyaGradientBoosting"
        self.session = None
        self.model = None
        self.best_params = None
        self.best_index = None
        self.parameters_space = dict(
            max_depth=[4, 6, 10],
            leaf_size=[3, 5],
            n_bins = [5, 20 , 50],
            n_trees=[100]
        )
        self.conda_env = {
            "name": "swat-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
                "swat==1.6.1",
                "pipefitter==1.0.0",
            ],
        }

    def get_pipeline(self, df):
        # Define data pipeline
        inputs = self.get_inputs(df)
        params = dict(target=self.dependent_var, inputs=inputs)
        model = DecisionTree(**params)

        pipe = pipefitter.pipeline.Pipeline([model])

        return pipe

    def fit_with_params(self, df):

        s = self.get_session()
        table_name = "PREDOPS-TRAIN" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(df, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)

        pipe = self.get_pipeline(df)
        pipe.set_params(**self.best_params)

        self.model = pipe.fit(castbl)
        self.model_table = self.model.stages[0].data
        s.promote(self.model_table)

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

    def tune_fit(self, df, splitter, n_iter):

        s = self.get_session()
        table_name = "PREDOPS-SEARCH" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(df, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)

        pipe = self.get_pipeline(df)

        generator = (
            (
                castbl.query("index in ({})".format(str(list(splitter[i][0]))[1:-1])),
                castbl.query("index in ({})".format(str(list(splitter[i][1]))[1:-1])),
            )
            for i in range(len(splitter))
        )

        hpt = HyperParameterTuning(estimator=pipe, param_grid=self.parameters_space, cv=generator)
        # TODO: Viya n_jobs in config parameters
        search = hpt.gridsearch(castbl).reset_index()

        # self.model = hpt
        # TODO: save table name in order to be able to score
        # Does the hpt retrain the model on all folds ?
        self.best_index = search["MeanScore"].idxmin()
        self.best_params = search.loc[self.best_index, "Parameters"]
        self.cv_results = self.to_cv_results(search)

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

    def predict(self, test):
        s = self.get_session()
        table_name = "PREDOPS-TEST" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(test, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)
        model_table = self.model.stages[0].data

        s.loadactionset("decisiontree")
        s.table.dropTable("KSCORE", quiet="True")
        r = castbl.decisiontree.gbtreescore(
            modeltable=model_table,
            copyvars=[self.id_col, self.time_col, self.dependent_var],
            casout={"name": "KSCORE", "promote": "True"},
        )

        res = s.CASTable("KSCORE").to_frame()
        res.rename(columns={"_DT_PredMean_": "prediction"}, inplace=True)
        res = res[[self.id_col, self.time_col, "prediction"]]
        res[self.time_col] = pd.to_datetime(res[self.time_col])

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

        return res


class ViyaDecisionTree(ViyaModel):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "ViyaDecisionTree"
        self.session = None
        self.model = None
        self.best_params = None
        self.best_index = None
        self.parameters_space = dict(
            max_depth=[4, 6, 10],
            leaf_size=[3, 5],
            n_bins = [5, 20 , 50]
        )
        self.conda_env = {
            "name": "swat-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
                "swat==1.6.1",
                "pipefitter==1.0.0",
            ],
        }

    def get_pipeline(self, df):
        # Define data pipeline
        inputs = self.get_inputs(df)
        params = dict(target=self.dependent_var, inputs=inputs)
        model = DecisionTree(**params)

        pipe = pipefitter.pipeline.Pipeline([model])

        return pipe

    def fit_with_params(self, df):

        s = self.get_session()
        table_name = "PREDOPS-TRAIN" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(df, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)

        pipe = self.get_pipeline(df)
        pipe.set_params(**self.best_params)

        self.model = pipe.fit(castbl)
        self.model_table = self.model.stages[0].data
        s.promote(self.model_table)

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

    def tune_fit(self, df, splitter, n_iter):

        s = self.get_session()
        table_name = "PREDOPS-SEARCH" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(df, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)

        pipe = self.get_pipeline(df)

        generator = (
            (
                castbl.query("index in ({})".format(str(list(splitter[i][0]))[1:-1])),
                castbl.query("index in ({})".format(str(list(splitter[i][1]))[1:-1])),
            )
            for i in range(len(splitter))
        )

        hpt = HyperParameterTuning(estimator=pipe, param_grid=self.parameters_space, cv=generator)
        # TODO: Viya n_jobs in config parameters
        search = hpt.gridsearch(castbl).reset_index()

        # self.model = hpt
        # TODO: save table name in order to be able to score
        # Does the hpt retrain the model on all folds ?
        self.best_index = search["MeanScore"].idxmin()
        self.best_params = search.loc[self.best_index, "Parameters"]
        self.cv_results = self.to_cv_results(search)

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

    def predict(self, test):
        s = self.get_session()
        table_name = "PREDOPS-TEST" + "-" + str(uuid.uuid4())
        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(test, casout=dict(name=table_name, promote=True))

        castbl = s.CASTable(table_name)
        model_table = self.model.stages[0].data

        s.loadactionset("decisiontree")
        s.table.dropTable("KSCORE", quiet="True")
        r = castbl.decisiontree.dtreescore(
            modeltable=model_table,
            copyvars=[self.id_col, self.time_col, self.dependent_var],
            casout={"name": "KSCORE", "promote": "True"},
        )

        res = s.CASTable("KSCORE").to_frame()
        res.rename(columns={"_DT_PredMean_": "prediction"}, inplace=True)
        res = res[[self.id_col, self.time_col, "prediction"]]
        res[self.time_col] = pd.to_datetime(res[self.time_col])

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

        return res


class ViyaMLPA(ViyaModel):
    def __init__(self, id_col, time_col, dependent_var, log):
        Model.__init__(self, id_col, time_col, dependent_var, log)
        self.name = "ViyaMLPA"
        self.session = None
        self.model = None
        self.best_params = None
        self.best_index = None
        self.conda_env = {
            "name": "swat-env",
            "channels": ["defaults"],
            "dependencies": [
                "python=3.9.2",
                "cloudpickle==1.6.0",
                "pandas==1.2.2",
                "numpy==1.20.1",
                "swat==1.6.1",
                "pipefitter==1.0.0",
            ],
        }

    def fit_with_params(self, df):
        pass

    def tune_fit(self, df, splitter, n_iter):

        s = self.get_session()
        table_name = "PREDOPS-SEARCH" + "-" + str(uuid.uuid4())
        project_name = (
            "PREDOPS_MLPA" + str(random.randint(1, 100)) + random.choice(string.ascii_letters)
        )
        # table_name = project_name + "_TRAIN"

        train_idx = list(splitter[0][0])
        test_idx = list(splitter[0][1])

        df["_PARTIND_"] = None
        df.loc[train_idx, "_PARTIND_"] = 1
        df.loc[test_idx, "_PARTIND_"] = 0

        s.table.dropTable(name=table_name, quiet="True")
        s.upload_frame(df, casout=dict(name=table_name, promote=True))

        var_list = list()
        # for col in self.get_inputs(df):
        #     var_list.append({"name" : col})
        for col in ["wday", "sell_price", "last", "_PARTIND_"]:
            var_list.append({"name": col})
        var_list.append({"name": self.dependent_var})

        print(var_list)

        s.table.dropTable(name=project_name + "_TRAIN", quiet="True")
        s.table.partition(
            table={"name": table_name, "vars": var_list},
            casout={"name": project_name + "_TRAIN", "promote": True},
        )

        # -----------------------------------------------
        # Get authentication token
        # -----------------------------------------------

        oauthToken = self.getToken()

        # -----------------------------------------------
        # REST request for creating MLPA project
        # -----------------------------------------------

        mlpaProject = self.createMLPAProject(
            oauthToken, project_name, project_name + "_TRAIN", self.dependent_var
        )

        self.log.info(mlpaProject)

        # -----------------------------------------------
        # Poll every 5 seconds until MLPA project state is completed
        # -----------------------------------------------

        # projectStateLink = list(filter(lambda x: x["rel"] == "state", mlpaProject["links"]))[0]

        s.table.dropTable(name=table_name, quiet="True")
        s.terminate()

    def predict(self, test):
        pass