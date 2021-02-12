import pandas as pd
import numpy as np

class Model(object):
    def __init__(self, id_col, time_col, dependent_var):
        self.id_col = id_col
        self.time_col = time_col
        self.dependent_var = dependent_var

    def fit(self, train):
        pass

    def predict(self, test):
        return np.zeros(len(test))
 

class Mean(Model):
    def __init__(self, id_col, time_col, dependent_var):
        Model.__init__(self, id_col, time_col, dependent_var)
        self.mean_value = None

    def fit(self, train):
        self.mean_value = train[self.dependent_var].mean()

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res["prediction"] = self.mean_value
        return res


class MeanTS(Model):
    def __init__(self, id_col, time_col, dependent_var):
        Model.__init__(self, id_col, time_col, dependent_var)
        self.means = None

    def fit(self, train):
        self.means = pd.DataFrame(train.groupby(self.id_col)[self.dependent_var].mean()).reset_index()
        self.means.rename(columns={self.dependent_var:"prediction"}, inplace=True)

    def predict(self, test):
        res = test.loc[:, [self.id_col, self.time_col]]
        res = res.merge(self.means, on=self.id_col, how="left")
        return res


# class Previous(Model):
#     def __init__(self, id_col, time_col, dependent_var):
#         Model.__init__(self, id_col, time_col, dependent_var)
#         self.mean_value = None

#     def fit(self, train, target_column, predict_horizon):
#         self.mean_value = train[target_column].mean()

#     def predict(self, test):
#         return self.mean_value*np.ones(len(test))