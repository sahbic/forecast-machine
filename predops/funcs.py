import datetime
import decimal
import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from slugify import slugify


def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
    dec = decimal.Decimal
    diff = d - datetime.datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    phase_index = math.floor((lunations % dec(1) * dec(8)) + dec("0.5"))
    return int(phase_index) & 7


def infer_frequency(df, time_col, id_col):
    freq = "nanosecond"
    diff_min = df.groupby(id_col)[time_col].diff().min()
    if diff_min >= pd.Timedelta(1, unit="microsecond"):
        freq = "microsecond"
    if diff_min >= pd.Timedelta(1, unit="millisecond"):
        freq = "millisecond"
    if diff_min >= pd.Timedelta(1, unit="second"):
        freq = "second"
    if diff_min >= pd.Timedelta(1, unit="minute"):
        freq = "minute"
    if diff_min >= pd.Timedelta(1, unit="hours"):
        freq = "hour"
    if diff_min >= pd.Timedelta(1, unit="days"):
        freq = "day"
    if diff_min >= pd.Timedelta(7, unit="days"):
        freq = "week"
    if diff_min >= pd.Timedelta(28, unit="days"):
        freq = "month"
    if diff_min >= pd.Timedelta(90, unit="days"):
        freq = "quarter"
    if diff_min >= pd.Timedelta(365, unit="days"):
        freq = "year"
    return freq


def extract_time_features(df, time_col, log):
    log.info("generate time features")

    df[time_col] = pd.to_datetime(df[time_col])

    # freq = infer_frequency(df,time_col, id_col)
    # all_freqs = ["nanosecond","microsecond","millisecond","second","minute","hour","day","week","month","quarter","year"]
    # rank_freq = all_freqs.index(freq)

    df["year"] = df[time_col].dt.year

    df["quarter"] = df[time_col].dt.quarter.astype(np.int8)

    df["month"] = df[time_col].dt.month.astype(np.int8)

    df["week"] = df[time_col].dt.isocalendar().week
    days = df[time_col].dt.day
    df["weekmonth"] = days.apply(lambda x: math.ceil(x / 7)).astype(np.int8)

    df["dayofweek"] = df[time_col].dt.dayofweek.astype(np.int8)
    df["weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
    df["day"] = days
    df["moon"] = df.date.apply(get_moon_phase)

    df["hour"] = df[time_col].dt.hour

    df["minute"] = df[time_col].dt.minute

    df["second"] = df[time_col].dt.second

    df["microsecond"] = df[time_col].dt.microsecond

    df["nanosecond"] = df[time_col].dt.nanosecond

    return df


def shift_with_pred_horizon(df, dependent_var, predict_horizon):
    df["target"] = df[dependent_var].shift(-predict_horizon)
    return df


def add_last_value(df, id_col, dependent_var, predict_horizon):
    var_name = "last"
    res = df.copy()
    res[var_name] = df.groupby(id_col)[dependent_var].shift(predict_horizon)
    return res


def split(df, time_col, end_train_time, end_test_time):
    train = df[df[time_col] <= end_train_time]
    test = df[(df[time_col] > end_train_time) & (df[time_col] <= end_test_time)]
    return train, test


def split_with_time_grouping(df, time_col, end_train_time, begin_test_time, end_test_time):
    train = df[df[time_col] <= end_train_time]
    test = df[(df[time_col] >= begin_test_time) & (df[time_col] <= end_test_time)]
    return train, test


def drop_constant_columns(df, log):
    dropped = []
    for col in df.columns:
        if len(df[col].unique()) == 1:
            dropped.append(col)
            df = df.drop(col, axis=1)
    log.info("dropped columns: " + str(len(dropped)))
    return df


def get_num_cat_columns(df, params, log):
    cols = df.drop(columns=[params.id_col, params.time_col, params.dependent_var], axis=1).columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    num_cols = list(set(num_cols))
    return num_cols, cat_cols


def compute_metric(res, test, id_col, time_col, dependent_var):
    res = res.merge(
        test[[id_col, time_col, dependent_var]],
        on=[id_col, time_col],
        how="left",
    )
    error = mean_squared_error(res["prediction"], res[dependent_var])
    return error


def get_splitter(df, time_col, n_folds, number_predictions):
    time_frame = pd.DataFrame({time_col: pd.Series(df[time_col].unique())})
    time_frame[time_col] = pd.to_datetime(time_frame[time_col])
    time_frame = time_frame.sort_values(by=time_col).reset_index()
    tscv = TimeSeriesSplit(n_splits=n_folds, test_size=number_predictions)
    splitter = []
    for train_index, test_index in tscv.split(time_frame):
        X_train, X_test = time_frame.iloc[train_index], time_frame.iloc[test_index]
        train_indexes = df.index[df[time_col].isin(X_train[time_col])]
        test_indexes = df.index[df[time_col].isin(X_test[time_col])]
        splitter.append((train_indexes, test_indexes))
    return splitter
