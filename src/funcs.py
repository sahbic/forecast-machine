import pandas as pd
import numpy as np

import datetime
import math, decimal
from slugify import slugify

from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
    dec = decimal.Decimal
    diff = datetime.datetime.strptime(str(d), '%Y-%m-%d') - datetime.datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
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


def extract_time_features(df, time_col, id_col, log):
    log.info('generate time features')

    df[time_col] = pd.to_datetime(df[time_col])

    freq = infer_frequency(df,time_col, id_col)

    all_freqs = ["nanosecond","microsecond","millisecond","second","minute","hour","day","week","month","quarter","year"]
    rank_freq = all_freqs.index(freq)

    df["year"] = df[time_col].dt.year.astype(np.int8)

    if rank_freq < all_freqs.index("year"):
        df["quarter"] = df[time_col].dt.quarter.astype(np.int8)

    if rank_freq < all_freqs.index("quarter"):
        df["month"] = df[time_col].dt.month.astype(np.int8)

    if rank_freq < all_freqs.index("month"):
        df["week"] = df[time_col].dt.week.astype(np.int8)
        days = df[time_col].dt.day
        df['weekmonth'] = days.apply(lambda x: ceil(x / 7)).astype(np.int8)

    if rank_freq < all_freqs.index("week"):
        df["dayofweek"] = df[time_col].dt.dayofweek.astype(np.int8)
        df["weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
        df["day"] = days
        df['moon'] = df.date.apply(get_moon_phase)

    if rank_freq < all_freqs.index("day"):
        df["hour"] = df[time_col].dt.hour

    if rank_freq < all_freqs.index("hour"):
        df["minute"] = df[time_col].dt.minute

    if rank_freq < all_freqs.index("minute"):
        df["second"] = df[time_col].dt.second

    if rank_freq < all_freqs.index("millisecond"):
        df["microsecond"] = df[time_col].dt.microsecond

    if rank_freq < all_freqs.index("microsecond"):
        df["nanosecond"] = df[time_col].dt.nanosecond

    return df

def shift_with_pred_horizon(df, dependent_var, predict_horizon):
    df["target"] = df[dependent_var].shift(-predict_horizon)
    return df

def add_last_value(df, id_col, dependent_var, predict_horizon):
    var_name = "last_"+str(predict_horizon)
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
            df = df.drop(col,axis=1)
    log.info("dropped categories: "+str(len(dropped)))
    return df

def one_hot_encode_categorical_columns(df, cat_cols, log):
    non_constant_cat = [cat for cat in cat_cols if cat in df.columns]
    for category in non_constant_cat:
        df[category] = df[category].apply(lambda x :slugify(x))
    
    df = pd.get_dummies(df, columns=non_constant_cat)
    df.columns = [x.replace("-", "_") for x in df.columns]
    log.info("encoded categories: "+str(len(non_constant_cat)))
    return df

def get_num_cat_columns(df, params, log):
    cols = df.columns
    num_cols = df._get_numeric_data().columns
    cat_cols = list(set(cols) - set(num_cols))
    # remove time and id from categorical
    cat_cols.remove(params.time_col)
    cat_cols.remove(params.id_col)
    # remove dependant variable from numerical
    num_cols = list(set(num_cols))
    num_cols.remove(params.dependent_var)
    # num_cols.remove("target")
    return num_cols, cat_cols

def preprocess_ml_sklearn_forests(df, cat_cols, params, log):
    df = drop_constant_columns(df, log)
    # _, cat_cols = get_num_cat_columns(df, params, log)
    df = one_hot_encode_categorical_columns(df, cat_cols, log)
    return df

def impute_missing_mean(train, val, num_cols, params, log):
    # num_cols, _ = get_num_cat_columns(train, params, log)
    float_cols = list(train.columns[(train.dtypes.values == np.dtype('float64'))])
    float_cols.remove(params.dependent_var)
    float_cols.remove("target")

    mean_imputer = SimpleImputer()
    a = mean_imputer.fit_transform(train.loc[:, float_cols])
    b = mean_imputer.transform(val.loc[:, float_cols])

    train_imp = train.copy()
    val_imp = val.copy()

    train_imp.loc[:, float_cols] = a
    val_imp.loc[:, float_cols] = b

    log.info("imputed numerical columns: "+str(len(float_cols)))
    return train_imp, val_imp


def compute_metric(res, test, params):
    res = res.merge(test[[params.id_col, params.time_col, params.dependent_var]], on=[params.id_col,params.time_col], how="left")
    error = mean_squared_error(res["prediction"], res[params.dependent_var])
    return error