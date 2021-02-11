import pandas as pd
import numpy as np

import datetime
import math, decimal

def get_moon_phase(d):  # 0=new, 4=full; 4 days/phase
    dec = decimal.Decimal
    diff = datetime.datetime.strptime(str(d), '%Y-%m-%d') - datetime.datetime(2001, 1, 1)
    days = dec(diff.days) + (dec(diff.seconds) / dec(86400))
    lunations = dec("0.20439731") + (days * dec("0.03386319269"))
    phase_index = math.floor((lunations % dec(1) * dec(8)) + dec('0.5'))
    return int(phase_index) & 7

def infer_frequency(ts):
    freq = "nanosecond"
    diff_min = ts.diff().min()
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


def extract_time_features(df, time_col):

    df[time_col] = pd.to_datetime(df[time_col])

    freq = infer_frequency(df[time_col])

    all_freqs = ["nanosecond","microsecond","millisecond","second","minute","hour","day","week","month","quarter","year"]
    rank_freq = all_freqs.index(freq)

    df["year"] = df[time_col].dt.year

    if rank_freq < all_freqs.index("year"):
        df["quarter"] = df[time_col].dt.quarter

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