import os
import swat
import math

from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from pandas.tseries.offsets import DateOffset

import predops.funcs as funcs

def generate_base(raw_dir_path, work_dir_path, time_column, logger, sampling, output_file):
    """Generate base file (on SAS Viya)

    Args:
        raw_dir_path (Path): input path to raw data
        work_dir_path (Path): output path
        time_column {str}: The name of the date/time column of the time series.
        logger (logging.Logger): Logger
        sampling (bool): activate sampling option

    Returns:
        pandas.DataFrame: output DataFrame
    """
    logger.info('start base file generation')

    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)
    
    train_path = raw_dir_path / "train.csv"

    df = pd.read_csv(train_path)
    df["Date"] = pd.to_datetime(df["Date"], infer_datetime_format=True)

    df_full = (
        df.groupby(["Store", "Dept"])["Date"]
        .apply(lambda s: pd.date_range(df.Date.min(), df.Date.max(), freq="W-FRI"))
        .explode()
        .reset_index()
    )

    df_fullv = df_full.merge(df, how='left')
    df_fullv = df_fullv.drop(columns="IsHoliday")
    df_fullv["Weekly_Sales"] = df_fullv["Weekly_Sales"].fillna(0)
    df_fullv['id'] = df_fullv.groupby(['Store', 'Dept']).ngroup()

    df_fullv.to_csv(work_dir_path / output_file, index=False)

    return df_fullv


def add_calendar_features(df, time_col, log):
    """add calendar features

    Args:
        df (pandas.DataFrame): input dataframe
        time_col (str): The name of the date/time column of the time series. Defaults to "date".
        log (logging.Logger): logger.

    Returns:
        pandas.DataFrame: Dataframe with time features (for seasonality)
    """
    log.info("- time features: adding time related data")
    # init time features DataFrame
    time_features = pd.DataFrame({time_col: pd.Series(df[time_col].unique())})
    # extract date features
    time_features = funcs.extract_calendar_features(time_features, time_col, log)
    # select time features to keep from ["nanosecond","microsecond","second","minute","hour","day","weekend","moon","dayofweek","week","weekmonth","month","quarter","year"]
    time_features = time_features[[time_col,"weekend","moon","dayofweek","week","weekmonth","month","quarter","year"]]
    # convert date column to date
    df[time_col] = pd.to_datetime(df[time_col])
    # join new features
    df = df.merge(time_features, on=time_col)
    return df

def generate_grid(df, id_col, time_col, dependent_var, predict_horizon, work_dir_path, log):
    log.info("generate grid: feature engineering")
    # add time related features
    df = add_calendar_features(df, time_col, log)

    # temporal feature engineering (depends on prediction horizon)
    # TODO: get num_lag_days 15 from config file
    # df = funcs.add_lag_days(df, id_col, dependent_var, predict_horizon, 15)

    num_rolling_day_list = [4, 8, 16]
    # TODO: get rolling list from config file
    # df = funcs.add_rolling_aggs(df, id_col, dependent_var, predict_horizon, num_rolling_day_list)

    # add business feature engineering
    # df = add_business_features(df, id_col, dependent_var)

    file_name = "temp_"+ str(predict_horizon) + ".csv"
    temp_path = str(work_dir_path / file_name)
    df.to_csv(temp_path, index=False)

    return df

def add_business_features(df, id_col, target):
    # TODO: add release week feature
    return df

def add_price_features(prices_df, calendar_df):
    prices_df['price_max'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('max')
    prices_df['price_min'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('min')
    prices_df['price_std'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('std')
    prices_df['price_mean'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('mean')
    prices_df['price_norm'] = prices_df['sell_price'] / prices_df['price_max']
    prices_df['price_nunique'] = prices_df.groupby(['store_id', 'item_id'])['sell_price'].transform('nunique')
    prices_df['item_nunique'] = prices_df.groupby(['store_id', 'sell_price'])['item_id'].transform('nunique')

    calendar_prices = calendar_df[['wm_yr_wk', 'month', 'year']]
    calendar_prices = calendar_prices.drop_duplicates(subset=['wm_yr_wk'])
    prices_df = prices_df.merge(calendar_prices[['wm_yr_wk', 'month', 'year']], on=['wm_yr_wk'], how='left')
    del calendar_prices

    prices_df['price_momentum'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id'])[
        'sell_price'].transform(lambda x: x.shift(1))
    prices_df['price_momentum_m'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'month'])[
        'sell_price'].transform('mean')
    prices_df['price_momentum_y'] = prices_df['sell_price'] / prices_df.groupby(['store_id', 'item_id', 'year'])[
        'sell_price'].transform('mean')

    prices_df['sell_price_cent'] = [math.modf(p)[0] for p in prices_df['sell_price']]
    prices_df['price_max_cent'] = [math.modf(p)[0] for p in prices_df['price_max']]
    prices_df['price_min_cent'] = [math.modf(p)[0] for p in prices_df['price_min']]

    del prices_df['month'], prices_df['year']

    return prices_df

def get_offset(number_predictions):
    """
    translate number of prediction into number of time steps in DateOffset format
    """
    return DateOffset(days=number_predictions)