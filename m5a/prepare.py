import os
import swat

from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
from pandas.tseries.offsets import DateOffset

import src.funcs as funcs

def generate_base(raw_dir_path, work_dir_path, time_column, logger, sampling):
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
    
    HOST_IP = os.getenv("HOST_IP")
    USER_NAME = os.getenv("USER_NAME")
    PASSWORD = os.getenv("PASSWORD")
    HOST_PORT = os.getenv("HOST_PORT")

    # Connect to CAS server
    s = swat.CAS(HOST_IP, int(HOST_PORT), USER_NAME, PASSWORD, caslib="Public")

    final_test = s.tableExists(caslib="CASUSER",name="M5_Final")
    if final_test.exists == 0:
        # Path to files
        calendar_path = str(raw_dir_path / "calendar.csv")
        sales_train_validation_path = str(raw_dir_path / "sales_train_validation.csv")
        sell_prices_path = str(raw_dir_path / "sell_prices.csv")

        # Test if tables are loaded
        sales_train_validation_test = s.tableExists(caslib="Public",name="M5_SALES_TRAIN_VALIDATION")
        sell_prices_test = s.tableExists(caslib="Public",name="M5_SELL_PRICES")
        calendar_test = s.tableExists(caslib="Public",name="M5_CALENDAR")

        # If not load tables to memory
        if sales_train_validation_test.exists == 0:
            sales_train_validation = s.read_csv(sales_train_validation_path, casout={"name":"M5_SALES_TRAIN_VALIDATION", "caslib":"Public", "promote":"True"})

        if sell_prices_test.exists == 0:
            sell_prices = s.read_csv(sell_prices_path, casout={"name":"M5_SELL_PRICES", "caslib":"Public", "promote":"True"})

        if calendar_test.exists == 0:
            calendar = s.read_csv(calendar_path, casout={"name":"M5_CALENDAR", "caslib":"Public", "promote":"True"})

        # Reference loaded tables
        sales_train_validation = s.CASTable('M5_SALES_TRAIN_VALIDATION',caslib='Public')
        sell_prices = s.CASTable('M5_SELL_PRICES',caslib='Public')
        calendar = s.CASTable('M5_CALENDAR',caslib='Public')

        # Transform Data

        ds = s.dataStep.runCode(code='''
        data CASUSER.M5_CALENDAR;
            set PUBLIC.M5_CALENDAR;
            format d2 $6.;
            d2=d;
            FORMAT newdate yymmdd10.;
            newdate=input(Date, yymmdd10.);
            drop date d;
            rename newdate=date d2=d;
        run;
        ''')

        ds = s.dataStep.runCode(code='''
        data CASUSER.M5_SALES_TRAIN_VALIDATION;
            set PUBLIC.M5_SALES_TRAIN_VALIDATION;
            s=1;
        run;
        ''')

        d_cols = {c for c in sales_train_validation.columns if 'd_' in c}
        dname = lambda name: dict(name=name) # helper function to make the action call code more clear
        ddrop = lambda name,drop: dict(name=name,drop=drop) # helper function to make the action call code more clear
        ddrop_list = lambda ls: [ddrop(el,True) for el in ls] # helper function to make the action call code more clear

        # Transpose

        s.loadactionset(actionset="transpose")

        s.transpose.transpose(
            table={"caslib":"CASUSER","name":"M5_SALES_TRAIN_VALIDATION","groupby":[dname("state_id"), dname("store_id"), dname("dept_id"), dname("cat_id"), dname("item_id"), dname("id")]},
            transpose=d_cols, id={"s"}, casOut={"caslib":"CASUSER", "name":"M5_SALES_TRANSPOSED", "replace":"true"}
        )

        # Rename columns
        s.table.alterTable(caslib="CASUSER", name="M5_SALES_TRANSPOSED",columns=[{"name":"1", "rename":"Quantity"},{"name":"_NAME_", "rename":"d"}])
        # Joins
        s.builtins.loadActionSet("fedSql")
        date=s.fedSql.execDirect(
        query='''create table CASUSER.SALES_Calendar as
            select a.id, a.item_id, a.dept_id, a.cat_id, a.store_id, a.state_id, a.Quantity, b.*
            from CASUSER.M5_SALES_TRANSPOSED as a 
            left join CASUSER.M5_CALENDAR as b
            on a.d=b.d'''
        )

        date=s.fedSql.execDirect(
        query='''create table CASUSER.M5_Final as
            select a.*, b.sell_price
            from CASUSER.SALES_CALENDAR as a 
            left join PUBLIC.M5_SELL_PRICES as b
            on (a.item_id=b.item_id and a.store_id=b.store_id and a.wm_yr_wk=b.wm_yr_wk)'''
        )

        s.table.alterTable(caslib="CASUSER",name="M5_Final",columns = ddrop_list(["d","weekday","year","wm_yr_wk","event_name_1","event_name_2","event_type_1","event_type_2"]))
        s.promote(caslib="CASUSER",name="M5_Final", targetLib="CASUSER")
    
    if sampling == True:
        logger.info("sampling data")

        # filters
        ds = s.dataStep.runCode(code='''
        data CASUSER.M5_Sample;
            set CASUSER.M5_Final;
            where state_id='TX' and store_id IN ('TX_1', 'TX_2') and dept_id='HOBBIES_1' and date>='01jan2014'd and date<'01jan2016'd; *new line;
            If state_id='CA'  then snap=snap_CA; 
            else if State_id='TX'  then snap=snap_TX; 
            else snap=snap_WI; 
            drop snap_CA snap_TX snap_WI;
        run;
        ''')

        sample = s.CASTable("M5_Sample", caslib="CASUSER")
        # products = sample.item_id.sample(10).item_id.values
    
        logger.info('take a sample of the data')
        products = ['HOBBIES_1_123','HOBBIES_1_362','HOBBIES_1_352','HOBBIES_1_052','HOBBIES_1_336','HOBBIES_1_224','HOBBIES_1_033','HOBBIES_1_280','HOBBIES_1_205','HOBBIES_1_142']
        sample = sample[sample.item_id.isin(list(products))]
    else:
        ds = s.dataStep.runCode(code='''
        data CASUSER.M5_Sample;
            set CASUSER.M5_Final;
            If state_id='CA'  then snap=snap_CA; 
            else if State_id='TX'  then snap=snap_TX; 
            else snap=snap_WI; 
            drop snap_CA snap_TX snap_WI;
        run;
        ''')
        sample = s.CASTable("M5_Sample", caslib="CASUSER")
    
    df = sample.to_frame()
    logger.info("number of rows = {}".format(len(df)))

    # add time related features
    df = add_time_related_features(df, time_column, logger)

    swat.options.cas.dataset.max_rows_fetched = 20000
    sample_path = str(work_dir_path / "abt.csv")
    df.to_csv(sample_path, index=False)

    s.terminate()

    return df


def add_time_related_features(df, time_col, log):
    """add time features

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
    time_features = funcs.extract_time_features(time_features, time_col, log)
    # select time features to keep from ["nanosecond","microsecond","second","minute","hour","day","weekend","moon","dayofweek","week","weekmonth","month","quarter","year"]
    time_features = time_features[[time_col, "day","weekend","moon","dayofweek","week","weekmonth","month","quarter","year"]]
    # convert date column to date
    df[time_col] = pd.to_datetime(df[time_col])
    # join new features
    df = df.merge(time_features, on=time_col)
    return df


def generate_grid(df, id_col, dependent_var, predict_horizon, work_dir_path, log):
    log.info("- generate grid: shifts and aggergations")
    # temporal feature engineering (depends on prediction horizon)
    df = funcs.add_last_value(df, id_col, dependent_var, predict_horizon)

    file_name = "temp_"+ str(predict_horizon) + ".csv"
    temp_path = str(work_dir_path / file_name)
    df.to_csv(temp_path, index=False)

    return df

def get_offset(number_predictions):
    """
    translate number of prediction into number of time steps in DateOffset format
    """
    return DateOffset(days=number_predictions)