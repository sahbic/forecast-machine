# predops/cli.py
# Command line interface (CLI) application.

import typer

from predops import config, main, download
from predops.datasets.m5a import prepare

app = typer.Typer()

@app.command()
def download_data(project_key: str):
    """Download raw data.

    Args:
        project_key (str): Project key for the organization of project specific files.
    """
    config.init_config(project_key)
    download.get_data(project_key, config.RAW_DIR)

# TODO:
# - add output file name option
# - remove save csv inside prepare.generate_base and save it in here
# - remove time feature engineering from base file
@app.command()
def generate_base_file(project_key: str, sample: bool = typer.Option(False, "--sample"), time_column: str = "date"):
    """Generate base file from raw data.

    Args:
        project_key (str): Project key for the organization of project specific files.
        sample (bool, optional): activate sampling. Defaults to typer.Option(False, "--sample").
        time_column (str, optional): The name of the date/time column of the time series. Defaults to "date".
    """
    config.init_config(project_key)
    prepare.generate_base(config.RAW_DIR, config.DATA_DIR, time_column, config.logger, sample)
    config.logger.info("Data transformed, base file generated!")


@app.command()
def train(
    project_key: str,
    id_column: str = "id",
    time_column: str = "date",
    target: str = "value",
    number_predictions: int = 6,
    n_predictions_groupby: int = 6,
    column_segment_groupby: str = None,
    n_folds: int = 3,
    input_file_name: str = "abt.csv",
):
    """Train models with hyperparameters tuning, and evaluate the results.

    Args:
        project_key (str): Project key for project specific code parts.
        id_column (str, optional): The name of the index column of the time series, id must be unique for each time series. Defaults to "id".
        time_column (str, optional): The name of the date/time column of the time series. Defaults to "date".
        target (str, optional): The target variable we want to predict. Defaults to "value".
        number_predictions (int, optional): [description]. Defaults to 6.
        n_predictions_groupby (int, optional): Number of predictions to group by for temporal aggregation. Defaults to 6.
        column_segment_groupby (str, optional): Name of the column to segment data with. Defaults to None.
        n_folds (int, optional): Number of folds for evaluation. Defaults to 3.
        data_dir_path (str, optional): [description]. Defaults to ".".
    """
    config.init_config(project_key)
    config.logger.info("Start training")

    if number_predictions % n_predictions_groupby != 0:
        config.logger.warning(
            "number_predictions must be a multiple of n_predictions_groupby. n_predictions_groupby is set to be equal to number_predictions (no time grouping)"
        )
        n_predictions_groupby = number_predictions

    main.train(
        project_key,
        id_column,
        time_column,
        target,
        number_predictions,
        n_predictions_groupby,
        column_segment_groupby,
        n_folds,
        config.DATA_DIR,
        input_file_name,
        config.STORES_DIR,
        config.logger)
    
    config.logger.info("Training completed")

@app.command()
def backtest(
    project_key: str,
    run_id: str = None,
    n_periods: int = 3,
    input_file_name: str = "abt.csv",
):
    config.init_config(project_key)
    config.logger.info("Start backtesting")

    main.backtest(
        project_key,
        run_id,
        n_periods,
        config.DATA_DIR,
        input_file_name,
        config.logger)
    
    config.logger.info("Backtesting completed")

# if __name__ == "__main__":
#     app()