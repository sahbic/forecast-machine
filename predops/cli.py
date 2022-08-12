# app/cli.py
# Command line interface (CLI) application.

from swat.utils.config import option_context
import typer
from importlib import import_module

from enum import Enum

from predops import config, main

app = typer.Typer()


@app.command()
def download_data(project_key: str):
    """Download raw data.

    Args:
        project_key (str): Project key for the organization of project specific files.
    """
    config.init_config(project_key)

    try:
        download = import_module("predops.datasets." + project_key + ".download")
        download.get_data(config.RAW_DIR)
        config.logger.info("Data downloaded!")
    except ModuleNotFoundError:
        config.logger.error("The download script must be implemented for this project")


@app.command()
def generate_base_file(
    project_key: str,
    sample: bool = typer.Option(False, "--sample"),
    time_column: str = "date",
    output_file: str = "abt.csv",
):
    """Generate base file from raw data.

    Args:
        project_key (str): Project key for the organization of project specific files.
        sample (bool, optional): activate sampling. Defaults to typer.Option(False, "--sample").
        time_column (str, optional): The name of the date/time column of the time series. Defaults to "date".
        output_file (str, optional): Output file to generate in work directory. Defaults to "abt.csv".
    """
    config.init_config(project_key)

    try:
        prepare = import_module("predops.datasets." + project_key + ".prepare")
        prepare.generate_base(
            config.RAW_DIR, config.DATA_DIR, time_column, config.logger, sample, output_file
        )
        config.logger.info(
            "Data transformed, base file generated in {}".format(str(config.DATA_DIR / output_file))
        )
    except ModuleNotFoundError:
        config.logger.error("The prepare script must be implemented for this project")


class TestMode(str, Enum):
    split = "split"
    cv = "cv"


@app.command()
def train(
    project_key: str,
    id_column: str = "id",
    time_column: str = "date",
    target: str = "value",
    number_predictions: int = 6,
    n_predictions_groupby: int = 6,
    column_segment_groupby: str = None,
    test_mode: TestMode = TestMode.split,
    n_periods: int = 3,
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
        test_mode,
        n_periods,
        config.DATA_DIR,
        input_file_name,
        config.STORES_DIR,
        config.logger,
    )

    config.logger.info("Training completed")


# TODO: add a retrain option
# if not retrain train on first period and predict on all others
@app.command()
def backtest(
    project_key: str,
    run_id: str = None,
    n_periods: int = 3,
    input_file_name: str = "abt.csv",
):
    config.init_config(project_key)
    config.logger.info("Start backtesting")

    main.backtest(project_key, run_id, n_periods, config.DATA_DIR, input_file_name, config.logger)

    config.logger.info("Backtesting completed")


if __name__ == "__main__":
    app()