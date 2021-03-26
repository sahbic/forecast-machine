# app/cli.py
# Command line interface (CLI) application.

import typer

from m5a import config, download
from m5a.config import logger

app = typer.Typer()

@app.command()
def download_data(project_key: str):
    # config = __import__(project_key+".config", fromlist=[])
    # download = __import__(project_key+".download", fromlist=["get_data"])
    print(config.DATA_DIR)
    # download.get_data(path=config.DATA_DIR)
    logger.info("✅ Data downloaded!")


@app.command()
def train_tune_eval(project_key: str,
                    id_column: str = "id",
                    time_column: str = "date",
                    target: str = "value",
                    number_predictions: int = 6,
                    n_predictions_groupby: int = 6,
                    column_segment_groupby: str = None,
                    n_folds: int = 3,
                    data_dir_path: str = ".",
                    output_name: str = "default",
):
    """Train models with hyperparameters tuning, and evaluate the results

    Args:
        project_key (str): Project key for project specific code parts
        id_column (str, optional): The name of the index column of the time series, id must be unique for each time series. Defaults to "id".
        time_column (str, optional): The name of the date/time column of the time series. Defaults to "date".
        target (str, optional): The target variable we want to predict. Defaults to "value".
        number_predictions (int, optional): [description]. Defaults to 6.
        n_predictions_groupby (int, optional): Number of predictions to group by for temporal aggregation. Defaults to 6.
        column_segment_groupby (str, optional): Name of the column to segment data with. Defaults to None.
        n_folds (int, optional): Number of folds for evaluation. Defaults to 3.
        data_dir_path (str, optional): [description]. Defaults to ".".
        output_name (str, optional): [description]. Defaults to "default".
    """


if __name__ == "__main__":
    app()