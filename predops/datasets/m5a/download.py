import os
import zipfile
from pathlib import Path

from dotenv import load_dotenv


def get_data(DATA_PATH):
    """Download data files

    Args:
        path ([type], optional): Path for downloaded files. Defaults to DATA_PATH.
    """
    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files("m5-forecasting-accuracy", path=DATA_PATH)

    file_path = DATA_PATH / "m5-forecasting-accuracy.zip"

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(DATA_PATH)
    os.remove(file_path)
