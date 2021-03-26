from dotenv import load_dotenv
from pathlib import Path


def get_data(path=DATA_PATH):
    """Download data files

    Args:
        path ([type], optional): Path for downloaded files. Defaults to DATA_PATH.
    """
    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)
    
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    api.competition_download_files('m5-forecasting-accuracy', path=DATA_PATH)