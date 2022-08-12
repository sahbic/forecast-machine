import os
import requests
import zipfile
from pathlib import Path

from dotenv import load_dotenv
from predops import config, data_loader
from kaggle.api.kaggle_api_extended import KaggleApi

import pandas as pd

def get_data(PROJECT_KEY, DATA_PATH):
    """Download data files

    Args:
        path ([type], optional): Path for downloaded files. Defaults to DATA_PATH.
    """
    env_path = Path.cwd() / ".env"
    load_dotenv(dotenv_path=env_path)

    if PROJECT_KEY == "m5a":
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files("m5-forecasting-accuracy", path=DATA_PATH)
        file_name = "m5-forecasting-accuracy.zip"
        file_path = DATA_PATH / file_name

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(DATA_PATH)
        os.remove(file_path)
        config.logger.info("Data downloaded!")

    elif PROJECT_KEY == "m1_yearly":
        file_name = "m1_yearly_dataset.zip"
        url='https://zenodo.org/record/4656193/files/' + file_name
        # Downloading the file by sending the request to the URL
        req = requests.get(url,verify=False)
        
        file_path = DATA_PATH / file_name
        # Writing the file to the local file system
        with open(file_path,'wb') as output_file:
            output_file.write(req.content)

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(DATA_PATH)
        os.remove(file_path)

        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = data_loader.convert_tsf_to_dataframe(DATA_PATH / "m1_yearly_dataset.tsf")

        print(loaded_data)
        print(frequency)
        print(forecast_horizon)
        print(contain_missing_values)
        print(contain_equal_length)

        config.logger.info("Data downloaded!")

    else:
        config.logger.error("Project key was not recognized")