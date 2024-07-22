import hydra
from omegaconf import DictConfig
import pandas as pd
from great_expectations.data_context import FileDataContext
import sys
import os

PROJECTPATH = os.environ['PROJECTPATH']

def download_from_kaggle(url, filename):
    os.environ['KAGGLE_USERNAME'] = 'nai1ka'
    os.environ['KAGGLE_KEY'] = 'a44922ed88b8f8675c22ef8269d2232c'

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(url, path=PROJECTPATH+"/data", unzip=True)

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    """Create a sample of the dataset"""
    dataset_name = cfg.dataset_name
    output_name = cfg.output_name
    sample_version = int(cfg.sample_version[1:])

    if not os.path.exists(PROJECTPATH+"/data/"+dataset_name):
        print("File doesn't exist. Downloading")
        download_from_kaggle(cfg.kaggle_url,dataset_name)

    # Load the dataset
    df = pd.read_csv(PROJECTPATH+"/data/" + dataset_name)

    # Calculate the sample size
    sample_size = int(cfg.sample_size * len(df))

    # Create the sample
    sample_df = df[(sample_version - 1) * sample_size:sample_version * sample_size]

    # Save the sample
    sample_df.to_csv(PROJECTPATH+"/data/samples/" + output_name, index=False)


if __name__ == "__main__":
    sample_data()
