# Add a function sample_data in a module src/data.py to read the data file and take a sample. Use Hydra to manage the data url in the configs and any configurations needed like the size of the sample, the name of the dataset,…etc. You need to define a structure for project configurations and use Hydra to read them from the source code. The function should read the data url from config files using Hydra and sample the data. Then, it stores the data samples in data/samples folder as sample.csv where the sample size is 20% of the total rows in the whole data. The sample size can be added to the config files too. The idea here is to make your project maintainable and ready for any changes with minimal modifications (DRY principle). For each new sample added, version the data using dvc to not lose any data samples.
# Run the function sample_data and it should generate a sample. The function should overwrite the file sample.csv if it already exists. You can manage the data versions using dvc.
import json

import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import os


# TODO add comments

def create_kaggle_json(username, key):
    kaggle_credentials = {
        "username": username,
        "key": key
    }
    os.makedirs("~/.kaggle")

    # Path to the kaggle.json file
    f = open("~/.kaggle/kaggle.json", 'w')
    f.write("Now the file has more content!")
    f.close()


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    os.environ['KAGGLE_USERNAME'] = 'nai1ka'
    os.environ['KAGGLE_KEY'] = 'a44922ed88b8f8675c22ef8269d2232c'
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    data_url = cfg.data.url
    sample_size = cfg.data.sample_size
    dataset_name = cfg.data.dataset_name

    kaggle.api.dataset_download_files(data_url, path="../data/temp", unzip=True)

    df = pd.read_csv("../data/temp/" + dataset_name)
    sample_df = df.sample(frac=sample_size, random_state=1)
    sample_df.to_csv("../data/samples/sample.csv", index=False)

    # Clean up temporary directory
    os.remove("../data/temp/" + dataset_name)
    os.rmdir("../data/temp")


if __name__ == "__main__":
    sample_data()
