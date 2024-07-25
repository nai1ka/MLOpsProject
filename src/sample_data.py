import hydra
from omegaconf import DictConfig
import pandas as pd
from great_expectations.data_context import FileDataContext
import os

PROJECTPATH = os.environ['PROJECTPATH']


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    """
    Create a sample of the dataset based on the configuration.

    Args:
        cfg (DictConfig): Configuration object containing parameters for sampling.
    """

    # Extract configuration parameters
    dataset_name = cfg.dataset_name
    output_name = cfg.output_name
    sample_version = int(cfg.sample_version[1:])

    # Check if the dataset file exists; if not, download it
    # Used only for CI/CD pipeline, since dataset is stored locally
    if not os.path.exists(PROJECTPATH+"/data/"+dataset_name):
        print("File doesn't exist. Downloading...")
        df = pd.read_csv(cfg.google_drive_url)
        df.to_csv(PROJECTPATH+"/data/"+dataset_name, index=False)
    else:
        df = pd.read_csv(PROJECTPATH+"/data/" + dataset_name)

    # Calculate the sample size
    sample_size = int(cfg.sample_size * len(df))

    # Create the sample
    sample_df = df[(sample_version - 1) * sample_size:sample_version * sample_size]

    # Save the sample
    sample_df.to_csv(PROJECTPATH+"/data/samples/" + output_name, index=False)

if __name__ == "__main__":
    sample_data()
