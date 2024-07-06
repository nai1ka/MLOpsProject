import hydra
from omegaconf import DictConfig
import pandas as pd
from great_expectations.data_context import FileDataContext
import sys


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def sample_data(cfg: DictConfig = None):
    """Create a sample of the dataset"""
    dataset_name = cfg.dataset_name
    output_name = cfg.output_name
    sample_version = cfg.sample_version

    # Load the dataset
    df = pd.read_csv("../data/" + dataset_name)

    # Calculate the sample size
    sample_size = int(cfg.sample_size * len(df))

    # Create the sample
    sample_df = df[(sample_version - 1) * sample_size:sample_version * sample_size]

    # Save the sample
    sample_df.to_csv("../data/samples/" + output_name, index=False)


if __name__ == "__main__":
    sample_data()
