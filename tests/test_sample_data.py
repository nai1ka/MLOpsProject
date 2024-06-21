import os
import pytest
from hydra import compose, initialize
from src.data import sample_data
import pandas as pd

@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="main")
        return cfg


def test_sample_data_creates_sample(cfg):
    sample_data(cfg)
    # Check if the sample file is created
    assert os.path.exists("../data/samples/sample.csv"), "Sample file was not created"


def test_sample_data_correct_sample_size(cfg):
    sample_data(cfg)
    # Load the dataset and the sample
    df = pd.read_csv("../data/" + cfg.data.dataset_name)
    sample_df = pd.read_csv("../data/samples/" + cfg.data.output_name)

    # Check the sample size
    expected_size = int(cfg.data.sample_size * len(df))
    assert len(sample_df) == expected_size, f"Sample size is incorrect. Expected {expected_size}, got {len(sample_df)}"
