import os
import pandas as pd
from src.sample_data import sample_data


PROJECTPATH = os.environ['PROJECTPATH']


def test_sample_data_creates_sample(config):
    """Test if the sample_data function creates the sample file"""
    sample_data(config)
    # Check if the sample file is created
    assert os.path.exists(PROJECTPATH+"/data/samples/sample.csv"), "Sample file was not created"


def test_sample_data_correct_sample_size(config):
    """Test if the sample_data function creates the sample file with the correct size"""
    sample_data(config)
    # Load the dataset and the sample
    df = pd.read_csv(PROJECTPATH+"/data/" + config.dataset_name)
    sample_df = pd.read_csv(PROJECTPATH+"/data/samples/" + config.output_name)

    # Check the sample size
    expected_size = int(config.sample_size * len(df))
    assert len(sample_df) == expected_size, f"Sample size is incorrect. Expected {expected_size}, got {len(sample_df)}"
