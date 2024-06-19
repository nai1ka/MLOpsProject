import os
import pytest
from hydra import compose, initialize
from src.data import sample_data


@pytest.fixture(scope="module")
def setup_kaggle_env():
    os.environ['KAGGLE_USERNAME'] = 'nai1ka'
    os.environ['KAGGLE_KEY'] = 'a44922ed88b8f8675c22ef8269d2232c'


@pytest.fixture(scope="module")
def cfg():
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="main")
        return cfg


def test_sample_data_creates_sample(setup_kaggle_env, cfg):
    sample_data(cfg)
    # Check if the sample file is created
    assert os.path.exists("../data/samples/sample.csv"), "Sample file was not created"


def test_sample_data_correct_sample_size(setup_kaggle_env, cfg):
    # TODO: Implement the test
    pass
