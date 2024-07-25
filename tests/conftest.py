from hydra import compose, initialize
import os
import pytest


PROJECTPATH = os.environ['PROJECTPATH']


@pytest.fixture(scope="module")
def config():
    """Load the configuration file for the tests"""
    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="main", overrides=[f'data_path={PROJECTPATH}/data/tests/sample.csv'])
        return config