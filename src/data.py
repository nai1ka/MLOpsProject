import pandas as pd
import os
import yaml
import zenml
from great_expectations import DataContext
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from omegaconf import OmegaConf

import sample_data



def read_datastore(base_path: str) -> pd.DataFrame:
    path = os.path.join(base_path, 'data/samples', 'sample.csv')
    df = pd.read_csv(path)
    return df

def get_data_version(config_path="../configs", config_name="main") -> str:
    with initialize(config_path=config_path, version_base=None):
        config = compose(config_name=config_name)
        return str(config.sample_version)

def extract_data(base_path: str) -> (pd.DataFrame, str):
    # cfg = OmegaConf.load(os.path.join('configs', 'main.yaml'))
    # return sample_data.sample_data(cfg)
    df = read_datastore(base_path)
    version = get_data_version()
    return df, version

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Your data preprocessing code
    X = df.drop(columns=['price'])
    y = pd.DataFrame(df['price'])
    return X, y

def validate_features(X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    context = FileDataContext(context_root_dir="../services/gx")
    suite_name = "data_validation"
    context.add_or_update_expectation_suite(suite_name)

    # Load the expectation suite


    # Add a pandas datasource (if not already added)
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")

    # Add a CSV asset for the new data
    sample = ds.add_csv_asset(
        name="sample_csv",
        filepath_or_buffer="data/samples/sample.csv"
    )
    data_asset = ds.add_dataframe_asset(name="sample_data")
    # Build a batch request
    # batch_request = sample.build_batch_request()
    batch_request = data_asset.build_batch_request(dataframe=X)

    # Get a validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    validator.expect_column_values_to_be_between("hour", min_value=0, max_value=24)

    validator.save_expectation_suite(discard_failed_expectations=False)

    checkpoint = context.add_or_update_checkpoint(
        name="my_checkpoint",
        validations=[
            {
                "batch_request": batch_request,
                "expectation_suite_name": suite_name,
            },
        ],
    )

    checkpoint.run()

    return (X, y)

def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str) -> None:
    # Example: Save the processed features
    df = pd.concat([X, y], axis=1)
    zenml.save_artifact(data=df, name="features_target", tags=[version])

def load_artifact(name: str, version: str) -> pd.DataFrame:
    return zenml.load_artifact(name, version)