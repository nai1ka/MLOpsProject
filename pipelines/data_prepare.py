# pipelines/data_prepare.py

import pandas as pd
import zenml
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
# from data import transform_data, extract_data, load_features, validate_transformed_data

import data
# from utils import get_sample_version
import os

#BASE_PATH = os.path.expandvars("/Users/arinagoncharova/PycharmProjects/MLOpsProjectAirflow")

BASE_PATH = os.path.expandvars("$PROJECTPATH")

@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="extracted_data",
                   tags=["data_preparation"]
                   )
    ],
    Annotated[str,
    ArtifactConfig(name="data_version",
                   tags=["data_preparation"])]
]:
    df, version = data.extract_data(BASE_PATH)

    return df, version


@step(enable_cache=False)
def transform(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="input_features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="input_target",
                   tags=["data_preparation"])]
]:
    # Your data transformation code
    X, y = data.preprocess_data(df)

    return X, y


@step(enable_cache=False)
def validate(X: pd.DataFrame,
             y: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="valid_input_features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="valid_target",
                   tags=["data_preparation"])]
]:
    X, y = data.validate_features(X, y)

    return X, y


@step(enable_cache=False)
def load(X: pd.DataFrame, y: pd.DataFrame, version: str) -> Tuple[
    Annotated[pd.DataFrame,
    ArtifactConfig(name="features",
                   tags=["data_preparation"])],
    Annotated[pd.DataFrame,
    ArtifactConfig(name="target",
                   tags=["data_preparation"])]
]:
    data.load_features(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()

    version = data.get_data_version()
    print(version)
    df = data.load_artifact(name="features_target", version=version)
    print("Retrieved DataFrame:")
    print(df.head())