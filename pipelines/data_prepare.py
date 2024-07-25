from hydra import compose, initialize
import pandas as pd
from typing_extensions import Tuple, Annotated
from zenml import step, pipeline, ArtifactConfig
import data


@step(enable_cache=False)
def extract() -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="extracted_data", tags=["data_preparation"]),
    ],
    Annotated[str, ArtifactConfig(name="data_version", tags=["data_preparation"])],
]:
    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="main")
        df, version = data.extract_data(cfg=config)
        print(df.shape, version)
        return df, version


@step(enable_cache=False)
def transform(df: pd.DataFrame, version: str) -> Tuple[
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_features", tags=["data_preparation"])
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="input_target", tags=["data_preparation"])
    ],
]:
    with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="main")
        X, y = data.transform_data(
            df=df, version=version, cfg=config, transformer_version="v1"
        )
        return X, y


@step(enable_cache=False)
def validate(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
    Annotated[
        pd.DataFrame,
        ArtifactConfig(name="valid_input_features", tags=["data_preparation"]),
    ],
    Annotated[
        pd.DataFrame, ArtifactConfig(name="valid_target", tags=["data_preparation"])
    ],
]:
     with initialize(config_path="../configs", version_base=None):
        config = compose(config_name="main")
        X, y = data.validate_features(X, y, cfg = config)
        return X, y
   


@step(enable_cache=False)
def load(X: pd.DataFrame, y: pd.DataFrame, version: str) -> Tuple[
    Annotated[pd.DataFrame, ArtifactConfig(name="features", tags=["data_preparation"])],
    Annotated[pd.DataFrame, ArtifactConfig(name="target", tags=["data_preparation"])],
]:
    data.save_features_target(X, y, version)

    return X, y


@pipeline()
def prepare_data_pipeline():
    df, version = extract()
    X, y = transform(df, version)
    X, y = validate(X, y)
    X, y = load(X, y, version)


if __name__ == "__main__":
    run = prepare_data_pipeline()
