import json
import pandas as pd
import numpy as np
import os
import yaml
import zenml
import dvc.api
from zenml.client import Client
from great_expectations import DataContext
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from zenml import save_artifact, load_artifact
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer

import sample_data

BASE_PATH = os.path.expandvars("$PROJECTPATH")


def extract_data(cfg=None, version=None) -> tuple[pd.DataFrame, str]:
    """
    Extract data from DVC remote store usign a specified version

    Args:
        cfg (dict, optional): Configuration settings. If None, it initializes the configuration.
        version (str, optional): Version tag for the data. If None, uses version from the configuration.

    Returns:
        tuple: DataFrame containing the data and the version used.
    """
    # Initialize configuration if not provided
    if cfg is None:
        initialize(version_base=None, config_path="../configs")
        cfg = compose(config_name="main")

    # Use provided version or get from configuration
    if version is None:
        version = cfg.sample_version

    data_path = cfg.data_path
    data_store = cfg.data_store

    # Get the URL in DVC remote storage
    path = dvc.api.get_url(
        rev=version,
        path=data_path,
        remote=data_store,
        repo=BASE_PATH
    )

    # Load the data into a DataFrame
    df = pd.read_csv(path)

    # Return the DataFrame and the version used
    return df, version


def check_and_impute_datetime(df, datetime_column, impute_value='1970-01-01 00:00:00'):
    impute_datetime = pd.to_datetime(impute_value)

    def is_valid_datetime(dt_str):
        try:
            pd.to_datetime(dt_str)
            return True
        except:
            return False

    df = df.copy()
    invalid_mask = ~df[datetime_column].apply(is_valid_datetime)
    df.loc[invalid_mask, datetime_column] = impute_datetime
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    return df


def transform_data(df, cfg, version=None, return_df=False, only_X=False, transformer_version=None,
                   only_transform=False, ):
    if cfg is None:
        initialize(version_base=None, config_path="../configs")
        cfg = compose(config_name="main")

    target_column = cfg.target_column
    day_of_week_column = cfg.day_of_week_column
    datetime_column = cfg.datetime_column
    categorical_features = list(cfg.categorical_columns)
    numerical_features = list(cfg.numerical_columns)
    date_features = list(cfg.date_columns)
    cyclical_features = {
        'hour': 24,
        'day_of_week': 7,
        'day': 31,
        'month': 12
    }

    if version is None:
        version = cfg.sample_version
    if (not only_X):
        df = df.dropna(subset=[target_column])
    df = check_and_impute_datetime(df, datetime_column)

    X_cols = [col for col in df.columns if col not in target_column]
    X = df[X_cols]
    if (not only_X):
        y = df[[target_column]]

    X[day_of_week_column] = pd.to_datetime(df[datetime_column]).dt.dayofweek

    if (only_transform):
        if transformer_version is None:
            transformer_version = version
        X_model = get_artifact("X_transform_pipeline", version=transformer_version)
        X_preprocessed = X_model.transform(X)
    else:

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        numerical_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler())
        ])

        def sin_transformer(period):
            return FunctionTransformer(lambda x: np.sin(x.astype(float) / period * 2 * np.pi))

        def cos_transformer(period):
            return FunctionTransformer(lambda x: np.cos(x.astype(float) / period * 2 * np.pi))

        dt_transformers = []
        for feature, period in cyclical_features.items():
            dt_transformers.append((f'{feature}_sin', sin_transformer(period), [feature]))
            dt_transformers.append((f'{feature}_cos', cos_transformer(period), [feature]))
        date_transformer = ColumnTransformer(transformers=dt_transformers)

        preprocessor = ColumnTransformer(
            transformers=[
                ('numerical', numerical_transformer, numerical_features),
                ('categorical', categorical_transformer, categorical_features),
                ('date', date_transformer, date_features)
            ],
            remainder="drop",
            n_jobs=4
        )

        pipe = make_pipeline(preprocessor)

        X_model = pipe.fit(X)
        X_preprocessed = X_model.transform(X)

        if transformer_version is None:
            transformer_version = version
        save_artifact(data=X_model, name="X_transform_pipeline", tags=[transformer_version],
                      materializer=SklearnMaterializer)

    cat_col_names = X_model.named_steps['columntransformer'].named_transformers_['categorical'].named_steps[
        'onehot'].get_feature_names_out(categorical_features)
    num_col_names = numerical_features
    date_col_names = []
    for feature in cyclical_features.keys():
        date_col_names.extend([f'{feature}_sin', f'{feature}_cos'])

    all_col_names = np.concatenate([num_col_names, cat_col_names, date_col_names])

    X_final = pd.DataFrame(X_preprocessed, columns=all_col_names)

    X_final.columns = X_final.columns.astype(str)

    def add_missing_columns(dff, required_columns):
        for col in required_columns:
            if col not in dff.columns:
                dff[col] = 0.0
        return dff

    with open(BASE_PATH + "/schema/schema.json", 'r') as file:
        column_names = json.load(file)
    X_final = add_missing_columns(X_final, column_names)

    X_final = X_final[column_names]

    if return_df:
        df = pd.concat([X_final, y], axis=1)
        return df
    else:
        if (only_X):
            return X_final
        return X_final, y


def validate_features(X: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    context = FileDataContext(context_root_dir="../services/gx")
    suite_name = "data_validation"
    features = ['hour', 'month', 'source', 'destination', 'name', 'distance',
                'surge_multiplier', 'latitude', 'longitude', 'apparentTemperature',
                'short_summary', 'precipIntensity', 'precipProbability', 'humidity',
                'windSpeed', 'visibility', 'pressure', 'windBearing', 'cloudCover',
                'uvIndex', 'precipIntensityMax', 'day_of_week']

    context.add_or_update_expectation_suite(suite_name)

    # Add a pandas datasource (if not already added)
    ds = context.sources.add_or_update_pandas(name="pandas_datasource")

    # Add a CSV asset for the new data
    sample = ds.add_csv_asset(
        name="sample_csv",
        filepath_or_buffer="data/samples/sample.csv"
    )
    data_asset = ds.add_dataframe_asset(name="sample_data")
    # Build a batch request

    batch_request = data_asset.build_batch_request(dataframe=X)

    # Get a validator
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    validator.expect_column_values_to_be_between("hour", min_value=0, max_value=24)
    validator.expect_column_values_to_be_between("month", min_value=1, max_value=12)

    validator.expect_column_values_to_be_between("precipIntensity", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("precipProbability", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("humidity", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("visibility", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("cloudCover", min_value=0, max_value=1)
    # TODO: check add day of week after transform
    # validator.expect_column_values_to_be_between("day_of_week", min_value=0, max_value=7)
    validator.expect_column_values_to_be_between("precipIntensityMax", min_value=0, max_value=1)

    positive_features = ["distance", "apparentTemperature", "pressure", "windSpeed", "visibility", "windBearing",
                         "uvIndex", "surge_multiplier"]
    for feature in positive_features:
        validator.expect_column_values_to_be_between(feature, min_value=0, max_value=None)

    not_categorical_columns = ['hour', 'month', 'distance',
                               'surge_multiplier', 'apparentTemperature', 'precipIntensity', 'precipProbability',
                               'humidity',
                               'windSpeed', 'visibility', 'pressure', 'windBearing', 'cloudCover',
                               'uvIndex', 'precipIntensityMax', 'day_of_week']
    encoded_features = [feature for feature in X.columns if feature not in not_categorical_columns]

    for categorical_features in encoded_features:
        validator.expect_column_values_to_be_between("precipIntensityMax", min_value=0, max_value=1)

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


def save_features_target(X: pd.DataFrame, y: pd.DataFrame, version: str):
    """
    Save features and target as a single artifact.

    Args:
        X (pd.DataFrame): Features DataFrame.
        y (pd.DataFrame): Target DataFrame.
        version (str): Version tag for the artifact.

    Returns:
        None
    """

    # Reset indices of X DataFrame to ensure they are in the default integer format
    X.reset_index(drop=True, inplace=True)
    # Reset indices of y DataFrame to ensure they are in the default integer format
    y.reset_index(drop=True, inplace=True)

    # Concatenate features (X) and target (y) into a single DataFrame
    df = pd.concat([X, y], axis=1)

    # Save the concatenated DataFrame as an artifact with the specified version tag
    zenml.save_artifact(data=df, name="features_target", tags=[version])

    print("Saved version:", version)


def get_artifact(name, version):
    """
    Retrieve the latest version of a specified artifact.

    Args:
        name (str): Name of the artifact.
        version (str): Version tag of the artifact.

    Returns:
        pd.DataFrame: The loaded artifact data.
    """
    client = Client()

    # List all versions of the artifact
    l = client.list_artifact_versions(name=name, tag=version, sort_by="version").items

    # Get the latest artifact based on creation date
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    # Load and return the latest artifact data
    return latest_artifact.load()


def extract_features(name, version, return_df=False):
    """
    Extract latest features and target from ZenML artifact store.

    Args:
        name (str): Name of the artifact.
        version (str): Version tag of the artifact.
        return_df (bool, optional): If True, returns the entire DataFrame. Default is False.

    Returns:
        tuple: Features (X) and target (y) DataFrames, or the entire DataFrame if return_df is True.
    """

    # Retrieve latest artifact with specified name and version
    df = get_artifact(name, version)

    print("size of df is ", df.shape)

    if (return_df):
        return df

    # Separate features and target
    X = df.drop('price', axis=1)
    y = df.price

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y


if __name__ == "__main__":
    df, version = extract_data()