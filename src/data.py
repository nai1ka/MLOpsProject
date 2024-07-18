import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import yaml
import zenml
from great_expectations import DataContext
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from zenml import save_artifact
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer
from zenml.client import Client

import sample_data

BASE_PATH = os.path.expandvars("$PROJECTPATH")

def read_datastore() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, 'data/samples', 'sample.csv')
    df = pd.read_csv(path)
    return df

def get_data_version(config_path="../configs", config_name="main") -> str:
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
        return str(cfg.sample_version)

def extract_data() -> (pd.DataFrame, str):
    df = read_datastore()
    version = get_data_version()
    return df, version

def transform_data(df, version, return_df = False, config_path="../configs", config_name="data"):
    with initialize(config_path="../configs", version_base=None):
        cfg = compose(config_name="data")
        target_column = cfg.target_column
        day_of_week_column = cfg.day_of_week_column
        datetime_column = cfg.datetime_column
        categorical_features = list(cfg.categorical_columns)
        numerical_features = list(cfg.numerical_columns)
        date_features = list(cfg.date_columns)

    df = df.dropna(subset=[target_column])

    X_cols = [col for col in df.columns if col not in target_column]
    X = df[X_cols]
    y = df[target_column]

    X[day_of_week_column] = pd.to_datetime(df[datetime_column]).dt.dayofweek

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

    cyclical_features = {
        'hour': 24,
        'day_of_week': 7,
        'day': 31,
        'month': 12
    }
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
        n_jobs = 4
    )

    pipe = make_pipeline(preprocessor)

    X_model = pipe.fit(X)
    X_preprocessed = X_model.transform(X)

    le = LabelEncoder()
    y_model = le.fit(y.values.ravel())
    y_encoded = y_model.transform(y.values.ravel())

    save_artifact(data = X_model, name="X_transform_pipeline", tags=[version], materializer=SklearnMaterializer)
    save_artifact(data = y_model, name="y_transform_pipeline", tags=[version], materializer=SklearnMaterializer)

    cat_col_names = X_model.named_steps['columntransformer'].named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(categorical_features)
    num_col_names = numerical_features
    date_col_names = []
    for feature in cyclical_features.keys():
        date_col_names.extend([f'{feature}_sin', f'{feature}_cos'])

    all_col_names = np.concatenate([num_col_names, cat_col_names, date_col_names])

    X_final = pd.DataFrame(X_preprocessed, columns=all_col_names)
    y_final = pd.DataFrame(y_encoded, columns=[target_column])

    X_final.columns = X_final.columns.astype(str)
    y_final.columns = y_final.columns.astype(str)

    if return_df:
        df = pd.concat([X_final, y_final], axis=1)
        return df
    else:
        return X_final, y_final

def validate_features(X: pd.DataFrame, y: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    context = FileDataContext(context_root_dir="../services/gx")
    suite_name = "data_validation"
    features = ['hour','month', 'source', 'destination', 'name', 'distance',
       'surge_multiplier', 'latitude', 'longitude', 'apparentTemperature',
       'short_summary', 'precipIntensity', 'precipProbability', 'humidity',
       'windSpeed', 'visibility', 'pressure', 'windBearing', 'cloudCover',
       'uvIndex', 'precipIntensityMax', 'day_of_week']
    
    

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
    validator.expect_column_values_to_be_between("month", min_value=1, max_value=12)

    validator.expect_column_values_to_be_between("precipIntensity", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("precipProbability", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("humidity", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("visibility", min_value=0, max_value=1)
    validator.expect_column_values_to_be_between("cloudCover", min_value=0, max_value=1)
    # TODO: check add day of week after transform
    # validator.expect_column_values_to_be_between("day_of_week", min_value=0, max_value=7)
    validator.expect_column_values_to_be_between("precipIntensityMax", min_value=0, max_value=1)

    positive_features = ["distance", "apparentTemperature", "pressure", "windSpeed", "visibility", "windBearing", "uvIndex", "surge_multiplier"]
    for feature in positive_features:
        print(feature)
        validator.expect_column_values_to_be_between(feature, min_value=0, max_value=None)
    
    not_categorical_columns = ['hour','month', 'distance',
       'surge_multiplier', 'apparentTemperature', 'precipIntensity', 'precipProbability', 'humidity',
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

def load_features(X: pd.DataFrame, y: pd.DataFrame, version: str) -> None:
    # Example: Save the processed features
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    df = pd.concat([X, y], axis=1)
    zenml.save_artifact(data=df, name="features_target", tags=[version])

def extract_features(name, version, size=1):
    client = Client()
    l = client.list_artifact_versions(name=name, tag=version, sort_by="version").items
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    df = latest_artifact.load()
    df = df.sample(frac=size, random_state=88)

    X = df.drop('price', axis=1)
    y = df.price

    return X, y

if __name__=="__main__":
    df, version = extract_data()
    X, y = transform_data(df, version)
    print(X.head())
    print(X.info())
    print(y)