import pandas as pd
import numpy as np
import os
import yaml
import zenml
from zenml.client import Client
from great_expectations import DataContext
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from zenml import save_artifact, load_artifact
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer

import sample_data

BASE_PATH = os.path.expandvars("$PROJECTPATH")

def read_datastore() -> pd.DataFrame:
    path = os.path.join(BASE_PATH, 'data/samples', 'sample.csv')
    df = pd.read_csv(path)
    return df



def extract_data() -> (pd.DataFrame, str):
    df = read_datastore()
    return df

def get_artifact(name, version):
    client = Client()
    l = client.list_artifact_versions(name = name, tag = version, sort_by="version").items
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    return latest_artifact.load()

def transform_data(df, cfg, version = None, return_df = False,  only_X = False, transformer_version = None,only_transform = False,):

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
        version = "v1"

    if(not only_X):
        df = df.dropna(subset=[target_column])

    X_cols = [col for col in df.columns if col not in target_column]
    X = df[X_cols]
    if(not only_X):
        y = df[target_column]

    X[day_of_week_column] = pd.to_datetime(df[datetime_column]).dt.dayofweek

    if(only_transform):
        if transformer_version is None:
            transformer_version = version
        X_model = get_artifact("X_transform_pipeline", version = transformer_version)
        if not only_X:
            y_model = get_artifact("y_transform_pipeline", version = transformer_version)

        X_preprocessed = X_model.transform(X)
        if not only_X:
            y_encoded = y_model.transform(y)
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
            n_jobs = 4
        )

        pipe = make_pipeline(preprocessor)

        X_model = pipe.fit(X)
        X_preprocessed = X_model.transform(X)

        if(not only_X):
            le = LabelEncoder()
            y_model = le.fit(y.values.ravel())
            y_encoded = y_model.transform(y.values.ravel())

        save_artifact(data = X_model, name="X_transform_pipeline", tags=[transformer_version], materializer=SklearnMaterializer)
        if(not only_X):
            save_artifact(data = y_model, name="y_transform_pipeline", tags=[transformer_version], materializer=SklearnMaterializer)

    cat_col_names = X_model.named_steps['columntransformer'].named_transformers_['categorical'].named_steps['onehot'].get_feature_names_out(categorical_features)
    num_col_names = numerical_features
    date_col_names = []
    for feature in cyclical_features.keys():
        date_col_names.extend([f'{feature}_sin', f'{feature}_cos'])

    all_col_names = np.concatenate([num_col_names, cat_col_names, date_col_names])

    X_final = pd.DataFrame(X_preprocessed, columns=all_col_names)
    if(not only_X):
        y_final = pd.DataFrame(y_encoded, columns=[target_column])

    X_final.columns = X_final.columns.astype(str)
    if(not only_X):
        y_final.columns = y_final.columns.astype(str)

    def add_missing_columns(dff, required_columns):
        for col in required_columns:
            if col not in dff.columns:
                dff[col] = 0.0
        return dff

# TODO to other place

    
    hardcoded = ['source_Back Bay', 'source_Beacon Hill', 'source_Boston University', 'source_Fenway', 'source_Financial District', 'source_Haymarket Square', 'source_North End', 'source_North Station', 'source_Northeastern University', 'source_South Station', 'source_Theatre District', 'source_West End', 'destination_Back Bay', 'destination_Beacon Hill', 'destination_Boston University', 'destination_Fenway', 'destination_Financial District', 'destination_Haymarket Square', 'destination_North End', 'destination_North Station', 'destination_Northeastern University', 'destination_South Station', 'destination_Theatre District', 'destination_West End', 'name_Black', 'name_Black SUV', 'name_Lux', 'name_Lux Black', 'name_Lux Black XL', 'name_Lyft', 'name_Lyft XL', 'name_Shared', 'name_Taxi', 'name_UberPool', 'name_UberX', 'name_UberXL', 'name_WAV', 'short_summary_ Clear ', 'short_summary_ Drizzle ', 'short_summary_ Foggy ', 'short_summary_ Light Rain ', 'short_summary_ Mostly Cloudy ', 'short_summary_ Overcast ', 'short_summary_ Partly Cloudy ', 'short_summary_ Possible Drizzle ', 'short_summary_ Rain ']
    
    X_final = add_missing_columns(X_final, hardcoded)
    right_order = ["apparentTemperature",
                    "cloudCover",
                    "day",
                    "day_cos",
                    "day_of_week",
                    "day_of_week_cos",
                    "day_of_week_sin",
                    "day_sin",
                    "destination_Back Bay",
                    "destination_Beacon Hill",
                    "destination_Boston University",
                    "destination_Fenway",
                    "destination_Financial District",
                    "destination_Haymarket Square",
                    "destination_North End",
                    "destination_North Station",
                    "destination_Northeastern University",
                    "destination_South Station",
                    "destination_Theatre District",
                    "destination_West End",
                    "distance",
                    "hour",
                    "hour_cos",
                    "hour_sin",
                    "humidity",
                    "month",
                    "month_cos",
                    "month_sin",
                    "name_Black",
                    "name_Black SUV",
                    "name_Lux",
                    "name_Lux Black",
                    "name_Lux Black XL",
                    "name_Lyft",
                    "name_Lyft XL",
                    "name_Shared",
                    "name_Taxi",
                    "name_UberPool",
                    "name_UberX",
                    "name_UberXL",
                    "name_WAV",
                    "precipIntensity",
                    "precipIntensityMax",
                    "precipProbability",
                    "pressure",
                    "short_summary_ Clear ",
                    "short_summary_ Drizzle ",
                    "short_summary_ Foggy ",
                    "short_summary_ Light Rain ",
                    "short_summary_ Mostly Cloudy ",
                    "short_summary_ Overcast ",
                    "short_summary_ Partly Cloudy ",
                    "short_summary_ Possible Drizzle ",
                    "short_summary_ Rain ",
                    "source_Back Bay",
                    "source_Beacon Hill",
                    "source_Boston University",
                    "source_Fenway",
                    "source_Financial District",
                    "source_Haymarket Square",
                    "source_North End",
                    "source_North Station",
                    "source_Northeastern University",
                    "source_South Station",
                    "source_Theatre District",
                    "source_West End",
                    "surge_multiplier",
                    "uvIndex",
                    "visibility",
                    "windBearing",
                    "windSpeed"]

    X_final = X_final[right_order]
   

    if return_df:
        df = pd.concat([X_final, y_final], axis=1)
        return df
    else:
        if(only_X):
            return X_final
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
    print("Version: "+version)


def extract_features(name, version, random_state, size = 1, return_df = False ):
    client = Client()
    l = client.list_artifact_versions(name = name, tag = version, sort_by="version").items
    latest_artifact = sorted(l, key=lambda x: x.created)[-1]
    df = latest_artifact.load()
    df = df.sample(frac = size, random_state = random_state)

    print("size of df is ", df.shape)
    print("df columns: ", df.columns)

    if(return_df):
        return df

    X = df.drop('price', axis=1)
    y = df.price

    print("shapes of X,y = ", X.shape, y.shape)

    return X, y

def load_artifact(name: str, version: str) -> pd.DataFrame:
    return zenml.load_artifact(name, version)

# if __name__=="__main__":
#     df, version = extract_data()
#     X, y = transform_data(df, version)
#     print(X.head())
#     print(X.info())
#     print(y)