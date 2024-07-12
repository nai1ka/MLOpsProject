from utils import init_hydra
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import calendar
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.impute import SimpleImputer
import dvc.api
from zenml import save_artifact
from zenml.integrations.sklearn.materializers.sklearn_materializer import SklearnMaterializer

BASE_PATH = os.path.expandvars("$PROJECTPATH")

# initialize Hydra using Compose API
cfg = init_hydra()

def extract_data(base_path = BASE_PATH):
    """
    Extracts data from dvc remote data store

    :param str base_path: The project path.
    """

    version = cfg.data_version

    data_path = cfg.data.sample_path

    data_store = cfg.data.data_store

    # Get relative url to retrieve the data sample
    url = dvc.api.get_url(
        rev = version,
        path = data_path,
        remote= data_store,
        repo=BASE_PATH
    )

    # Get absolute path to the file
    url = base_path + url

    df = pd.read_csv(url)

    return df, version


def convert_month_abr(month):
    """
    A converter to transform the month abbreviations into numbers, e.g. `jan` to 0.

    :param str month: The month abbreviation
    """
    if month is None:
        return month

    months = {month.lower(): index-1 for index, month in enumerate(calendar.month_abbr) if month}

    return month.map(lambda m : months[m.lower()])


def convert_data(X):
    """
    Converts and fixes specific data features

    :param pd.DataFrame X: the input dataframe
    """
    
    # Get all cols of type month
    month_cols = list(cfg.data.dt_cols['month'])

    # convert it
    X[month_cols] = X[month_cols].apply(convert_month_abr)
    
    return X

def transform_data(df, version, return_df = False):
    """
    Transform the raw data into features

    :param pd.DataFrame df: the input raw dataframe
    :param str version: the version of the data sample
    :param bool return_df: True if the returned value is a concatenated dataframe, False if the returned value is input dataframe X and target column y.
    """

    # Define labels and features
    labels = cfg.data.target_col_name
    
    X_cols = [col for col in df.columns if col not in labels]
    X = df[X_cols]
    y = df[labels]

    # Define the category of features
    categorical_features = list(cfg.data.cat_cols)
    binary_features = list(cfg.data.bin_cols)
    numerical_features = list(cfg.data.num_cols)
    dt_features = list(sum(cfg.data.dt_cols.values(), []))

    # Convert and fix some specific data features
    X = convert_data(X)

    # Define the preprocessing transformers
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    binary_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Define the cyclical feature transformers
    def sin_transformer(period):
        return FunctionTransformer(lambda x: np.sin(x.astype(float) / period * 2 * np.pi))

    def cos_transformer(period):
        return FunctionTransformer(lambda x: np.cos(x.astype(float) / period * 2 * np.pi))


    dt_transformer = ColumnTransformer(transformers=[
        ('day_sin', sin_transformer(31), list(cfg.data.dt_cols['day'])),
        ('day_cos', cos_transformer(31), list(cfg.data.dt_cols['day'])),
        ('month_sin', sin_transformer(12),  list(cfg.data.dt_cols['month'])),
        ('month_cos', cos_transformer(12),  list(cfg.data.dt_cols['month']))
    ])
    
    # Combine the preprocessing transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('bin', binary_transformer, binary_features),
            ('dt', dt_transformer, dt_features)
        ],
        remainder="drop", # Drop all other features which did not pass through any feature transformer
        n_jobs = 4 # parallelism
    )

    print(numerical_features, categorical_features, binary_features, dt_features, labels)

    pipe = make_pipeline(preprocessor)


    # This will draw a diagram if you run it in a Jupyter notebook.
    from sklearn import set_config
    set_config(display="diagram")
    print(pipe)
    
    # Fit input data X
    X_model = pipe.fit(X)

    # Transform input data X
    X_preprocessed = X_model.transform(X)

    # Define the label encoder for the target variable
    le = LabelEncoder() # This encoder cannot be used in a column transformer
    y_model = le.fit(y.values.ravel())
    y_encoded = y_model.transform(y.values.ravel())

    save_artifact(data = X_model, name = "X_transform_pipeline", tags=[version], materializer=SklearnMaterializer)
    save_artifact(data = y_model, name = "y_transform_pipeline", tags=[version], materializer=SklearnMaterializer)

    # X_preprocessed
    X = pd.DataFrame(X_preprocessed)
    y = pd.DataFrame(y_encoded) # type: ignore

    # Do not forget to make columns of string type
    X.columns = X.columns.astype(str)
    y.columns = y.columns.astype(str)

    if return_df:
        df = pd.concat([X, y], axis = 1)
        return df
    else:
        return X, y


if __name__=="__main__":
    df, version = extract_data()
    df = transform_data(df, version, return_df = True)
    print(df.head()) # type: ignore