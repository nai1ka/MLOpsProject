import pandas as pd
import os
import yaml
import zenml
from great_expectations import DataContext
from great_expectations.checkpoint import Checkpoint
from great_expectations.data_context import FileDataContext
from hydra import compose, initialize
from omegaconf import OmegaConf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
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
    # Remove visibility1 column as a duplicate.
    df = df.drop(['visibility.1'], axis = 1)

    # Missing values in the target variable are randomly distributed, imputing them might introduce noise or bias. Therefore, we remove missing values in the target variable.
    df = df.dropna(subset=['price'])

    # Price of the ride is not affected by id, because it just uniquely identifies the rides. Therefore it can be deleted.
    df = df.drop(['id'], axis = 1)
    
    # Icon is just an image representation of the weather, so it can be deleted (the weather itself will be counted in other features).
    df = df.drop(['icon'], axis = 1)

    # Timezone feature has only 1 value as it should be, because the goal of the project is to only predict cab prices in Boston, so this column is also useless.
    df = df.drop(['timezone'], axis = 1)

    # Instead of using datetime and timestamp, we will use month, day_of_week, and hour.
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df = df.drop(['datetime', 'day', 'timestamp'], axis = 1)

    # All the temperature features are very highly correlated, so we will leave apparentTemperature because it has an average correlation with all other temperature features and often has a more direct impact on human comfort and behavior, which can influence cab prices.
    df = df.drop(['temperature', 'temperatureHigh', 'temperatureLow', 'apparentTemperatureHigh', 'apparentTemperatureLow', 'temperatureMin', 'temperatureMax', 'apparentTemperatureMin', 'apparentTemperatureMax'], axis = 1)
    
    # Since we are solving the problem for our own cab company, we do not need the cab_type column, which includes df about other cab companies.
    df = df.drop(['cab_type'], axis = 1)

    # Short summary column is brief version of long summary one. Therefore it is a duplicate and we should remove it.
    df = df.drop(['long_summary'], axis = 1)

    # Since the time of all measurements is very close to the time of travel, we can delete the features related to time of measurements.
    df = df.drop(['windGustTime', 'temperatureHighTime', 'temperatureLowTime', 'apparentTemperatureHighTime', 'apparentTemperatureLowTime', 'sunriseTime', 'sunsetTime', 'uvIndexTime', 'temperatureMinTime', 'temperatureMaxTime', 'apparentTemperatureMinTime', 'apparentTemperatureMaxTime'], axis = 1)

    # The price of the ride is not influenced by the moon phase, therefore it is reasonable to delete it.
    df = df.drop(['moonPhase'], axis = 1)

    # The ozone level also is not relevant feature for the price prediction.
    df = df.drop(['ozone'], axis = 1)

    # The dewPoint and windGust are not relevant feature for the price prediction.
    df = df.drop(['dewPoint', 'windGust'], axis = 1)

    # The product id represent the same information as name column: category of the auto (product). Therefore, we delete product_id, because name column has human-friendly names.
    df = df.drop(['product_id'], axis = 1)

    X = df.drop(columns=['price'])
    y = pd.DataFrame(df['price'])

    # Encode categorical features using onehot encoder.
    def encode_features_one_hot(dataframe, features_names, encoder):
        new_features = encoder.transform(dataframe[features_names])
        new_columns_df = pd.DataFrame(new_features, columns=encoder.get_feature_names_out(features_names))
        new_dataframe = pd.concat([dataframe.reset_index(drop=True), new_columns_df.reset_index(drop=True)], axis=1)
        new_dataframe.drop(features_names, axis=1, inplace=True)
        return new_dataframe

    features_names_to_encode = list(X.select_dtypes(exclude='number').columns)
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(X[features_names_to_encode])
    X = encode_features_one_hot(X, features_names_to_encode, encoder)

    # Scaling data using MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X), columns = X.columns)

    return X, y
    

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

def load_artifact(name: str, version: str) -> pd.DataFrame:
    return zenml.load_artifact(name, version)