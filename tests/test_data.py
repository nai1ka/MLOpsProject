import pandas as pd
from unittest.mock import patch, MagicMock
import os


from src.data import extract_data, check_and_impute_datetime, transform_data, validate_features, save_features_target, get_artifact, extract_features


@patch('src.data.dvc.api.get_url')
@patch('src.data.pd.read_csv')
@patch('src.data.initialize')
@patch('src.data.compose')
def test_extract_data(mock_compose, mock_initialize, mock_read_csv, mock_get_url):
    mock_initialize.return_value = None
    mock_compose.return_value = MagicMock(sample_version='v1', data_path='data/path', data_store='remote_store')
    mock_get_url.return_value = 'mock_url'
    mock_read_csv.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

    df, version = extract_data()

    mock_get_url.assert_called_once_with(rev='v1', path='data/path', remote='remote_store', repo=os.path.expandvars("$PROJECTPATH"))
    mock_read_csv.assert_called_once_with('mock_url')
    assert df.shape == (2, 2)
    assert version == 'v1'


def test_check_and_impute_datetime():
    df = pd.DataFrame({'datetime_col': ['2021-01-01', 'invalid_date', '2021-01-03']})
    df_imputed = check_and_impute_datetime(df, 'datetime_col')

    assert df_imputed['datetime_col'].iloc[1] == pd.to_datetime('1970-01-01 00:00:00')
    assert pd.to_datetime(df_imputed['datetime_col']).isna().sum() == 0


def test_transform_data(config):
    df = pd.read_csv(config.data_path)
    print(df)
    X_transformed = transform_data(df, config, only_X=True)

    assert X_transformed.shape[1] > df.shape[1]  # Check that the number of columns has increased due to transformations


@patch('src.data.FileDataContext')
@patch('src.data.get_artifact')
def test_validate_features(mock_get_artifact, MockFileDataContext):
    mock_context = MockFileDataContext.return_value
    mock_context.get_validator.return_value = MagicMock()
    mock_context.add_or_update_expectation_suite.return_value = None
    mock_get_artifact.return_value = pd.DataFrame()

    df = pd.DataFrame({'hour': [1], 'month': [1], 'precipIntensity': [0.5]})
    X_validated, y = validate_features(df, pd.DataFrame())

    assert X_validated.equals(df)
    mock_context.add_or_update_expectation_suite.assert_called_once()
    mock_context.get_validator.return_value.expect_column_values_to_be_between.assert_any_call('hour', min_value=0, max_value=24)


@patch('src.data.zenml.save_artifact')
def test_save_features_target(mock_save_artifact):
    X = pd.DataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    y = pd.DataFrame({'target': [10, 20]})

    save_features_target(X, y, 'v1')

    mock_save_artifact.assert_called_once()


@patch('src.data.Client')
def test_get_artifact(mock_Client):
    mock_client = mock_Client.return_value
    mock_client.list_artifact_versions.return_value.items = [MagicMock(created='2024-07-23T12:00:00', load=lambda: pd.DataFrame({'data': [1]}))]

    df = get_artifact('artifact_name', 'v1')

    assert df.shape == (1, 1)


@patch('src.data.get_artifact')
def test_extract_features(mock_get_artifact):
    df = pd.DataFrame({'feature1': [1, 2], 'price': [10, 20]})
    mock_get_artifact.return_value = df

    X, y = extract_features('artifact_name', 'v1')

    assert X.shape == (2, 1)
    assert y.shape == (2,)
