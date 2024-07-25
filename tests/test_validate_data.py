from unittest.mock import patch, MagicMock
from validate_data import validate_data


@patch('validate_data.FileDataContext')
@patch('validate_data.DictConfig')
@patch('builtins.print')
def test_validate_data(mock_print, mock_dict_config, mock_file_data_context, config):
    # Create a mock context
    mock_context = MagicMock()
    mock_file_data_context.return_value = mock_context

    # Create a mock datasource
    mock_datasource = MagicMock()
    mock_context.sources.add_or_update_pandas.return_value = mock_datasource

    # Create a mock CSV asset
    mock_csv_asset = MagicMock()
    mock_datasource.add_csv_asset.return_value = mock_csv_asset

    # Create a mock batch request
    mock_batch_request = MagicMock()
    mock_csv_asset.build_batch_request.return_value = mock_batch_request

    # Create a mock validator
    mock_validator = MagicMock()
    mock_context.get_validator.return_value = mock_validator

    # Set up the mock validation result
    mock_validation_results = MagicMock()
    mock_validation_results.success = True
    mock_validator.validate.return_value = mock_validation_results

    validate_data(config)

    # Check if print was called with the success message
    mock_print.assert_called_with("All data validations passed.")

    mock_file_data_context.assert_called_once_with(context_root_dir="../services/gx")
    mock_context.sources.add_or_update_pandas.assert_called_once_with(name="pandas_datasource")
    mock_datasource.add_csv_asset.assert_called_once_with(name="sample_csv", filepath_or_buffer=f"../{config.data_path}")
    mock_csv_asset.build_batch_request.assert_called_once()
    mock_context.get_validator.assert_called_once_with(batch_request=mock_batch_request, expectation_suite_name="initial_data_validation")
    mock_validator.validate.assert_called_once()
