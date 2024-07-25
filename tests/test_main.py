import numpy as np
from unittest.mock import patch, MagicMock
from main import run


@patch('main.log_metadata')
@patch('main.evaluate')
@patch('main.train')
@patch('main.extract_features')
def test_run(mock_extract_features, mock_train, mock_evaluate, mock_log_metadata):
    # Mocking extract_features to return sample data
    mock_extract_features.side_effect = [
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([10, 11, 12])),  # For training data
        (np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]]), np.array([22, 23, 24]))  # For test data
    ]

    # Mocking train to return a mock model
    mock_gs = MagicMock()
    mock_train.return_value = mock_gs

    # Mock configuration
    mock_cfg = MagicMock()
    mock_cfg.train_data_version = "v1"
    mock_cfg.test_size = 0.2
    mock_cfg.random_state = 42
    mock_cfg.test_data_version = "v2"

    run(mock_cfg)

    assert mock_extract_features.call_count == 2
    mock_train.assert_called_once()

    evaluate_args, evaluate_kwargs = mock_evaluate.call_args
    assert evaluate_kwargs['evaluate_saved_model'] == False
    assert evaluate_kwargs['model'] == mock_gs

    assert set(tuple(row) for row in evaluate_kwargs['X_test']) <= set(tuple(row) for row in [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert set(evaluate_kwargs['y_test']) <= set([10, 11, 12])

    mock_log_metadata.assert_called_once()
