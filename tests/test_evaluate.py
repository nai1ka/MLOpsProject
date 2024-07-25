from unittest.mock import patch, MagicMock
from evaluate import evaluate, load_local_model

@patch('mlflow.sklearn.load_model')
@patch('evaluate.BASE_PATH', new_callable=lambda: "/mock/path")
def test_load_local_model(mock_base_path, mock_load_model):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    model_name = "test_model"
    result = load_local_model(model_name)
    
    mock_load_model.assert_called_once_with("/mock/path/models/test_model")
    assert result == mock_model

@patch('evaluate.extract_features')
@patch('evaluate.mlflow.sklearn.load_model')
def test_evaluate_with_provided_data(mock_load_model, mock_extract_features):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1, 2, 3]  # Mock predictions
    X_test = [1, 2, 3]
    y_test = [1, 2, 3]
    
    evaluate(evaluate_saved_model=False, model=mock_model, X_test=X_test, y_test=y_test)
    
    mock_load_model.assert_not_called()
    mock_model.predict.assert_called_once_with(X_test)
