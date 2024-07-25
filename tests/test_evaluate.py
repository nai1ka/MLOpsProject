import os
from unittest.mock import patch, MagicMock
from evaluate import evaluate, load_local_model


PROJECTPATH = os.environ['PROJECTPATH']

@patch('mlflow.sklearn.load_model')
@patch('evaluate.BASE_PATH', new_callable=lambda: PROJECTPATH)
def test_load_local_model(mock_base_path, mock_load_model):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    model_name = "test_model"
    result = load_local_model(model_name)
    
    mock_load_model.assert_called_once_with(PROJECTPATH+"/models/test_model")
    assert result == mock_model
