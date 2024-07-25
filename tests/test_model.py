import os
from unittest.mock import patch, mock_open, MagicMock
from model import read_model_meta, load_local_model


PROJECTPATH = os.environ.get('PROJECTPATH', '/default/path')


@patch('mlflow.sklearn.load_model')
@patch('evaluate.BASE_PATH', new_callable=lambda: PROJECTPATH)
def test_load_local_model(mock_base_path, mock_load_model):
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    model_name = "test_model"
    result = load_local_model(model_name)
    
    mock_load_model.assert_called_once_with(PROJECTPATH+"/models/test_model")
    assert result == mock_model


@patch('model.os.path.join')
@patch('model.open', new_callable=mock_open, read_data="model_name: test_model\nmodel_version: 1.0")
def test_read_model_meta(mock_open, mock_path_join):
    model_alias = "test_model"
    expected_name = "test_model"
    expected_version = "1.0"

    mock_path_join.return_value = os.path.join(PROJECTPATH, "models", model_alias, "registered_model_meta")
    
    model_name, model_version = read_model_meta(model_alias)
    
    mock_open.assert_called_once_with(os.path.join(PROJECTPATH, "models", model_alias, "registered_model_meta"), 'r')
    
    assert model_name == expected_name
    assert model_version == expected_version
