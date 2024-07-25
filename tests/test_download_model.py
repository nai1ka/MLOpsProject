from unittest.mock import patch, MagicMock
from src.download_model import download_model

@patch('src.download_model.MlflowClient')
@patch('src.download_model.mlflow.pyfunc.load_model')
@patch('src.download_model.os.path.expandvars')
def test_download_model(mock_expandvars, mock_load_model, mock_MlflowClient):
    mock_expandvars.return_value = "/mock/path"
    mock_client = MagicMock()
    mock_MlflowClient.return_value = mock_client
    mock_model = MagicMock()
    mock_model.metadata.run_id = "mock_run_id"
    mock_load_model.return_value = mock_model

    download_model()

    mock_load_model.assert_called_once_with(model_uri="models:/random_forest@champion")
    mock_MlflowClient.assert_called_once()
    mock_client.download_artifacts.assert_called_once_with("mock_run_id", "basic_rf", "models")
