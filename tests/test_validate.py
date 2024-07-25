from unittest.mock import patch, MagicMock
import pandas as pd
from omegaconf import OmegaConf
from src.validate import core_validate

CONFIG = OmegaConf.create(
    {
        "test_data_version": "v1",
        "sample_url": "https://example.com/sample.csv",
        "target_column": "target",
        "categorical_columns": ["cat1", "cat2"],
        "dataset_name": "test_dataset",
        "model": {
            "best_model_name": "best_model",
            "best_model_alias": "best_alias",
            "model_aliases_to_validate": ["alias1", "alias2"],
            "r2_threshold": 0.8,
            "mape_threshold": 0.2,
        },
    }
)


@patch("src.validate.extract_data")
@patch("src.validate.transform_data")
@patch("src.validate.load_local_model")
@patch("src.validate.giskard.Dataset")
@patch("src.validate.giskard.Model")
@patch("src.validate.giskard.scan")
@patch("src.validate.giskard.Suite")
@patch("src.validate.giskard.testing.test_r2")
@patch("src.validate.mlflow.pyfunc.PyFuncModel")
@patch("builtins.print")
@patch("src.validate.pd.read_csv")
@patch("src.validate.os.path.expandvars")
@patch("src.validate.read_model_meta")
def test_core_validate(
    mock_read_model_meta,
    mock_expandvars,
    mock_read_csv,
    mock_print,
    mock_pyfunc_model,
    mock_test_r2,
    mock_suite,
    mock_scan,
    mock_model,
    mock_dataset,
    mock_load_local_model,
    mock_transform_data,
    mock_extract_data,
):
    # Mocking the environment variables and data
    mock_expandvars.return_value = "/mock/path"
    mock_read_csv.return_value = pd.DataFrame(
        {"target": [1, 2, 3], "cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]}
    )

    # Mocking external function returns
    mock_extract_data.return_value = (
        pd.DataFrame(
            {"target": [1, 2, 3], "cat1": ["A", "B", "C"], "cat2": ["X", "Y", "Z"]}
        ),
        "v1",
    )
    mock_transform_data.return_value = pd.DataFrame({"cat1": ["A"], "cat2": ["X"]})
    mock_load_local_model.return_value = MagicMock()
    mock_model.return_value = MagicMock()
    mock_scan.return_value = MagicMock()
    mock_suite.return_value = MagicMock()
    mock_test_r2.return_value = MagicMock()

    # Mocking read_model_meta to return expected values
    mock_read_model_meta.return_value = ("test_model", "1.0")

    core_validate(cfg=CONFIG)

    mock_read_csv.assert_called()
    mock_transform_data.assert_called()
    mock_load_local_model.assert_called()
    mock_dataset.assert_called()
    mock_scan.assert_called()
    mock_suite.assert_called()
    mock_test_r2.assert_called()
    mock_print.assert_any_call("Passed model validation for test_model!")
