import os
import giskard
import hydra
import mlflow
import pandas as pd
from data import extract_data, transform_data
from model import load_local_model, read_model_meta
from omegaconf import DictConfig
from sklearn.metrics import mean_absolute_percentage_error

BASE_PATH = os.path.expandvars("$PROJECTPATH")


@giskard.test(name="MAPE score", tags=["quality", "custom"])
def test_mape(model: giskard.models.base.BaseModel,
              dataset: giskard.datasets.Dataset,
              threshold: float):
    y_true = dataset.df[dataset.target]
    y_pred = model.predict(dataset).raw_prediction

    # Compute MAPE
    mape = mean_absolute_percentage_error(y_true, y_pred)

    passed = mape < threshold

    return giskard.TestResult(passed=passed, metric=mape)


def core_validate(cfg: DictConfig = None):
    """
    Validate models using the provided configuration.
    
    Args:
        cfg (DictConfig): Configuration object from Hydra.
    """
    test_version = cfg.test_data_version

    if (not "sample_url" in cfg):
        df, version = extract_data(cfg=cfg, version=test_version)
    else:
        # Download sample from URL (for CI/CD purposes)
        print("Downloading file from URL:")
        df = pd.read_csv(cfg.sample_url)
        version = test_version

    TARGET_COLUMN = cfg.target_column

    CATEGORICAL_COLUMNS = list(cfg.categorical_columns)

    dataset_name = cfg.dataset_name

    # Drop rows where the target column has missing values
    df = df.dropna(subset=[TARGET_COLUMN])

    # Wrap the pandas DataFrame with giskard.Dataset for validation or test set
    giskard_dataset = giskard.Dataset(
        df=df,
        target=TARGET_COLUMN,
        name=dataset_name,
        cat_columns=CATEGORICAL_COLUMNS
    )

    model_name = cfg.model.best_model_name
    model_alias = cfg.model.best_model_alias
    model_aliases = cfg.model.challenger_model_aliases

    for model_alias in model_aliases:
        # Load the local model and metadata (from models folder) using its alias
        model_name, model_version = read_model_meta(model_alias)
        model: mlflow.pyfunc.PyFuncModel = load_local_model(model_alias)

        # Add missing columns to the dataframe and fill them with zeros

        def predict(raw_df):
            # Transform data before prediction
            X = transform_data(
                df=raw_df,
                version=version,
                cfg=cfg,
                return_df=False,
                only_X=True
            )
            return model.predict(X)

        # Test the prediction function with a sample
        predict(df[df.columns].head())

        # Wrap the prediction function with giskard.Model
        giskard_model = giskard.Model(
            model=predict,
            model_type="regression",
            feature_names=df.columns,
            name=model_name
        )

        # Scan the model with the dataset using giskard
        scan_results = giskard.scan(giskard_model, giskard_dataset, raise_exceptions=True)

        # Save the results in `html` file
        scan_results_path = f"{BASE_PATH}/reports/test_suite_{model_name}_{model_version}_{dataset_name}_{test_version}.html"
        scan_results.to_html(scan_results_path)

        suite_name = f"test_suite_{model_name}_{dataset_name}_{version}"
        test_suite = giskard.Suite(name=suite_name)

        # Define an R2 score test
        test1 = giskard.testing.test_r2(
            model=giskard_model,
            dataset=giskard_dataset,
            threshold=cfg.model.r2_threshold
        )

        test2 = test_mape(model=giskard_model, dataset=giskard_dataset, threshold=cfg.model.mape_threshold)

        test_suite.add_test(test1)
        test_suite.add_test(test2)

        test_results = test_suite.run()
        if (test_results.passed):
            print(f"Passed model validation for {model_name}!")
        else:
            print(f"Model {model_name} has vulnerabilities!")


@hydra.main(config_path="../configs", config_name="main")
def validate(cfg: DictConfig = None):
    core_validate(cfg)


if __name__ == "__main__":
    validate()
