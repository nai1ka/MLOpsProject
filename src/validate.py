# src/validate.py

import os

import pandas as pd
from data import extract_data, transform_data
from evaluate import load_local_model
import giskard
import hydra
from sklearn.metrics import r2_score
from omegaconf import DictConfig, OmegaConf
import mlflow

from model import get_models_with_alias

hydra.core.global_hydra.GlobalHydra.instance().clear()

BASE_PATH = os.path.expandvars("$PROJECTPATH")


@hydra.main(config_path="../configs", config_name="main")
def validate(cfg: DictConfig=None):
    test_version = cfg.test_data_version

    if(not "sample_url" in cfg):
        df, version = extract_data(cfg=cfg, version=test_version)
    else:
        # Download sample from URL (for CI/CD purposes)
        print("Downloading file from URL:")
        df = pd.read_csv(cfg.sample_url)
        version = test_version

    TARGET_COLUMN = cfg.target_column

    CATEGORICAL_COLUMNS = list(cfg.categorical_columns)

    dataset_name = cfg.dataset_name

    df = df.dropna(subset=[TARGET_COLUMN])

    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name,  # Optional: Give a name to your dataset
        cat_columns=CATEGORICAL_COLUMNS
        # List of categorical columns. Optional, but improves quality of results if available.

    )

    model_name = cfg.model.best_model_name

    # You can sweep over challenger aliases using Hydra
    model_alias = cfg.model.best_model_alias

    model_names = cfg.model.challenger_model_names
    model_aliases = cfg.model.challenger_model_aliases

    # TODO change name
    for model_name, model_alias in zip(model_names, model_aliases):
        model: mlflow.pyfunc.PyFuncModel = load_local_model(model_alias)

        # Add missing columns to the dataframe and fill them with zeros

        def predict(raw_df):
            X = transform_data(
                df=raw_df,
                version=version,
                cfg=cfg,
                return_df=False,
                only_X=True
            )
            return model.predict(X)

        predict(df[df.columns].head())

        giskard_model = giskard.Model(
            model=predict,
            model_type="regression",  # regression
            feature_names=df.columns,
            name=model_name
        )

        scan_results = giskard.scan(giskard_model, giskard_dataset, raise_exceptions=True)

        # Save the results in `html` file
        scan_results_path = BASE_PATH + f"/reports/test_suite_{model_name}_{dataset_name}_{test_version}.html"
        scan_results.to_html(scan_results_path)

        suite_name = f"test_suite_{model_name}_{dataset_name}_{version}"
        test_suite = giskard.Suite(name=suite_name)

        test1 = giskard.testing.test_r2(
            model=giskard_model,
            dataset=giskard_dataset,
            threshold=cfg.model.r2_threshold
        )

        test_suite.add_test(test1)

        test_results = test_suite.run()
        if (test_results.passed):
            print(f"Passed model validation for {model_name}!")
        else:
            print(f"Model {model_name} has vulnerabilities!")


if __name__ == "__main__":
    validate()
    #sample_url="https://drive.google.com/uc?export=download&id=1zMnmmt1vUn1k09TdpdlENv_MBRyWz-hx", sample_version='v5'