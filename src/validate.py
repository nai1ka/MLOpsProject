# src/validate.py

from data import read_datastore
from data import transform_data
from evaluate import load_local_model
import giskard
import hydra
from sklearn.metrics import r2_score
from omegaconf import DictConfig, OmegaConf
import mlflow

from model import get_models_with_alias

hydra.core.global_hydra.GlobalHydra.instance().clear()

@hydra.main(config_path="../configs", config_name="main")
def validate(cfg : DictConfig):
    version  = cfg.test_data_version

    df = read_datastore()
    testdata_version = cfg.test_data_version

    TARGET_COLUMN = cfg.target_column

    CATEGORICAL_COLUMNS = list(cfg.categorical_columns)

    dataset_name = cfg.dataset_name

    df = df.dropna(subset=[TARGET_COLUMN])

    # Wrap your Pandas DataFrame with giskard.Dataset (validation or test set)
    giskard_dataset = giskard.Dataset(
        df=df,  # A pandas.DataFrame containing raw data (before pre-processing) and including ground truth variable.
        target=TARGET_COLUMN,  # Ground truth variable
        name=dataset_name, # Optional: Give a name to your dataset
        cat_columns=CATEGORICAL_COLUMNS  # List of categorical columns. Optional, but improves quality of results if available.
    )

    model_name = cfg.model.best_model_name

    # You can sweep over challenger aliases using Hydra
    model_alias = cfg.model.best_model_alias

    model: mlflow.pyfunc.PyFuncModel = load_local_model("challenger")


    # Add missing columns to the dataframe and fill them with zeros
    
    def predict(raw_df):
        X = transform_data(
                            df = raw_df, 
                            version = version, 
                            cfg = cfg, 
                            return_df = False,
                            only_X = True)

        return model.predict(X) 

   
    predictions = predict(df[df.columns].head())

    giskard_model = giskard.Model(
        model=predict,
        model_type = "regression", # regression
        feature_names = df.columns, # By default all columns of the passed dataframe
        name=model_name
    )

    scan_results = giskard.scan(giskard_model, giskard_dataset, raise_exceptions=True)

    # Save the results in `html` file
    scan_results_path = f"test_suite_{model_name}_{dataset_name}_{testdata_version}.html"
    scan_results.to_html(scan_results_path)

    suite_name = f"test_suite_{model_name}_{dataset_name}_{version}"
    test_suite = giskard.Suite(name = suite_name)

    # TODO: probably move r2_threshold in yaml
    test1 = giskard.testing.test_r2(model = giskard_model, 
                                    dataset = giskard_dataset,
                                    threshold=cfg.model.r2_threshold)

    test_suite.add_test(test1)

    test_results = test_suite.run()
    if (test_results.passed):
        print("Passed model validation!")
    else:
        print("Model has vulnerabilities!")

if __name__ == "__main__":
    validate()