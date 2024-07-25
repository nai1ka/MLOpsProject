import os
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import importlib

BASE_PATH = os.path.expandvars("$PROJECTPATH")


def train(X_train, y_train, cfg):
    """
    Train a machine learning model using GridSearchCV.
    
    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target.
        cfg (Config): Configuration object provided by Hydra.

    Returns:
        GridSearchCV: Fitted GridSearchCV object.
    """

    # Define the model hyperparameters
    params = cfg.model.params

    # Train the model
    module_name = cfg.model.module_name
    class_name = cfg.model.class_name

    import importlib

    # Dynamically import the module and class
    class_instance = getattr(importlib.import_module(module_name), class_name)

    # Instantiate the estimator
    estimator = class_instance(**params)

    # Define cross-validation strategy
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=cfg.model.folds, random_state=cfg.random_state, shuffle=True)

    param_grid = dict(params)

    # Define the scoring metrics
    scoring = list(cfg.model.metrics.values())

    evaluation_metric = cfg.model.evaluation_metric

    # Initialize GridSearchCV
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric,
        cv=cv,
        verbose=1,
        return_train_score=True
    )

    # Fit the GridSearchCV
    gs.fit(X_train, y_train)

    return gs


def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    """
    Log metadata and artifacts to MLflow.
    
    Parameters:
        cfg (Config): Configuration object provided by Hydra.
        gs (GridSearchCV): Fitted GridSearchCV object.
        X_train (pd.DataFrame): Training features.
        y_train (pd.DataFrame): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test target.
    """

    # Extract cross-validation results
    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_] for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k: v for k, v in zip(best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    print(cv_results, cv_results.columns)

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        # Create a new MLflow Experiment
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException as e:
        # If experiment already exists, get its ID
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id  # type: ignore

    print("experiment-id : ", experiment_id)

    cv_evaluation_metric = cfg.model.cv_evaluation_metric
    run_name = "_".join([cfg.run_name, cfg.model.model_name, cfg.model.evaluation_metric,
                         str(params[cv_evaluation_metric]).replace(".", "_")])  # type: ignore
    print("run name: ", run_name)

    if (mlflow.active_run()):
        mlflow.end_run()

    # Fake run
    with mlflow.start_run():
        pass

    # Parent run
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        # Log training and testing datasets
        df_train_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_train,
                                                                  targets=cfg.target_column)  # type: ignore
        df_test_dataset = mlflow.data.pandas_dataset.from_pandas(df=df_test, targets=cfg.target_column)  # type: ignore
        mlflow.log_input(df_train_dataset, "training")
        mlflow.log_input(df_test_dataset, "testing")

        # Log the hyperparameters
        mlflow.log_params(gs.best_params_)

        # Log the performance metrics
        mlflow.log_metrics(best_metrics_dict)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        def save_plot(fig, filename, artifact_path="plots"):
            """
            Save a plot as an artifact in MLflow.
            
            Parameters:
                fig (matplotlib.figure.Figure): Matplotlib figure to save.
                filename (str): Filename for the saved plot.
                artifact_path (str): Path in MLflow to store the artifact.
            """
            plt_path = f"{filename}.png"
            fig.savefig(plt_path)
            mlflow.log_artifact(plt_path, artifact_path=artifact_path)

        # Infer the model signature
        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn
        )

        client = mlflow.client.MlflowClient()
        client.set_model_version_tag(name=cfg.model.model_name, version=model_info.registered_model_version,
                                     key="source", value="best_Grid_search_model")

        for index, result in cv_results.iterrows():
            child_run_name = "_".join(['child', run_name, str(index)])  # type: ignore
            with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id,
                                  nested=True) as child_run:  # , tags=best_metrics_dict):
                ps = result.filter(regex='param_').to_dict()
                ms = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                # Remove param_ from the beginning of the keys
                ps = {k.replace("param_", ""): v for (k, v) in ps.items()}

                mlflow.log_params(ps)
                mlflow.log_metrics(ms)
                mlflow.log_metrics(stds)

                # We will create the estimator at runtime
                module_name = cfg.model.module_name  # e.g. "sklearn.linear_model"
                class_name = cfg.model.class_name  # e.g. "LogisticRegression"

                # Load "module.submodule.MyClass"
                class_instance = getattr(importlib.import_module(module_name), class_name)

                estimator = class_instance(**ps)
                estimator.fit(X_train, y_train)

                signature = mlflow.models.infer_signature(X_train, estimator.predict(X_train))

                model_info = mlflow.sklearn.log_model(
                    sk_model=estimator,
                    artifact_path=cfg.model.artifact_path,
                    signature=signature,
                    input_example=X_train.iloc[0].to_numpy(),
                    registered_model_name=cfg.model.model_name,
                    pyfunc_predict_fn=cfg.model.pyfunc_predict_fn
                )

                model_uri = model_info.model_uri
                loaded_model = mlflow.sklearn.load_model(model_uri=model_uri)

                predictions = loaded_model.predict(X_test)
                residuals = y_test - predictions

                # Actual vs Predicted Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, predictions)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Actual vs Predicted Plot')
                ax.grid(True)
                plt.tight_layout()
                save_plot(fig, "actual_vs_predicted_plot")

                # Distribution of Residuals
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(residuals, kde=True, ax=ax)
                ax.set_title('Distribution of Residuals')
                ax.set_xlabel('Residuals')
                ax.grid(True)
                plt.tight_layout()
                save_plot(fig, "distribution_of_residuals")

                # Residuals vs. Fitted Values Plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(predictions, residuals)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Fitted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residuals vs Fitted Values')
                ax.grid(True)
                plt.tight_layout()
                save_plot(fig, "residuals_vs_fitted_plot")

                eval_data = pd.DataFrame(y_test)
                eval_data.columns = ["label"]
                eval_data["predictions"] = predictions

                results = mlflow.evaluate(
                    data=eval_data,
                    model_type="regressor",
                    targets="label",
                    predictions="predictions",
                    evaluators=["default"]
                )

                print(f"metrics:\n{results.metrics}")

                artifact_uri = mlflow.get_artifact_uri(artifact_path="plots")
                dst_path = f"{BASE_PATH}/results/{child_run.info.run_id}"
                mlflow.artifacts.download_artifacts(artifact_uri=artifact_uri, dst_path=dst_path)


def get_models_with_alias(model_name, model_alias, return_version=False):
    """
    Retrieve model from MLflow registry using model name and alias.
    
    Parameters:
        model_name (str): Name of the model in the MLflow registry.
        model_alias (str): Alias of the model version in the MLflow registry.
        return_version (bool): Whether to return the model version information. Default is False.

    Returns:
        Model object or tuple: Loaded model or tuple containing the model and its version info.
    """
    client = mlflow.MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")
    if (return_version):
        return model, client.get_model_version_by_alias(name=model_name, alias=model_alias)
    return model


def save_model(model, model_alias):
    """
    Save a model locally using MLflow.
    
    Parameters:
        model: Model to be saved.
        model_alias (str): Alias name for saving the model.
    """
    mlflow.sklearn.save_model(model, BASE_PATH + "/models/" + model_alias)


def download_model(model_name, model_alias):
    """
    Download a model from MLflow registry to local storage.
    
    Parameters:
        model_name (str): Name of the model in the MLflow registry.
        model_alias (str): Alias of the model version in the MLflow registry.
    """
    client = mlflow.MlflowClient()
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")
    client.download_artifacts(model.metadata.run_id, "basic_rf", "models")


def load_local_model(name):
    """
    Load a locally saved model using MLflow.
    
    Parameters:
        name (str): Name of the locally saved model.

    Returns:
        Model object: Loaded model.
    """
    return mlflow.sklearn.load_model(BASE_PATH + "/models/" + name)
