import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import mlflow
import mlflow.sklearn
import importlib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import random
import numpy as np

def train(X_train, y_train, cfg):
    params = cfg.model.params
    module_name = cfg.model.module_name  # e.g. "sklearn.ensemble"
    class_name = cfg.model.class_name  # e.g. "RandomForestRegressor"
    print(class_name, params)

    class_instance = getattr(importlib.import_module(module_name), class_name)
    estimator = class_instance()

    # TODO Is random_state=cfg.random_state enough for model reprodusibility or we need lines from 14-16?
    cv = StratifiedKFold(n_splits=cfg.model.folds, random_state=cfg.random_state, shuffle=True)

    param_grid = dict(params)
    scoring = list(cfg.model.metrics.values())
    evaluation_metric = cfg.model.evaluation_metric

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=cfg.cv_n_jobs,
        refit=evaluation_metric, # maybe no need
        cv=cv, # cv=3
        verbose=1,
        return_train_score=True
    )

    gs.fit(X_train, y_train)
    return gs

def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    eval_data = pd.DataFrame({'label': y_test, 'predictions': predictions})

    results = mlflow.evaluate(
        data=eval_data,
        model_type="regressor",
        targets="label",
        predictions="predictions",
        evaluators=["default"]
    )

    return results.metrics

def log_metadata(cfg, gs, X_train, y_train, X_test, y_test):
    cv_results = pd.DataFrame(gs.cv_results_).filter(regex=r'std_|mean_|param_').sort_index(axis=1)
    best_metrics_values = [result[1][gs.best_index_] for result in gs.cv_results_.items()]
    best_metrics_keys = [metric for metric in gs.cv_results_]
    best_metrics_dict = {k: v for k, v in zip(best_metrics_keys, best_metrics_values) if 'mean' in k or 'std' in k}

    params = best_metrics_dict

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)

    experiment_name = cfg.model.model_name + "_" + cfg.experiment_name

    try:
        experiment_id = mlflow.create_experiment(name=experiment_name)
    except mlflow.exceptions.MlflowException:
        experiment_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    run_name = f"{cfg.run_name}_{cfg.model.model_name}_{cfg.model.evaluation_metric}_{params[cfg.model.cv_evaluation_metric]:.4f}".replace(".", "_")

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(best_metrics_dict)
        mlflow.set_tag(cfg.model.tag_key, cfg.model.tag_value)

        predictions = gs.best_estimator_.predict(X_test)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs. Predicted Values')
        plt.grid(True)
        plt.tight_layout()
        plt_path = "actual_vs_predicted_plot.png"
        plt.savefig(plt_path)
        mlflow.log_artifact(plt_path, artifact_path="plots")

        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.title("Learning curves")
        plt.legend(loc="best")
        plt.tight_layout()
        lc_path = "learning_curve.png"
        plt.savefig(lc_path)
        mlflow.log_artifact(lc_path, artifact_path="plots")

        signature = mlflow.models.infer_signature(X_train, gs.predict(X_train))

        model_info = mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path=cfg.model.artifact_path,
            signature=signature,
            input_example=X_train.iloc[0].to_numpy(),
            registered_model_name=cfg.model.model_name,
            pyfunc_predict_fn=cfg.model.pyfunc_predict_fn
        )

        client = mlflow.tracking.MlflowClient()
        client.set_model_version_tag(
            name=cfg.model.model_name,
            version=model_info.version,
            key="source",
            value="best_grid_search_model"
        )

        for index, result in cv_results.iterrows():
            child_run_name = f"child_{run_name}_{index}"
            with mlflow.start_run(run_name=child_run_name, experiment_id=experiment_id, nested=True):
                params = result.filter(regex='param_').to_dict()
                metrics = result.filter(regex='mean_').to_dict()
                stds = result.filter(regex='std_').to_dict()

                params = {k.replace("param_", ""): v for k, v in params.items()}

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(stds)

                class_instance = getattr(importlib.import_module(cfg.model.module_name), cfg.model.class_name)
                estimator = class_instance(**params)
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

        # Cleanup: Remove the local plot file
        # if os.path.exists(plt_path):
        #     os.remove(plt_path)

        dst_path = "results"
        artifact_uri = mlflow.get_artifact_uri(run_id=run.info.run_id)
        mlflow.artifacts.download_artifacts(artifact_uri, dst_path)
                
def retrieve_model_with_alias(model_name, model_alias="champion"):
    best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")
    return best_model

def retrieve_model_with_version(model_name, model_version="v1"):
    best_model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_version}")
    return best_model