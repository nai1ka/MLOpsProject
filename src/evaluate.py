import os
import hydra
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from data import extract_features
from model import load_local_model

BASE_PATH = os.path.expandvars("$PROJECTPATH")


def evaluate(
    evaluate_saved_model=True,
    sample_version=None,
    model_alias=None,
    model=None,
    X_test=None,
    y_test=None,
):
    """
    Evaluate a model using various regression metrics.

    Parameters:
        evaluate_saved_model (bool): Whether to evaluate a saved model or a provided model.
        sample_version (str): Version of the sample to use for extracting features.
        model_alias (str): Alias of the model to load if evaluating a saved model.
        model (sklearn model): Model to evaluate if not evaluating a saved model.
        X_test (pd.DataFrame): Test features if not evaluating a saved model.
        y_test (pd.DataFrame): Test targets if not evaluating a saved model.
    """

    if evaluate_saved_model:
        # Extract features and target using the provided sample version
        X, y = extract_features(name="features_target", version=sample_version)
        model = load_local_model(model_alias)
    else:
        # Load the local model using the provided alias
        X, y = X_test, y_test

    # Predict using the model
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    print(f"MAPE: {mape}")


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    evaluate(
        sample_version=cfg.evaluate_sample_version, model_alias=cfg.evaluate_model_alias
    )


if __name__ == "__main__":
    main()
