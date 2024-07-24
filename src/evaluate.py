import os
import hydra
import mlflow
from data import extract_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

BASE_PATH = os.path.expandvars("$PROJECTPATH")

def load_local_model(name):
    return mlflow.sklearn.load_model(BASE_PATH+"/models/"+name)


def evaluate(sample_version, sample_random_state, model_alias):
    X, y = extract_features(name = "features_target", version=sample_version, random_state=sample_random_state)
    model = load_local_model(model_alias)

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R^2 Score: {r2}")
    print(f'MAPE: {mape}')

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg = None):
    evaluate(
        sample_version=cfg.evaluate_sample_version,
        sample_random_state=cfg.random_state,
        model_alias=cfg.evaluate_model_alias
    )


if __name__ == "__main__":
    main()
