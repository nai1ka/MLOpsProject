import mlflow
from mlflow.tracking import MlflowClient
import os

BASE_PATH = os.path.expandvars("$PROJECTPATH")

def download_model():
    client = MlflowClient()

    # Specify the registered model name
    model_name = "random_forest"
    model_alias = "champion"

    model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{model_alias}")
    client.download_artifacts(model.metadata.run_id, "basic_rf", "models")


download_model()
