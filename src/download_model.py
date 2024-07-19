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

    print(model.metadata)





# # Download the champion model
# champion_model_uri = f"models:/{model_name}/{champion_version}"
# champion_local_path = os.path.join(local_models_path, f"champion_model_v{champion_version}")
# mlflow.artifacts.download_artifacts(artifact_uri=champion_model_uri, dst_path=champion_local_path)

# # Download the challenger model
# challenger_model_uri = f"models:/{model_name}/{challenger_version}"
# challenger_local_path = os.path.join(local_models_path, f"challenger_model_v{challenger_version}")
# mlflow.artifacts.download_artifacts(artifact_uri=challenger_model_uri, dst_path=challenger_local_path)

# print(f"Champion model downloaded to: {champion_local_path}")
# print(f"Challenger model downloaded to: {challenger_local_path}")

download_model()
