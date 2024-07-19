from mlflow.tracking import MlflowClient


def get_models_with_alias(alias):
    client = MlflowClient()

    # List all registered models
    registered_models = client.list_registered_models()
    
    # Set to keep track of model names with 'champion' alias
    champion_models = set()

    # Iterate through each registered model
    for model in registered_models:
        model_name = model.name
        
        # List all versions of the current model
        versions = client.list_model_versions(model_name)
        
        # Check each version for the 'champion' alias
        for version in versions:
            version_details = client.get_model_version(model_name, version.version)
            if alias in version_details.aliases:
                champion_models.add(model_name)
                break  # No need to check further versions for this model

    return champion_models

def evaluate(sample_version, model_alias):
    # Initialize the MLflow client
    print(get_models_with_alias("champion"))

evaluate(1,1)