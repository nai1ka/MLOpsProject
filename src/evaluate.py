import os
import hydra
import mlflow
from mlflow.tracking import MlflowClient

BASE_PATH = os.path.expandvars("$PROJECTPATH")

def load_local_model(name):
    return mlflow.sklearn.load_model(BASE_PATH+"/models/"+name)


def evaluate(sample_version, model_alias):
    print(load_local_model(BASE_PATH+"/models/"+model_alias))
    # TODO
   
    

@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg = None):
    evaluate(cfg.evaluate_sample_version, cfg.evaluate_model_alias)

if __name__ == "__main__":
    main()
