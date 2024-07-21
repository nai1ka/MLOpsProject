import json
import requests
import hydra
from data import extract_features

@hydra.main(config_path="../configs", config_name="main", version_base=None) 
def predict(cfg = None):
    X, y = extract_features(name = "features_target", 
                        version = cfg.example_version, 
                        random_state=cfg.random_state)

    example = X.iloc[0,:]
    example_target = y[0]

    example = json.dumps( 
    { "inputs": example.to_dict() }
     )

    payload = example
    print(payload)

    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    print(response.json())
    print("Example target:", example_target)


if __name__=="__main__":
    predict()