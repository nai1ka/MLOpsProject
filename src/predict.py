import json
import random
import requests
import hydra

from data import extract_features

@hydra.main(config_path="../configs", config_name="main", version_base=None) 
def predict(cfg=None):
    """
    Function to make predictions using a deployed model.
    
    Parameters:
        cfg (Config): Configuration object provided by Hydra.
    """
    # Extract features and target from ZenML artifact store
    X, y = extract_features(name="features_target", 
                            version=cfg.example_version)

    # Set the random state for reproducibility
    random.seed(555)

    # Generate a random index
    random_index = random.randint(0, len(X) - 1)
    
    # Select the example and its corresponding target
    example = X.iloc[random_index, :]
    example_target = y[random_index]

    # Convert the example to JSON format
    example = json.dumps({"inputs": example.to_dict()})

    # Send a POST request with the example data
    response = requests.post(
        url=f"http://localhost:{cfg.port}/invocations",
        data=example,
        headers={"Content-Type": "application/json"},
    )

    # Print the prediction and the actual target value
    print("\nPrediction:", response.json()['predictions'])
    print("Target:", example_target)


if __name__ == "__main__":
    predict()
