import hydra
from sklearn.model_selection import train_test_split
from evaluate import evaluate
from model import train, log_metadata
from data import extract_features


def run(cfg):
    """
    Main function to run the training and evaluation pipeline.

    Parameters:
        cfg (Config): Configuration object provided by Hydra.
    """

    # Extract features and target for training data using the specified version (from config)
    train_data_version = cfg.train_data_version
    X, y = extract_features(name="features_target", version=train_data_version)

    print("Train dataset: ", X.shape, train_data_version)
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Extract features and target for test data using the specified version
    test_data_version = cfg.test_data_version
    X_test, y_test = extract_features(name="features_target", version=test_data_version)
    print("Test dataset: ", X_test.shape, test_data_version)

    # Train the model using the training data and provided configuration
    gs = train(X_train, y_train, cfg=cfg)

    # Evaluate the model using the validation data
    evaluate(evaluate_saved_model=False, model=gs, X_test=X_val, y_test=y_val)

    # Log metadata related to the training and evaluation
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)


@hydra.main(config_path="../configs", config_name="main", version_base=None)
def main(cfg=None):
    run(cfg)


if __name__ == "__main__":
    main()
