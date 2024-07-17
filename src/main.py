import hydra
from data import extract_features
from model import train, log_metadata, evaluate
from omegaconf import OmegaConf

def run(cfg):
    train_data_version = cfg.train_data_version
    X, y = extract_features(name="features_target", version=train_data_version)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.experiment.test_size, random_state=cfg.experiment.random_state)

    test_data_version = cfg.test_data_version
    X_test, y_test = extract_features(name="features_target", version=test_data_version)

    print(f"Shapes: X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_val={y_val.shape}, y_test={y_test.shape}")

    gs = train(X_train, y_train, cfg=cfg)
    sevaluation_metrics = evaluate(gs, X_val, y_val)
    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)

    test_evaluation_metrics = evaluate(gs, X_test, y_test)
    print(f"Test Evaluation Metrics: {test_evaluation_metrics}")

@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def main(cfg=None):
    run(cfg)

if __name__ == "__main__":
    main()
