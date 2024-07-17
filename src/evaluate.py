import hydra
from model import retrieve_model_with_alias, retrieve_model_with_version, load_features
from omegaconf import OmegaConf

@hydra.main(config_path="../configs", config_name="evaluate", version_base=None)
def main(cfg):
    model = retrieve_model_with_alias(model_name="random_forest")
    # model = retrieve_model_with_version(model_name="random_forest")
    
    test_data_version = cfg.test_data_version
    X_test, y_test = load_features(name="features_target", version=test_data_version)

    predictions = model.predict(X_test)

    eval_data = pd.DataFrame(y_test)
    eval_data.columns = ["label"]
    eval_data["predictions"] = predictions

    results = mlflow.evaluate(
        data=eval_data,
        model_type="regressor",
        targets="label",
        predictions="predictions",
        evaluators=["default"]
    )

    print(results)

if __name__ == "__main__":
    main()
