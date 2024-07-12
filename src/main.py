import hydra
from model import train, load_features, log_metadata
from omegaconf import OmegaConf


def run(cfg):
    train_data_version = cfg.train_data_version

    X_train, y_train = load_features(name = "features_target", version=1)

    #test_data_version = cfg.test_data_version

    X_test, y_test = load_features(name = "features_target", version=1)

    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)

    
@hydra.main(config_path="../configs", config_name="main", version_base=None) # type: ignore
def main(cfg=None):
    run(cfg)



if __name__=="__main__":
    main()
