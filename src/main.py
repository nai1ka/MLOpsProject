import hydra
from model import train, log_metadata
from data import extract_features
from sklearn.model_selection import train_test_split

def run(cfg):
    train_data_version = cfg.train_data_version

    X, y = extract_features(name = "features_target", version=train_data_version)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.random_state) 

    test_data_version = cfg.test_data_version

    X_test, y_test = extract_features(name = "features_target", version=test_data_version)

    gs = train(X_train, y_train, cfg=cfg)

    log_metadata(cfg, gs, X_train, y_train, X_test, y_test)

    
@hydra.main(config_path="../configs", config_name="main", version_base=None) 
def main(cfg=None):
    run(cfg)



if __name__=="__main__":
    main()
