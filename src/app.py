import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="../configs", config_name="main")
def app(cfg : DictConfig = None) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(cfg.db.password)
    print(cfg['db']['password'])

if __name__ == "__main__":
    app()