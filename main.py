import hydra

from omegaconf import DictConfig

from src.train.train_main import train_main


@hydra.main(config_path=".", config_name="main_args")
def main(cfg: DictConfig):
    print(cfg)
    print(cfg.model.config_name)
    if cfg.mode == "train":
        train_main(cfg)


if __name__ == "__main__":
    main()
