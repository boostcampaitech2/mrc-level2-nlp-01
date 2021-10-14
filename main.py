import hydra

from omegaconf import DictConfig


@hydra.main(config_path=".", config_name="main_args")
def main(cfg: DictConfig):
    print(cfg)
    print(cfg.model.config_name)


if __name__ == "__main__":
    main()
