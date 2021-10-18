import hydra

from omegaconf import DictConfig

from src.train import train


@hydra.main(config_path=".", config_name="main_args")
def main(cfg: DictConfig):
    print(f"Start {cfg.project.name} !")
    if cfg.mode == "model_train":
        train(cfg.project, cfg.model, cfg.data, cfg.train)
    elif cfg.mode == "retrieval_train":
        print("dense retrieval train 만들기")
    elif cfg.mode == "validation":
        print("평가데이터 만들기")


if __name__ == "__main__":
    main()
