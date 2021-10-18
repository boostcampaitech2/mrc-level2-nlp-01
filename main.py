import hydra
import wandb
import copy

from omegaconf import DictConfig

from src.train import train
from src.hp_opt import hp_optimizing


@hydra.main(config_path=".", config_name="main_args")
def main(cfg: DictConfig):
    cfg = copy.deepcopy(cfg)
    print(f"Start {cfg.project.name} !")
    wandb.init(name=cfg.wandb.name, project=cfg.wandb.project, entity=cfg.wandb.entity)
    if cfg.mode == "model_train":
        train(cfg.project, cfg.model, cfg.data, cfg.train)
    elif cfg.mode == "hyperparameter_tune":
        hp_optimizing(cfg.project, cfg.model, cfg.data, cfg.hp)
    elif cfg.mode == "retrieval_train":
        print("dense retrieval train 만들기")
    elif cfg.mode == "validation":
        print("평가데이터 만들기")


if __name__ == "__main__":
    main()
