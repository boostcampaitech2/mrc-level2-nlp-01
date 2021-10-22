import hydra
import copy
import os

from omegaconf import DictConfig

from src.train import train
from src.hp_opt import hp_optimizing
from src.inference import inference


@hydra.main(config_path=".", config_name="main_args")
def main(cfg: DictConfig):
    cfg = copy.deepcopy(cfg)
    print(f"Start {cfg.project.name} !")
    os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    if cfg.mode == "model_train":
        train(cfg.project, cfg.model, cfg.data, cfg.train)
    elif cfg.mode == "hyperparameter_tune":
        hp_optimizing(cfg.project, cfg.model, cfg.data, cfg.hp)
    elif cfg.mode == "retrieval_train":
        print("dense retrieval train 만들기")
    elif cfg.mode == "inference":
        inference(cfg.model, cfg.data)


if __name__ == "__main__":
    main()
