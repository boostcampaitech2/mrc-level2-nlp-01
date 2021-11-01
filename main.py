import hydra
import copy
import os
import argparse

from omegaconf import DictConfig

from src.train import train
from src.hp_opt import hp_optimizing
from src.inference import inference
from src.retrieval import make_retrieval_datasets


def main(cf_name):
    @hydra.main(config_path="configs", config_name=cf_name)
    def inner_main(cfg: DictConfig):
        cfg = copy.deepcopy(cfg)
        print(f"Start {cfg.project.name} !")
        if cfg.mode == "model_train":
            os.environ["WANDB_ENTITY"] = cfg.wandb.entity
            os.environ["WANDB_PROJECT"] = cfg.wandb.project
            train(cfg.project, cfg.model, cfg.data, cfg.train, cfg.early_stopping)
        elif cfg.mode == "hyperparameter_tune":
            hp_optimizing(cfg.project, cfg.model, cfg.data, cfg.hp)
        elif cfg.mode == "retrieval":
            make_retrieval_datasets(cfg.retrieval)
        elif cfg.mode == "inference":
            inference(cfg.model, cfg.data, cfg.inference)

    return inner_main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name", type=str, default="main_args", help="설정 yaml 파일을 입력해주세요."
    )
    print(parser.parse_args())
    cf_name = parser.parse_args().config_name
    main(cf_name)
