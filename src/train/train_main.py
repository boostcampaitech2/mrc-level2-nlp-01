import os
import sys

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

if __name__ == "__main__":  # py를 직접 실행할때 절대 경로 설정
    print(__file__)
    root_folder = os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    print(root_folder)
    sys.path.append(root_folder)

from src.train.run_mrc import run_mrc


def load_model_and_tokenizer(model_name: str, tokenizer_name: str, config_name: str):
    config = AutoConfig.from_pretrained(config_name)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,  # True = Rust 컴파일러, False = 파이썬 컴파일러
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
    )
    return model, tokenizer


def train_main(args):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # argument 분리
    model_args, data_args, training_args = args.model, args.data, TrainingArguments
    print(f"모델 이름 or 위치 : {model_args.name}")

    # [참고] argument를 manual하게 수정하고 싶은 경우에 아래와 같은 방식을 사용할 수 있습니다
    # training_args.per_device_train_batch_size = 4
    # print(training_args.per_device_train_batch_size)

    print(f"model is from {model_args.name}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    # logging.basicConfig(
    #     format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    #     datefmt="%m/%d/%Y %H:%M:%S",
    #     handlers=[logging.StreamHandler(sys.stdout)],
    # )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    # 데이터셋을 로드합니다.

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # 모델과 토크나이저를 불러옵니다.
    model_name = model_args.name
    tokenizer_name = model_args.tokenizer_name
    config_name = model_args.config_name

    model, tokenizer = load_model_and_tokenizer(model_name, tokenizer_name, config_name)

    # 설정 확인
    print(
        type(training_args),
        type(model_args),
        type(datasets),
        type(tokenizer),
        type(model),
    )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


if __name__ == "__main__":
    print("Training python!")
