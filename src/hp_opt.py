import os

from typing import Dict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaForQuestionAnswering,
)
from datasets import load_from_disk


from src.arguments import ProjectArguments, ModelArguments, DataArguments
from src.magic_box.preprocess import (
    prepare_train_features_with_setting,
    prepare_validation_features_with_setting,
)
from src.magic_box.postprocess import post_processing_function_with_args
from src.magic_box.train_qa import QuestionAnsweringTrainer
from src.magic_box.utils_qa import EM_F1_compute_metrics, set_seed


def hp_optimizing(project_args, model_args, dataset_args, hp_args):
    # wandb disable
    os.environ["WANDB_DISABLED"] = "true"

    # 기본 변수 설정
    project_args = ProjectArguments(**project_args)
    model_args = ModelArguments(**model_args)
    dataset_args = DataArguments(**dataset_args)

    # 모델 및 토크나이저 로드
    model_name = model_args.name_or_path

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # 데이터 셋 로드

    datasets = load_from_disk(dataset_args.dataset_path)

    # 토크나이징 진행

    is_roberta = isinstance(model, RobertaForQuestionAnswering)
    tokenized_train_datasets = datasets["train"].map(
        prepare_train_features_with_setting(tokenizer, dataset_args, is_roberta),
        batched=True,
        remove_columns=datasets["train"].column_names,
    )

    tokenized_valid_datasets = datasets["validation"].map(
        prepare_validation_features_with_setting(tokenizer, dataset_args, is_roberta),
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    # 데이터 콜레터 진행
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # 트레이닝 옵션 설정
    output_dir = os.path.join(project_args.base_path, project_args.name, hp_args.output)
    log_dir = os.path.join(project_args.base_path, project_args.name, hp_args.log)
    training_args = set_training_args(output_dir, log_dir, hp_args)

    def model_init():
        set_seed(42)
        return AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=tokenized_train_datasets,
        eval_dataset=tokenized_valid_datasets,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function_with_args(
            dataset_args.max_answer_length, datasets["validation"]
        ),
        compute_metrics=EM_F1_compute_metrics(),
    )

    trainer.hyperparameter_search(
        direction="maximize",
        backend="sigopt",
        n_trials=hp_args.n_trials,
        hp_space=closure_hp_space_sigopt(hp_args),
        compute_objective=EM_F1_selector_with_compute_objective(hp_args.eval_type),
    )


def set_training_args(output_dir, log_dir, hp_args):
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    return TrainingArguments(
        output_dir=output_dir,  # output directory
        logging_dir=log_dir,
        evaluation_strategy="epoch",  # evaluation strategy to adopt during training
        fp16=True,
        save_strategy=hp_args.save_strategy,
        save_total_limit=hp_args.save_total_limit,
        metric_for_best_model="exact_match",
        greater_is_better=True,
        disable_tqdm=True,
    )


def EM_F1_selector_with_compute_objective(eval_type):
    if eval_type == "exact_match":
        eval_type = "eval_exact_match"
    elif eval_type == "f1":
        eval_type = "eval_f1"

    def EM_compute_objective(metrics: Dict[str, float]) -> float:
        return metrics[eval_type]

    return EM_compute_objective


def closure_hp_space_sigopt(hp_args):
    def hp_space_sigopt(trial):
        return [
            {
                "bounds": {"min": 1e-6, "max": 1e-4},
                "name": "learning_rate",
                "type": "double",
                "transformamtion": "log",
            },
            {
                "bounds": {"min": 6, "max": 10},
                "name": "num_train_epochs",
                "type": "int",
            },
            {"bounds": {"min": 1, "max": 50}, "name": "seed", "type": "int"},
            {
                "bounds": {"min": 1e-4, "max": 5e-1},
                "name": "weight_decay",
                "type": "double",
                "transformamtion": "log",
            },
            {
                "bounds": {"min": 100, "max": 1000},
                "name": "warmup_steps",
                "type": "int",
            },
            {
                "bounds": {"min": 1, "max": 80},
                "name": "gradient_accumulation",
                "type": "int",
            },
        ]

    return hp_space_sigopt
