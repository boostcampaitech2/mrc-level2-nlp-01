import os


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaForQuestionAnswering,
)
from datasets import load_from_disk


from src.magic_box.preprocess import (
    prepare_train_features_with_setting,
    prepare_validation_features_with_setting,
)
from src.magic_box.postprocess import post_processing_function_with_args
from src.magic_box.train_qa import QuestionAnsweringTrainer
from src.magic_box.utils_qa import EM_F1_compute_metrics, set_seed


def train(project_args, model_args, dataset_args, train_args):
    model_name = model_args.name

    # 모델 및 토크나이저 로드

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # 시드 설정
    set_seed(train_args.seed)

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
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if train_args.fp16 else None
    )

    # 트레이닝 옵션 설정
    output_dir = os.path.join(
        project_args.base_path, project_args.name, train_args.output
    )
    log_dir = os.path.join(project_args.base_path, project_args.name, train_args.log)
    training_args = set_training_args(
        output_dir, log_dir, train_args, project_args.name
    )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_datasets if training_args.do_train else None,
        eval_dataset=tokenized_valid_datasets if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function_with_args(
            dataset_args.max_answer_length, datasets["validation"]
        ),
        compute_metrics=EM_F1_compute_metrics(),
    )

    # Training
    if training_args.do_train:
        trainer.train()
        tokenizer.save_pretrained(
            os.path.join(project_args.base_path, project_args.name, "best_model")
        )
        model.save_pretrained(
            os.path.join(project_args.base_path, project_args.name, "best_model")
        )

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(tokenized_valid_datasets)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


def set_training_args(output_dir, log_dir, train_args, name):
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    return TrainingArguments(
        output_dir=output_dir,  # output directory
        logging_dir=log_dir,
        do_train=train_args.do_train,
        do_eval=train_args.do_eval,
        save_total_limit=train_args.save_total_limit,  # number of total save model.
        save_steps=train_args.save_steps,  # model saving step.
        num_train_epochs=train_args.num_train_epochs,  # total number of training epochs
        learning_rate=train_args.learning_rate,  # learning_rate
        per_device_train_batch_size=train_args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=train_args.batch_size,  # batch size for evaluation
        warmup_steps=train_args.warmup_steps,  # number of warmup steps for learning rate scheduler
        weight_decay=train_args.weight_decay,  # strength of weight decay
        logging_steps=train_args.logging_steps,  # log saving step.
        evaluation_strategy=train_args.evaluation_strategy,  # evaluation strategy to adopt during training
        fp16=train_args.fp16,
        dataloader_pin_memory=train_args.dataloader_pin_memory,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=train_args.eval_steps,  # evaluation step.
        load_best_model_at_end=train_args.load_best_model_at_end,
        seed=train_args.seed,
        metric_for_best_model=train_args.metric,
        greater_is_better=True,
        disable_tqdm=train_args.disable_tqdm,
        report_to="wandb",
        run_name=name,
    )
