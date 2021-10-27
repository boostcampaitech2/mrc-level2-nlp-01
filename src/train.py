import os


from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    DataCollatorWithPadding,
    RobertaForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_from_disk, load_metric


from src.arguments import ProjectArguments, ModelArguments, DataArguments
from src.magic_box.preprocess import (
    prepare_train_features_with_setting,
    prepare_validation_features_with_setting,
    prepare_train_features_with_setting_for_seq2seq,
    prepare_validation_features_with_setting_for_seq2seq,
)
from src.magic_box.postprocess import post_processing_function_with_args
from src.magic_box.train_qa import QuestionAnsweringTrainer
from src.magic_box.utils_qa import EM_F1_compute_metrics, set_seed, compute_metrics_for_seq2seq

import nltk

def train(project_args, model_args, dataset_args, train_args):
    # 기본 변수 설정
    project_args = ProjectArguments(**project_args)
    model_args = ModelArguments(**model_args)
    dataset_args = DataArguments(**dataset_args)

    # 모델 및 토크나이저 로드
    model_name = model_args.name_or_path

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_args.seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(model_name, config=config)

    # 시드 설정
    set_seed(42)

    # 데이터 셋 로드

    datasets = load_from_disk(dataset_args.dataset_path)

    # 토크나이징 진행

    is_roberta = isinstance(model, RobertaForQuestionAnswering)
    is_seq2seq = model_args.seq2seq

    if is_seq2seq:
        tokenized_train_datasets = datasets["train"].map(
            prepare_train_features_with_setting_for_seq2seq(tokenizer, dataset_args),
            batched=True,
            num_proc=12,
            remove_columns=datasets["train"].column_names,
            load_from_cache_file=False,
        )
        tokenized_valid_datasets = datasets["validation"].map(
            prepare_validation_features_with_setting_for_seq2seq(
                tokenizer, dataset_args
            ),
            batched=True,
            num_proc=12,
            remove_columns=datasets["validation"].column_names,
            load_from_cache_file=False,
        )
    else:
        tokenized_train_datasets = datasets["train"].map(
            prepare_train_features_with_setting(tokenizer, dataset_args, is_roberta),
            batched=True,
            remove_columns=datasets["train"].column_names,
        )

        tokenized_valid_datasets = datasets["validation"].map(
            prepare_validation_features_with_setting(
                tokenizer, dataset_args, is_roberta
            ),
            batched=True,
            remove_columns=datasets["validation"].column_names,
        )

    # 데이터 콜레터 진행
    if is_seq2seq:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=tokenizer.pad_token_id,
            pad_to_multiple_of=8 if train_args.fp16 else None,
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if train_args.fp16 else None
        )

    # 트레이닝 옵션 설정
    train_args.output_dir = os.path.join(
        project_args.base_path, project_args.name, train_args.output_dir
    )
    train_args.logging_dir = os.path.join(
        project_args.base_path, project_args.name, train_args.logging_dir
    )

    training_args = set_training_args(train_args, project_args.name, is_seq2seq)

    metric = load_metric("squad")
    def postprocess_text(preds, labels):
        """
        postprocess는 nltk를 이용합니다.
        Huggingface의 TemplateProcessing을 사용하여
        정규표현식 기반으로 postprocess를 진행할 수 있지만
        해당 미션에서는 nltk를 이용하여 간단한 후처리를 진행합니다
        """

        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        print(preds, labels)
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        # print(eval_preds)
        preds, labels = eval_preds.predictions, eval_preds.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        # print(preds)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels은 rouge metric을 위한 것이며, f1/em을 구할 때 사용되지 않음
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        print(decoded_preds, decoded_labels)
        # 간단한 post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        formatted_predictions = [{"id": ex["id"], "prediction_text": decoded_preds[i]} for i, ex in enumerate(datasets["validation"])]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in datasets["validation"]]

        result = metric.compute(predictions=formatted_predictions, references=references)
        return result
    # Trainer 초기화
    if is_seq2seq:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_valid_datasets,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
    else:
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
    # if training_args.do_eval:
    #     metrics = trainer.evaluate()

    #     metrics["eval_samples"] = len(tokenized_valid_datasets)

    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)


def set_training_args(train_args, name, is_seq2seq):
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    if is_seq2seq:
        return Seq2SeqTrainingArguments(
            do_train=True,
            do_eval=True,
            report_to="wandb",
            run_name=name,
            greater_is_better=True,
            **train_args
        )
    else:
        return TrainingArguments(
            do_train=True,
            do_eval=True,
            report_to="wandb",
            run_name=name,
            greater_is_better=True,
            **train_args
        )
