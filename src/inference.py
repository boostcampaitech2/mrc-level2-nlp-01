import json
import os
import errno


from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoConfig,
    pipeline,
)
from datasets import load_from_disk

from src.arguments import ModelArguments, DataArguments, InferenceArguments


def inference(model_args, dataset_args, inf_args):
    model_args = ModelArguments(**model_args)
    dataset_args = DataArguments(**dataset_args)
    inf_args = InferenceArguments(**inf_args)

    model_path = model_args.name_or_path

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        config=config,
    )

    datasets = load_from_disk(dataset_args.dataset_path)

    question_answerer = pipeline(
        "question-answering", model=model, tokenizer=tokenizer, device=0
    )

    predictions = {}
    n_predictions = {}
    result_list = []
    i = 0

    for data in tqdm(datasets):
        i += 1
        prediction = ""
        n_prediction = []
        for context in data["context"]:
            result = question_answerer(
                question=data["question"],
                context=context,
                topk=inf_args.topk,
                max_answer_len=dataset_args.max_answer_length,
                doc_stride=dataset_args.stride,
                max_seq_len=dataset_args.max_length,
            )
            if inf_args.topk == 1:
                result = [result]
            result_list.append(result)
            n_prediction.extend(result)
        if i > 3:
            break
        n_prediction.sort(key=lambda x: x["score"], reverse=True)
        n_prediction = n_prediction[: inf_args.topk]
        prediction = n_prediction[0]["answer"]
        predictions[data["id"]] = prediction
        n_predictions[data["id"]] = n_prediction

    if not os.path.exists(inf_args.output_dir):
        try:
            os.makedirs(inf_args.output_dir)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(os.path.join(inf_args.output_dir, "predictions.json"), "w") as outfile:
        json.dump(predictions, outfile)
    with open(os.path.join(inf_args.output_dir, "n_predictions.json"), "w") as outfile:
        json.dump(n_predictions, outfile)
