# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Pre-processing
Post-processing utilities for question answering.
"""
import collections
import json
import logging
import os
from typing import Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

import torch
import random

from transformers import (
    is_torch_available,
    PreTrainedTokenizerFast,
    TrainingArguments,
    EvalPrediction,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.trainer_utils import get_last_checkpoint
from datasets import DatasetDict, load_metric


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForSpanMasking(DataCollatorForLanguageModeling):
    
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch
        
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        
        masked_indices = torch.zeros(size=labels.shape)

        mask_token = self.tokenizer.mask_token_id
        for i in range(len(inputs)):
            # sep ???????????? question??? context??? ???????????? ??????.
            sep_idx = np.where(inputs[i].numpy() == self.tokenizer.sep_token_id)
            # q_ids = > ????????? sep ????????????
            q_ids = sep_idx[0][0]
            mask_idxs = set()
            while len(mask_idxs) < 1:
                # 1 ~ q_ids????????? Question ??????
                ids = random.randrange(1, q_ids-1)
                mask_idxs.add(ids)

            for mask_idx in list(mask_idxs):
                inputs[i][mask_idx: mask_idx+2] = mask_token
                masked_indices[i][mask_idx: mask_idx+2] = True

        masked_indices = masked_indices.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        return inputs, labels


def set_seed(seed: int = 42):
    """
    seed ???????????? ?????? (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def postprocess_qa_predictions(
    examples,  # eval_example??? ??????
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes : qa model??? prediction ?????? ??????????????? ??????
    ????????? start logit??? end logit??? ???????????? ?????????, ?????? ???????????? original text??? ???????????? ???????????? ?????????

    Args:
        examples: ????????? ?????? ?????? ???????????? (see the main script for more information).
        features: ???????????? ????????? ???????????? (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            ????????? ????????? :start logits??? the end logits??? ???????????? two arrays              ????????? ????????? :obj:`features`??? element??? ????????? ????????????.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            ????????? ?????? ??????????????? ????????????????????? ????????? ?????????
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            ????????? ?????? ??? ????????? n-best prediction ??? ??????
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            ????????? ??? ?????? ????????? ?????? ??????
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            null ????????? ???????????? ??? ???????????? threshold
            : if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            ????????? ?????? ???????????? ??????
            dictionary : predictions, n_best predictions (with their scores and logits) if:obj:`version_2_with_negative=True`,
            dictionary : the scores differences between best and null answers
        prefix (:obj:`str`, `optional`):
            dictionary??? `prefix`??? ???????????? ?????????
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            ??? ??????????????? main process?????? ??????(logging/save??? ???????????? ????????? ????????? ???????????? ??? ?????????)
    """
    assert (
        len(predictions) == 2
    ), "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # example??? mapping?????? feature ??????
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(
            i
        )  # ????????? ?????? ???????????? ????????????

    # prediction, nbest??? ???????????? OrderedDict ???????????????.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(
        f"Post-processing {len(examples)} example predictions split into {len(features)} features."
    )

    # ?????? example?????? ?????? main Loop
    for example_index, example in enumerate(tqdm(examples)):
        # ???????????? ?????? example index
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # ?????? example??? ?????? ?????? feature ???????????????.
        for feature_index in feature_indices:
            # ??? featureure??? ?????? ?????? prediction??? ???????????????.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # logit??? original context??? logit??? mapping?????????.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional : `token_is_max_context`, ???????????? ?????? ?????? ???????????? ????????? ??? ?????? max context??? ?????? answer??? ???????????????
            token_is_max_context = features[feature_index].get(
                "token_is_max_context", None
            )

            # minimum null prediction??? ???????????? ?????????.
            feature_null_score = start_logits[0] + end_logits[0]
            if (
                min_null_prediction is None
                or min_null_prediction["score"] > feature_null_score
            ):
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # `n_best_size`?????? ??? start and end logits??? ???????????????.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()

            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # out-of-scope answers??? ???????????? ????????????.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # ????????? < 0 ?????? > max_answer_length??? answer??? ???????????? ????????????.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue
                    # ?????? context??? ?????? answer??? ???????????? ????????????.
                    if (
                        token_is_max_context is not None
                        and not token_is_max_context.get(str(start_index), False)
                    ):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (
                                offset_mapping[start_index][0],
                                offset_mapping[end_index][1],
                            ),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "doc_score": example["scores"]
                            if "score" in example.keys()
                            else None,
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )

        if version_2_with_negative:
            # minimum null prediction??? ???????????????.
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # ?????? ?????? `n_best_size` predictions??? ???????????????.
        predictions = sorted(
            prelim_predictions, key=lambda x: x["score"], reverse=True
        )[:n_best_size]

        # ?????? ????????? ?????? ????????? ?????? minimum null prediction??? ?????? ???????????????.
        if version_2_with_negative and not any(
            p["offsets"] == (0, 0) for p in predictions
        ):
            predictions.append(min_null_prediction)

        # offset??? ???????????? original context?????? answer text??? ???????????????.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # rare edge case?????? null??? ?????? ????????? ????????? ????????? failure??? ????????? ?????? fake prediction??? ????????????.
        if len(predictions) == 0 or (
            len(predictions) == 1 and predictions[0]["text"] == ""
        ):

            predictions.insert(
                0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0}
            )

        # ?????? ????????? ?????????????????? ???????????????(we do it with numpy to stay independent from torch/tf in this file, using the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # ???????????? ????????? ???????????????.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # best prediction??? ???????????????.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # else case : ?????? ?????? ?????? ?????? ????????? ????????? ????????? ?????????
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # threshold??? ???????????? null prediction??? ???????????????.
            score_diff = (
                null_score
                - best_non_null_pred["start_logit"]
                - best_non_null_pred["end_logit"]
            )
            scores_diff_json[example["id"]] = float(score_diff)  # JSON-serializable ??????
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # np.float??? ?????? float??? casting -> `predictions`??? JSON-serializable ??????
        all_nbest_json[example["id"]] = [
            {
                k: (
                    float(v)
                    if isinstance(v, (np.float16, np.float32, np.float64))
                    else v
                )
                for k, v in pred.items()
            }
            for pred in predictions
        ]

    # output_dir??? ????????? ?????? dicts??? ???????????????.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir,
            "predictions.json" if prefix is None else f"predictions_{prefix}".json,
        )
        nbest_file = os.path.join(
            output_dir,
            "nbest_predictions.json"
            if prefix is None
            else f"nbest_predictions_{prefix}".json,
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir,
                "null_odds.json" if prefix is None else f"null_odds_{prefix}".json,
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n"
            )
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w", encoding="utf-8") as writer:
            writer.write(
                json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n"
            )
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w", encoding="utf-8") as writer:
                writer.write(
                    json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n"
                )

    return all_predictions


def check_no_error(
    data_args,
    training_args: TrainingArguments,
    datasets: DatasetDict,
    tokenizer,
) -> Tuple[Any, int]:

    # last checkpoint ??????.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Tokenizer check: ?????? script??? Fast tokenizer??? ??????????????????.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warn(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    if "validation" not in datasets:
        raise ValueError("--do_eval requires a validation dataset")
    return last_checkpoint, max_seq_length


def EM_F1_compute_metrics():
    metric = load_metric("squad")

    def compute_metrics(p: EvalPrediction):
        result = metric.compute(predictions=p.predictions, references=p.label_ids)
        return {"eval_exact_match": result["exact_match"], "eval_f1": result["f1"]}

    return compute_metrics
