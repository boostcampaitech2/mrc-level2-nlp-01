from transformers import EvalPrediction


from src.magic_box.utils_qa import postprocess_qa_predictions

import pandas as pd

def post_processing_function_with_args(max_answer_length, valid_datasets):
    def post_processing_function(examples, features, predictions, training_args):
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=max_answer_length,
            output_dir=training_args.output_dir,
        )
        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        ids = [k for k, v in predictions.items()]
        texts = [v for k, v in predictions.items()]
        save_file = {"id":ids, "prediction_text": texts}
        save_file = pd.DataFrame(save_file).to_csv("/opt/ml/save_file.csv")
        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex["answers"]} for ex in valid_datasets
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    return post_processing_function
