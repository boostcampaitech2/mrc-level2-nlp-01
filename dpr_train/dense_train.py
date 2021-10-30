from datasets import load_from_disk, Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
)

from encoder import RobertaEncoder
from dense_retrieval import DenseRetrieval_with_Faiss


def main():
    # 평가를 안하니 검증데이터와 훈련데이터를 합칩니다.
    train_dataset = load_from_disk("/home/ubuntu/workspace/data/train_dataset")
    train = train_dataset["train"].to_dict()
    valid = train_dataset["validation"].to_dict()
    for key in train.keys():
        train[key].extend(valid[key])
    train_dataset = Dataset.from_dict(train)
    args = TrainingArguments(
        output_dir="dense_retireval_roberta_large",
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=20,
        weight_decay=0.01,
    )
    model_checkpoint = "klue/roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = RobertaEncoder.from_pretrained(model_checkpoint)
    q_encoder = RobertaEncoder.from_pretrained(model_checkpoint)
    retriever = DenseRetrieval_with_Faiss(
        args=args,
        dataset=train_dataset,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder,
        wiki_path="/home/ubuntu/workspace/data/wiki_preprocessed_droped",
    )
    retriever.train()
    retriever.build_faiss(
        index_file_path="/home/ubuntu/workspace/mrc-level2-nlp-01/dpr_train/wiki.index"
    )


if __name__ == "__main__":
    main()
