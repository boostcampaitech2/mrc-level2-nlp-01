from dataclasses import dataclass, field


@dataclass
class ColumnArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    question_column_name: str = field(
        default="question",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    context_column_name: str = field(
        default="context",
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    answer_column_name: str = field(
        default="answers",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


if __name__ == "__main__":
    print(ColumnArguments())
    print(ColumnArguments(answer_column_name="나나나"))
