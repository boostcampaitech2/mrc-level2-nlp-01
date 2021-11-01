from dataclasses import dataclass, field


@dataclass
class ProjectArguments:
    base_path: str = field(
        default="/opt/ml/code/models", metadata={"help": "프로젝트 기본 경로"}
    )
    name: str = field(default=None, metadata={"help": "프로젝트 이름"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataArguments:
    dataset_path: str = field(
        default="/opt/ml/data/train_dataset",
        metadata={"help": "데이터 경로"},
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "토큰 최대길이"},
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "토큰 겹치는 부분"},
    )
    max_answer_length: int = field(
        default=32,
        metadata={"help": "정답 토큰 최대길이"},
    )
    preprocessing_num_workers: int = field(
        default=1,
        metadata={"help": "num workers"},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )


@dataclass
class EarlyStoppingArguments:
    """
    Arguments for Early-Stopping
    """

    setting: bool = field(
        default=False,
        metadata={"help": "Whether to use early-stopping or not"},
    )
    patience: int = field(
        default=5,
        metadata={"help": "Patience for early-stopping"},
    )


@dataclass
class InferenceArguments:
    topk: int = (
        field(
            default=5,
            metadata={"help": "Top-K 값"},
        ),
    )
    output_dir: str = field(
        default="/opt/ml/code/outputs/sample",
        metadata={"help": "출력장소"},
    )
