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


@dataclass
class RetrievalArguments:
    output_name: str = field(
        default="new_test_dataset",
        metadata={"help": "출력할 데이터셋 이름"},
    )
    k: int = field(
        default=10,
        metadata={"help": "Top-K 값"},
    )
    data_path: str = field(
        default="/opt/ml/data",
        metadata={"help": "데이터셋 폴더 위치"},
    )
    target_dataset: str = field(
        default="test_dataset",
        metadata={"help": "변환할 데이터셋 이름"},
    )
    context_path: str = field(
        default="wiki_preprocessed_droped",
        metadata={"help": "data_path와 자동으로 Join"},
    )
    tokenizer_name: str = field(
        default="xlm-roberta-large",
        metadata={"help": "토크나이저로 사용할 모델 이름"},
    )
    type: str = field(
        default="Plus",
        metadata={"help": "BM25타입 Plus | L | Okapi"},
    )
    k1: float = field(
        default=1.6,
        metadata={"help": "BM25 k1 값"},
    )
    b: float = field(
        default=0.3,
        metadata={"help": "BM25 b 값"},
    )
    ep: float = field(
        default=0.25,
        metadata={"help": "BM25 ep 값, Okapi만 사용"},
    )
    delta: float = field(
        default=0.7,
        metadata={"help": "BM25 delta 값, L과 Plus가 사용"},
    )
    retrieval_class: str = field(
        default="TokenizerRetrieval",
        metadata={"help": "retrieval/sparse에 있는 클래스 이름"},
    )
    is_join: bool = field(
        default=True,
        metadata={"help": "context를 조인할지, 아니면 따로 내보낼지"},
    )
