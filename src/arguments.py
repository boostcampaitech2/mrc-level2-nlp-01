from dataclasses import dataclass, field

@dataclass
class ProjectArguments:
    base_path: str = field(
        default="/opt/ml/code/models",
        metadata={
            "help": "프로젝트 기본 경로"
        }
    )
    name: str = field(
        default=None,
        metadata={
            "help": "프로젝트 이름"
        }
    )
    
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
        metadata={
            "help": "데이터 경로"
        },
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "토큰 최대길이"
        },
    )
    stride: int = field(
        default=128,
        metadata={
            "help": "토큰 겹치는 부분"
        },
    )
    max_answer_length: int = field(
        default=32,
        metadata={
            "help": "정답 토큰 최대길이"
        },
    )

@dataclass
class EarlyStoppingArguments:
    setting: bool = field(
        default=True,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    patience: int = field(
        default=5,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )