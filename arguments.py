from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# import third party modules
import yaml

# Read config.yaml file
with open("arguments.yaml") as infile:
    SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)

DATA_CFG = SAVED_CFG["data"]
TRAIN_CFG = SAVED_CFG["train"]
RETRIEVAL_CFG = SAVED_CFG["retrieval"]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=TRAIN_CFG["model_name_or_path"],
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=TRAIN_CFG["config_name"],
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=TRAIN_CFG["tokenizer_name"],
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=DATA_CFG["dataset_name"],
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=DATA_CFG["overwrite_cache"],
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=DATA_CFG["preprocessing_num_workers"],
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=DATA_CFG["max_seq_length"],
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=DATA_CFG["pad_to_max_length"],
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=DATA_CFG["doc_stride"],
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=DATA_CFG["max_answer_length"],
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=DATA_CFG["eval_retrieval"],
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=DATA_CFG["num_clusters"],
        metadata={"help": "Define how many clusters to use for faiss."},
    )
    top_k_retrieval: int = field(
        default=RETRIEVAL_CFG["top_k_retrieval"],
        metadata={"help": "Define how many top-k passages to retrieve based on similarity."},
    )
    use_faiss: bool = field(
        default=RETRIEVAL_CFG["use_faiss"], metadata={"help": "Whether to build with faiss"}
    )
