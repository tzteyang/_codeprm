import argparse
import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import Seq2SeqTrainingArguments, HfArgumentParser

from train_prm.train_qwen import run_exp

def print_rank_0(msg):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == 0:
        print(msg)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model", "required": True}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_path: str = field(
        metadata={"help": "Path to dataset", "required": True}
    )
    server: str = field(
        default="1",
        metadata={"help": "Server configuration"}
    )

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print_rank_0('*'*30+f'Model arguments:\n{model_args}\nData arguments:\n{data_args}\nTraining arguments:\n{training_args}\n'+'*'*30)

    run_exp(model_args, data_args, training_args)

if __name__ == '__main__':
    main()