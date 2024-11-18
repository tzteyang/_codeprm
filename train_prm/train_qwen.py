
import torch
import argparse
import os
import random

from typing import Tuple, List, Dict, Union, Optional
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, Seq2SeqTrainingArguments
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torch.nn import BCEWithLogitsLoss
from datasets import concatenate_datasets
from datasets import load_dataset


def print_rank_0(msg):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    if local_rank == 0:
        print(msg)

def print_rank(msg: str):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    print(f'rank {local_rank}:\n{msg}')


def setup_model_and_tokenizer(model_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        add_eos_token=False, 
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to('cuda')

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # LoRA for causal language modeling task
        r=8,  # Rank of LoRA
        lora_alpha=32,  # Alpha scaling factor for LoRA
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific layers
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


class DatasetProcessor:
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = ' Rating'
        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}")
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1]
        
    def print_example(self, example):
        print_rank_0('*' * 20 + ' Example View ' + '*' * 20)
        print_rank_0('Tokenized Data:\n' + '=' * 30 + '\n'
            f'Input_ids: {example["input_ids"]}\nAttention_mask: {example["attention_mask"]}\nLabels: {example["labels"]}')
        valid_labels = [label for label in example["labels"] if label != -100]
        print_rank_0('Decoded Data:\n' + '=' * 30 + '\n'
            f'Input: {self.tokenizer.decode(example["input_ids"])}\nLabels: {self.tokenizer.decode(valid_labels)}')

    def preprocess_example(self, example):
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["response"]},
        ]
        end_token_of_one_turn = '<|im_end|>\n'
        input_w_template = self.tokenizer.apply_chat_template(messages, tokenize=False)
        input_w_template = input_w_template.removesuffix(end_token_of_one_turn)
        input_w_template += self.step_tag
        
        tokenized_inputs = self.tokenizer(input_w_template, padding=True)
        
        indices = [i for i, x in enumerate(tokenized_inputs['input_ids']) if x == self.step_tag_id]
        
        if len(indices) != len(example['label']):
            example['label'] = example['label'][:len(indices)]
        
        assert len(indices) == len(example['label'])

        length = len(tokenized_inputs['input_ids'])
        tokenized_inputs['labels'] = [-100] * length
        for i, idx in enumerate(indices):
            if example['label'][i] in ['positive', 1]:
                tokenized_inputs['labels'][idx] = self.candidate_tokens[0]
            elif example['label'][i] in ['negative', 0]:
                tokenized_inputs['labels'][idx] = self.candidate_tokens[1]
            else:
                raise ValueError('Invalid label value')
            tokenized_inputs['attention_mask'][idx] = 0
        assert len(tokenized_inputs["input_ids"]) == len(tokenized_inputs["labels"]) == len(tokenized_inputs["attention_mask"])
        return tokenized_inputs

    def prepare_datasets(self, data_path, training_args: Seq2SeqTrainingArguments, test_size=0.2, seed=42):
        dataset = load_dataset('json', data_files=data_path, split='train')
        dataset = dataset.filter(lambda x: x["prompt"])
        
        splits = dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            shuffle=True
        )

        with training_args.main_process_first(desc="Tokenizing datasets"):
            tokenized_datasets = {
                split: splits[split].map(
                    self.preprocess_example,
                    remove_columns=splits[split].column_names,
                )
                for split in splits
            }

        print_rank_0(f"Training set size: {len(tokenized_datasets['train'])}")
        print_rank_0(f"Test set size: {len(tokenized_datasets['test'])}")
        
        ridx = random.randint(0, len(tokenized_datasets["train"]) - 1)
        self.print_example(tokenized_datasets["train"][ridx])
        # breakpoint()
        return tokenized_datasets

    # Define a custom metric function (e.g., accuracy for binary classification)
    def preprocess_logits_for_metrics(self, logits, labels):
        labels_index = torch.argwhere(torch.bitwise_or(
            labels == self.candidate_tokens[0], 
            labels == self.candidate_tokens[1]
        ))
        gold = torch.where(
            labels[labels_index[:, 0], labels_index[:, 1]] == self.candidate_tokens[1], 
            0, 1
        )
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [
            self.candidate_tokens[1], 
            self.candidate_tokens[0]
        ]]
        prob = torch.softmax(logits, dim=-1)
        return prob[:, 1], gold

    def compute_metrics(self, eval_pred):
        pre, labels = eval_pred
        auc = roc_auc_score(pre[1], pre[0])
        ll = log_loss(pre[1], pre[0])
        acc = accuracy_score(pre[1], pre[0] > 0.5)
        result = {
            'auc': auc,
            'll': ll,
            'acc': acc,
        }
        print_rank_0(result)
        return result

def run_exp(model_args, data_args, training_args):
    print_rank_0('loading model and toeknizer...')
    model, tokenizer = setup_model_and_tokenizer(model_args.model_name_or_path)
    print_rank_0(model)
    processor = DatasetProcessor(tokenizer)
    print_rank_0('start data processing...')
    tokenized_datasets = processor.prepare_datasets(data_args.data_path, training_args)
    # Data collator for padding inputs dynamically
    # print_rank(tokenized_datasets["train"][101]["input_ids"])
    # print_rank(tokenizer.decode(tokenized_datasets["train"][101]["input_ids"]))
    data_collator = DataCollatorForSeq2Seq(tokenizer)

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    per_device_total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    TOTAL_BATCH_SIZE = per_device_total_batch_size * world_size
    print_rank_0(f"Total batch size: {TOTAL_BATCH_SIZE}")

    fp = f'bs_{TOTAL_BATCH_SIZE}_g_{training_args.gradient_accumulation_steps}_lr_{training_args.learning_rate}_ep_{training_args.num_train_epochs}'
    training_args.output_dir = os.path.join(training_args.output_dir, fp)
    training_args.logging_dir = os.path.join(training_args.output_dir, 'logs')
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # Replace with a validation set if available
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=processor.preprocess_logits_for_metrics,
        compute_metrics=processor.compute_metrics,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--server", type=str, default='1')
    
    #
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # 
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    args = parser.parse_args()

    print('*' * 50 + f'\n{args}\n' + '*' * 50)

    run_exp(args)