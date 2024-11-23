
import torch
import argparse
import torch.nn as nn
import os
import random
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Union, Optional, Any
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, Seq2SeqTrainingArguments
from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers.modeling_outputs import CausalLMOutputWithPast
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torch.nn import BCEWithLogitsLoss
from datasets import load_dataset
from accelerate import PartialState

DIST_STATE = PartialState()

@DIST_STATE.on_local_main_process
def print_rank_0(msg):
    print(msg)

def print_rank(msg: str):
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    print(f'[LOCAL_RANK {local_rank}]:\n{msg}')


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
    if DIST_STATE.is_local_main_process:
        model.print_trainable_parameters()
    return model, tokenizer


class DatasetProcessor:
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.good_token = '+'
        self.bad_token = '-'
        self.step_tag = ' Rating'
        self.candidate_token_ids = self.tokenizer.encode(f" {self.good_token} {self.bad_token}")
        self.step_tag_id = self.tokenizer.encode(f"{self.step_tag}")[-1]
        
    def print_example(self, example):
        print_rank_0('*' * 20 + ' Example View ' + '*' * 20)
        print_rank_0('Tokenized Data:\n' + '=' * 30 + '\n'
            f'Input_ids: {example["input_ids"]}\nAttention_mask: {example["attention_mask"]}\nLabels: {example["labels"]}')
        ignore_index = -100 if not self.args.use_soft_label else -100.0
        valid_labels = [label for label in example["labels"] if label != ignore_index]
        if not self.args.use_soft_label:
            print_rank_0('Decoded Data:\n' + '=' * 30 + '\n'
                f'Input: {self.tokenizer.decode(example["input_ids"])}\nLabels: {self.tokenizer.decode(valid_labels)}')
        else:
            print_rank_0('Decoded Data:\n' + '=' * 30 + '\n'
                f'Input: {self.tokenizer.decode(example["input_ids"])}\nLabels: {valid_labels}')

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

        if not self.args.use_soft_label:
            tokenized_inputs['labels'] = [-100] * length
            for i, idx in enumerate(indices):
                if example['label'][i] in ['positive', 1]:
                    tokenized_inputs['labels'][idx] = self.candidate_token_ids[0]
                elif example['label'][i] in ['negative', 0]:
                    tokenized_inputs['labels'][idx] = self.candidate_token_ids[1]
                else:
                    raise ValueError('Invalid label value')
                tokenized_inputs['attention_mask'][idx] = 0
        else: # use soft labels
            tokenized_inputs['labels'] = [-100.0] * length
            for i, idx in enumerate(indices):
                tokenized_inputs['labels'][idx] = example['label'][i]
                tokenized_inputs['attention_mask'][idx] = 0

        assert len(tokenized_inputs["input_ids"]) == len(tokenized_inputs["labels"]) == len(tokenized_inputs["attention_mask"])
        return tokenized_inputs

    def prepare_datasets(self, training_args: Seq2SeqTrainingArguments, test_size=0.2, seed=42):
        dataset = load_dataset('json', data_files=self.args.data_path, split='train')
        dataset = dataset.filter(lambda x: x["prompt"])
        
        splits = dataset.train_test_split(
            test_size=test_size,
            seed=seed,
            shuffle=True
        )
        # breakpoint()
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
        # breakpoint()
        labels_index = torch.argwhere(torch.bitwise_or(
            labels == self.candidate_token_ids[0], 
            labels == self.candidate_token_ids[1]
        ))
        gold = torch.where(
            labels[labels_index[:, 0], labels_index[:, 1]] == self.candidate_token_ids[1], 
            0, 1
        )
        labels_index[:, 1] -= 1
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [
            self.candidate_token_ids[1],
            self.candidate_token_ids[0]
        ]]
        prob = torch.softmax(logits, dim=-1)
        return prob[:, 1], gold
    
    def preprocess_logits_for_soft_label_metrics(self, logits, labels):
        # breakpoint()
        labels_index = labels.ne(-100.0).nonzero()
        positive_labels = labels[labels_index[:, 0], labels_index[:, 1]]
        negative_labels = 1 - positive_labels
        gold = torch.stack([positive_labels, negative_labels], dim=-1).argmax(dim=-1)
        labels_index[:, 1] -= 1
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, self.candidate_token_ids]
        return logits, gold

    def compute_metrics(self, eval_pred):
        # breakpoint()
        if not self.args.use_soft_label:
            pre, labels = eval_pred
            auc = roc_auc_score(pre[1], pre[0])
            ll = log_loss(pre[1], pre[0])
            acc = accuracy_score(pre[1], pre[0] > 0.5)
            result = {
                'auc': auc,
                'll': ll,
                'acc': acc,
            }
        else:
            predictions, labels = eval_pred
            acc = accuracy_score(predictions[0].argmax(axis=-1), predictions[1])
            result = {
                'acc': acc,
            }
        print_rank_0(result)
        return result

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

@dataclass
class DataCollatorForSeq2SeqWithSoftLabels:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    This version supports soft labels (float values) in the label tensors.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`], *optional*):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
        label_pad_token_id (`float`, *optional*, defaults to 0.0):
            The value to use when padding the labels. Changed to 0.0 for soft labels.
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: float = -100.0  # Changed to float
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        
        # Handle None labels
        if labels is not None and all(label is None for label in labels):
            labels = None
            
        non_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]

        # Process inputs without labels
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # Process labels if they exist
        no_padding = self.padding is False or self.padding == PaddingStrategy.DO_NOT_PAD
        if labels is not None:
            if no_padding:
                if isinstance(features[0][label_name], list):
                    batch["labels"] = list(labels)
                else:
                    batch["labels"] = [np.concatenate([label, []]) for label in labels]
            else:
                max_padding = self.padding == PaddingStrategy.MAX_LENGTH and self.max_length is not None
                max_label_length = max(len(l) for l in labels) if not max_padding else self.max_length
                
                if self.pad_to_multiple_of is not None:
                    max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                    )

                padding_side = self.tokenizer.padding_side
                
                if isinstance(features[0][label_name], list):
                    for idx, label in enumerate(labels[0]):
                        if type(label) != type(self.label_pad_token_id):
                            raise ValueError(
                                f'The {idx} th label is of type {type(label)} while the label_pad_token_id is of type {type(self.label_pad_token_id)}, '
                                'you should make sure that they are of the same type'
                            )
                        
                    batch["labels"] = [
                        label + [self.label_pad_token_id] * (max_label_length - len(label))
                        if padding_side == "right"
                        else [self.label_pad_token_id] * (max_label_length - len(label)) + label
                        for label in labels
                    ]
                else:
                    # Convert to float16 for soft labels
                    batch["labels"] = [
                        np.concatenate(
                            [
                                label.astype(np.float16),
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.float16),
                            ]
                        )
                        if padding_side == "right"
                        else np.concatenate(
                            [
                                np.array([self.label_pad_token_id] * (max_label_length - len(label)), dtype=np.float16),
                                label.astype(np.float16),
                            ]
                        )
                        for label in labels
                    ]

        # Convert to appropriate tensor type
        if batch.get("labels", None) is not None:
            if return_tensors == "pt":
                import torch
                batch["labels"] = torch.tensor(batch["labels"], dtype=torch.float16)  # Changed to float16
            else:
                raise NotImplementedError(f"return_tensors='{return_tensors}' not supported yet.")
        else:
            batch["labels"] = None

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["labels"])
            batch["decoder_input_ids"] = decoder_input_ids

        return batch


class PRMTrainerForTokenPrediction(Trainer):
    def __init__(self, prm_use_tokens_cfg: Dict[str, Union[int, List[int]]], **kwargs):
        super().__init__(**kwargs)
        self.prm_use_tokens_cfg = prm_use_tokens_cfg
        self.loss_func = nn.functional.cross_entropy
        # self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        placeholder_token_id = self.prm_use_tokens_cfg["placeholder_token_id"]
        candidate_token_ids_for_prediction = self.prm_use_tokens_cfg["candidate_token_ids_for_prediction"]

        logits = outputs.logits
        # new_labels = torch.zeros_like(logits).to(outputs.logits.dtype)
        # positive_labels = labels.to(logits.dtype)
        # negative_labels = 1 - positive_labels
        # new_labels[..., candidate_token_ids_for_prediction] = torch.stack([negative_labels, positive_labels], dim=-1)
        
        # reference from https://github.com/OpenRLHF/OpenRLHF/blob/460477d628751bfaa95297af2763f2fd729ecd20/openrlhf/models/loss.py#L259
        placeholder_positions = (inputs["input_ids"] == placeholder_token_id).nonzero()
        shift_placeholder_positions = placeholder_positions.clone()
        shift_placeholder_positions[:, -1] -= 1
        logits = logits[shift_placeholder_positions[:, 0], shift_placeholder_positions[:, 1], :]
        labels = labels[placeholder_positions[:, 0], placeholder_positions[:, 1]]
        if len(candidate_token_ids_for_prediction) != 2:
            raise ValueError("The number of candidate tokens for prediction must be 2.")
        logits = logits[..., candidate_token_ids_for_prediction]
        positive_labels = labels.to(logits.dtype)
        negative_labels = 1 - positive_labels
        labels = torch.stack([positive_labels, negative_labels], dim=-1)
        reduction = 'sum' if num_items_in_batch is not None else 'mean'
        loss = self.loss_func(logits, labels, reduction=reduction)
        if reduction == 'sum':
            loss /= num_items_in_batch

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def run_exp(model_args, data_args, training_args):
    print_rank_0('loading model and toeknizer...')
    model, tokenizer = setup_model_and_tokenizer(model_args.model_name_or_path)

    processor = DatasetProcessor(data_args, tokenizer)
    print_rank_0('start data processing...')
    tokenized_datasets = processor.prepare_datasets(training_args)
    # Data collator for padding inputs dynamically
    # print_rank(tokenized_datasets["train"][101]["input_ids"])
    # print_rank(tokenizer.decode(tokenized_datasets["train"][101]["input_ids"]))
    if not data_args.use_soft_label:
        data_collator = DataCollatorForSeq2Seq(tokenizer)
    else:
        data_collator = DataCollatorForSeq2SeqWithSoftLabels(tokenizer)

    # world_size = int(os.environ.get('WORLD_SIZE', 1))
    world_size = DIST_STATE.num_processes
    per_device_total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    TOTAL_BATCH_SIZE = per_device_total_batch_size * world_size
    print_rank_0(f"Total batch size: {TOTAL_BATCH_SIZE}")

    fp = f'bs_{TOTAL_BATCH_SIZE}_g_{training_args.gradient_accumulation_steps}_lr_{training_args.learning_rate}_ep_{training_args.num_train_epochs}'
    training_args.output_dir = os.path.join(training_args.output_dir, fp)
    training_args.logging_dir = os.path.join(training_args.output_dir, 'logs')

    prm_use_tokens_cfg = {
        "placeholder_token_id": processor.step_tag_id,
        "candidate_token_ids_for_prediction": processor.candidate_token_ids,
    }
    if data_args.use_soft_label:
        trainer = PRMTrainerForTokenPrediction(
            prm_use_tokens_cfg=prm_use_tokens_cfg,
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],  # Replace with a validation set if available
            data_collator=data_collator,
            tokenizer=tokenizer,
            preprocess_logits_for_metrics=processor.preprocess_logits_for_soft_label_metrics,
            compute_metrics=processor.compute_metrics,
        )
    else:
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
    # parser = argparse.ArgumentParser()

    # parser.add_argument("--model_path", type=str, required=True)
    # parser.add_argument("--data_path", type=str, required=True)
    # parser.add_argument("--server", type=str, default='1')
    
    # #
    # parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--epochs", type=int, default=3)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    
    # # 
    # parser.add_argument("--eval_steps", type=int, default=500)
    # parser.add_argument("--save_steps", type=int, default=500)
    # parser.add_argument("--save_total_limit", type=int, default=3)
    # parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # args = parser.parse_args()

    # print('*' * 50 + f'\n{args}\n' + '*' * 50)

    # run_exp(args)
    ...