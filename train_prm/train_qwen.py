
import torch
import argparse
import os
import random

from typing import Tuple, List, Dict, Union, Optional
from peft import PeftModel
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from torch.nn import BCEWithLogitsLoss
from datasets import concatenate_datasets
from datasets import load_dataset


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

    return model, tokenizer


def preprocess_function(example, tokenizer: AutoTokenizer):
    good_token, bad_token = '+', '-'
    step_tag = ' Rating' # 4689

    candidate_tokens = tokenizer.encode(f" {good_token} {bad_token}") # [488, 481]
    step_tag_id = tokenizer.encode(f" {step_tag}") # [19216]
    assert len(step_tag_id) == 1
    step_tag_id = step_tag_id[-1]
    
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["response"]},
    ]
    end_token_of_one_turn = '<|im_end|>\n'
    input_w_template = tokenizer.apply_chat_template(messages, tokenize=False)
    # if not example["has_final_step"]:
    input_w_template.rstrip(end_token_of_one_turn)
    input_w_template += step_tag
    # input = f"{example['prompt']}{example['response']}"
    tokenized_inputs = tokenizer(
        input_w_template, padding=True,
    )
    
    def find_all_indices(lst, element):
        return [i for i, x in enumerate(lst) if x == element]
    
    length = len(tokenized_inputs['input_ids'])
    indices = find_all_indices(tokenized_inputs['input_ids'], step_tag_id)
    
    if len(indices) != len(example['label']):
        example['label'] = example['label'][:len(indices)]
    
    assert len(indices) == len(example['label'])
    
    tokenized_inputs['labels'] = [-100] * length

    for i in range(len(indices)):
        if example['label'][i] == '+' or example['label'][i] == 1:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[0]
        elif example['label'][i] == '-' or example['label'][i] == 0:
            tokenized_inputs['labels'][indices[i]] = candidate_tokens[1]
        else:
            raise ValueError('label is wrong')
        tokenized_inputs['attention_mask'][indices[i]] = 0
    
    return tokenized_inputs

def prepare_datasets(data_path, tokenizer, test_size=0.2, seed=42):
    dataset = load_dataset(
        'json', 
        data_files=data_path,
        split='train'
    )
    dataset = dataset.filter(lambda x: x["prompt"])
    
    splits = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
        shuffle=True
    )
    
    tokenized_datasets = {
        split: splits[split].map(
            lambda x: preprocess_function(x, tokenizer),
            remove_columns=['prompt', 'response', 'label', 'has_final_step'],
            # num_proc=4,
            desc=f"Processing {split} split",
        )
        for split in splits
    }
    
    print(f"训练集大小: {len(tokenized_datasets['train'])}")
    print(f"测试集大小: {len(tokenized_datasets['test'])}")
    
    return tokenized_datasets


def run_exp(args):
    print('loading model and toeknizer...')
    model_path = args.model_path
    model, tokenizer = setup_model_and_tokenizer(model_path)

    print('start data processing...')
    tokenized_datasets = prepare_datasets(args.data_path, tokenizer)

    # Data collator for padding inputs dynamically
    data_collator = DataCollatorWithPadding(tokenizer)

    BATCH_SIZE = args.total_batch_size
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // args.per_device_train_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        
        GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

    print(world_size)
    print(ddp)


    fp = f'bs_{args.total_batch_size}_lr_{args.learning_rate}_datasets_{args.datasets}'
    output_path = f'./prm_results_qwen_new.{args.server}/{fp}'


    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        evaluation_strategy="no",  # Evaluate at the end of each epoch
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        # fp16=True,  # Enable mixed precision for better performance on supported hardware
        bf16=True,
        report_to="none",  # Set to "wandb" if you are using Weights and Biases for logging
        dataloader_num_workers=4,
        deepspeed=None,
        ddp_find_unused_parameters=False,
    )

    # Define a custom metric function (e.g., accuracy for binary classification)
    def compute_metrics(eval_pred):
        # pass
        # print(eval_pred)
        print('bb')
        pre, labels = eval_pred
        auc = roc_auc_score(pre[1], pre[0])
        ll = log_loss(pre[1], pre[0])
        acc = accuracy_score(pre[1], pre[0] > 0.5)
        result ={
            'auc': auc, 
            'll': ll, 
            'acc': acc, 
        } 
        print(result)
        return result

    def preprocess_logits_for_metrics(logits,labels):
        print('aa')
        # return logits,labels
        labels_index = torch.argwhere(torch.bitwise_or(labels == candidate_tokens[0], labels == candidate_tokens[1]))
        gold = torch.where(labels[labels_index[:, 0], labels_index[:, 1]] == candidate_tokens[1], 0, 1)
        # labels_index[: , 1] = labels_index[: , 1] - 1
        logits = logits[labels_index[:, 0], labels_index[:, 1]][:, [candidate_tokens[1], candidate_tokens[0]]]
        prob = torch.softmax(logits, dim=-1)
        return prob[:, 1], gold
        

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],  # Replace with a validation set if available
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save the fine-tuned model and tokenizer
    model.save_pretrained('./fine_tuned_math_shepherd_lora_8bit')
    tokenizer.save_pretrained('./fine_tuned_math_shepherd_lora_8bit')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    # parser.add_argument("--total_batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--server", type=str, default='1')

    args = parser.parse_args()

    print('*' * 50 + f'\n{args}\n' + '*' * 50)

    run_exp(args)