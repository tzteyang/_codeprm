import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_path = '/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path, torch_dtype=torch.bfloat16
)
lora_adapter_path = '/root/autodl-tmp/projects/_codeprm/train_prm/outputs/prm_bs_8_g_4_lr_0.0001_ep_3.0'
model = PeftModel.from_pretrained(
    model=base_model, model_id=lora_adapter_path
)
merged_model = model.merge_and_unload()

merged_model_path = '/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct-PRM'
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)