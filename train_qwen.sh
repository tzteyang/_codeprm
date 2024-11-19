accelerate launch --config_file "/root/autodl-tmp/projects/_codeprm/train_prm/dist_configs/multi_gpu.yaml" \
    ./run_prm_train.py \
    --model_name_or_path "/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct" \
    --data_path "/root/autodl-tmp/projects/_codeprm/train_prm/data/raw_data/prm_train_raw.json" \
    --output_dir "/root/autodl-tmp/projects/_codeprm/train_prm/outputs" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 2 \
    --eval_strategy "steps" \
    --save_strategy "steps" \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --bf16 \
    --report_to "wandb" \
    --ddp_find_unused_parameters False \
    # --resume_from_checkpoint ""