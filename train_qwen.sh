accelerate launch --config_file "/root/autodl-tmp/projects/_codeprm/train_prm/dist_configs/single_gpu.yaml" \
    ./run_prm_train.py \
    --model_name_or_path "/root/autodl-tmp/projects/models/Qwen2.5-Coder-7B-Instruct" \
    --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/prm_train_raw_soft.json" \
    --use_soft_label \
    --output_dir "/root/autodl-tmp/projects/_codeprm/train_prm/outputs" \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --save_strategy "steps" \
    --eval_strategy "steps" \
    --save_steps 200 \
    --eval_steps 10 \
    --save_total_limit 2 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --log_level "info" \
    --bf16 \
    --report_to "none" \
    --ddp_find_unused_parameters False \
    # --resume_from_checkpoint ""