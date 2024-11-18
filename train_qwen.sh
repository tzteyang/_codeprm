accelerate launch --config_file "/root/autodl-tmp/projects/_codeprm/train_prm/dist_configs/multi_gpu.yaml" \
    ./run_prm_train.py \
    --model_path "/root/autodl-tmp/projects/models/Qwen2.5-Coder-7B-Instruct" \
    --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/prm_train_raw.json" \
    --output_dir "/root/autodl-tmp/projects/_codeprm/train_prm/outputs" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 3 \
    --learning_rate 1e-4 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 2 \
    # --resume_from_checkpoint ""
