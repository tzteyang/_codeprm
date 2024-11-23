## prepare for statistics
# python data_format_utils.py collect \
#     --data_dir "/root/autodl-tmp/projects/_codeprm/process_data_annotation/outputs" \
#     --output_file "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json"

# python data_format_utils.py analyze_steps_info \
#     --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json" \
#     --output_file "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json"

## do analyze
# python data_format_utils.py analyze_value_distribution \
#     --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json" \
#     --output_file "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/value_distribution_view.png"

# python data_format_utils.py analyze_steps_info \
#     --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json" \
#     --preprocess False \
#     --output_file "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/steps_info_view.png"

# align the data format for prm training
python data_format_utils.py to_prm_train_format \
    --data_path "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/collected_process_annotation_steps_data.json" \
    --output_file "/root/autodl-tmp/projects/_codeprm/process_data_annotation/data/raw_data/prm_train_raw_soft.json" \
    --use_hard_label "False"