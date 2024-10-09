for degree in high; do
    python my_run_clm.py \
    --model_name_or_path deepseek-ai/deepseek-coder-6.7b-base \
    --train_file data/ft_final/my_code_contests_train_${degree}.jsonl \
    --validation_file data/ft_final/my_code_contests_valid_${degree}.jsonl \
    --output_dir tmp/deepseek/${degree} \
    --save_steps 100 \
    --logging_steps 30 \
    --evaluation_strategy steps \
    --max_eval_samples 50 \
    --torch_dtype bfloat16 \
    --block_size 2048 \
    --preprocessing_num_workers 8 \
    --trust_remote_code 1 \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.01 \
    --low_cpu_mem_usage True \
    --overwrite_output_dir True \
    --report_to wandb \
    --run_name deepseekcoder-7b-${degree}-mod \
    --resume_from_checkpoint /data/kdy20401/Workspace/Proj-Code-Generation/MC/tmp/deepseek/high/checkpoint-200
done

# --max_train_samples 50 \


# # start from checkpoint
# degree=low
# python my_run_clm.py \
#     --model_name_or_path meta-llama/CodeLlama-7b-hf \
#     --train_file data/ft_final/my_code_contests_train_${degree}.jsonl \
#     --validation_file data/ft_final/my_code_contests_valid_${degree}.jsonl \
#     --output_dir tmp/CodeLlama \
#     --save_steps 5 \
#     --evaluation_strategy steps \
#     --max_train_samples 10 \
#     --torch_dtype bfloat16 \
#     --block_size 2048 \
#     --preprocessing_num_workers 8 \
#     --trust_remote_code 1 \
#     --do_train \
#     --do_eval \
#     --learning_rate 5e-5 \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.01 \
#     --low_cpu_mem_usage True \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --run_name codellama-7b-${degree}-mod \
#     --resume_from_checkpoint /data/kdy20401/Workspace/Proj-Code-Generation/MC/tmp/CodeLlama/checkpoint-5 \

    

# --max_train_samples 50 \


# python my_run_clm.py \
#     --model_name_or_path meta-llama/CodeLlama-7b-hf \
#     --train_file data/ft_final/my_code_contests_train_low.jsonl \
#     --validation_file data/ft_final/my_code_contests_valid_low.jsonl \
#     --output_dir tmp/CodeLlama \
#     --save_steps 60 \
#     --evaluation_strategy steps \
#     --max_train_samples 50 \
#     --max_eval_samples 1 \
#     --torch_dtype bfloat16 \
#     --block_size 2048 \
#     --preprocessing_num_workers 8 \
#     --trust_remote_code 1 \
#     --do_train \
#     --do_eval \
#     --learning_rate 5e-5 \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 1 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.01 \
#     --low_cpu_mem_usage True \
#     --overwrite_output_dir True \
#     --report_to wandb \
#     --run_name codellama-7b-low-mod \
    

    


# # --max_eval_samples 50 \
# # --logging_steps 20 \

