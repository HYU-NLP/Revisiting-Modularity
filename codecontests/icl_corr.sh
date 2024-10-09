#/bin/bash

# inference for correlation experiment
# CL 7b, pass@1(n=10)
seed=42
size=7
model=meta-llama/CodeLlama-${size}b-hf
# size=6.7
# model=deepseek-ai/deepseek-coder-${size}b-base
num_gpu=4
dtype=float16
num_icl_shot=1
num_gen=10
temperature=0.1
swap_space=8
for metric in var_len; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 python icl_corr.py \
    --seed ${seed} \
    --model ${model} \
    --num_gpu ${num_gpu} \
    --dtype ${dtype} \
    --num_icl_shot ${num_icl_shot} \
    --num_gen ${num_gen} \
    --temperature ${temperature} \
    --max_new_token 1024 \
    --top_p 0.95 \
    --swap_space ${swap_space} \
    --metric ${metric} \
    > log/inference/cl${size}b_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${metric}.log 2>&1
    echo cl${size}b ${metric} inference ends
done

