#/bin/bash

# CL
size=7
model=meta-llama/CodeLlama-${size}b-hf
num_icl_shot=1
num_gen=10
temperature=0.1
k=1

for metric in var_len; do
    python evaluate_corr.py \
    --model ${model} \
    --num_icl_shot ${num_icl_shot} \
    --num_gen ${num_gen} \
    --temperature ${temperature} \
    --metric ${metric} \
    --k ${k} \
    > log/evaluation/cl${size}b_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${metric}_corr.log 2>&1
    echo cl${size}b ${metric} score correlation evaluation ends
done


# # DS
# size=6.7
# model=deepseek-ai/deepseek-coder-${size}b-base
# num_gpu=1
# num_icl_shot=1
# num_gen=10
# temperature=0.1
# k=1

# for metric in style modularity; do
#     python evaluate_corr.py \
#     --model ${model} \
#     --num_icl_shot ${num_icl_shot} \
#     --num_gen ${num_gen} \
#     --temperature ${temperature} \
#     --metric ${metric} \
#     --k ${k} \
#     > log/evaluation/ds${size}b_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${metric}_corr.log 2>&1
#     echo ds${size}b ${metric} score correlation evaluation ends
# done