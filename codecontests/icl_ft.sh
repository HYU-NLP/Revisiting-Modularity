#/bin/bash

# num_gpu=4
# dtype=float16
# num_icl_shot=2
# num_gen=50
# temperature=0.6
# swap_space=8
# code_type=modular

# for size in 34; do
#     for seed in 27 42 101 134 169; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python icl.py \
#         --seed ${seed} \
#         --model meta-llama/CodeLlama-${size}b-hf \
#         --num_gpu ${num_gpu} \
#         --dtype ${dtype} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --top_p 0.95 \
#         --swap_space ${swap_space} \
#         --code_type ${code_type} \
#         > log/cl${size}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo cl${size} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done 

# # CL 7b, pass@1(n=10)
# num_gpu=4
# dtype=float16
# num_icl_shot=2
# num_gen=10
# temperature=0.1
# swap_space=8
# code_type=monolithic

# for size in 34; do
#     for seed in 27 42 101 134 169; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python icl.py \
#         --seed ${seed} \
#         --model meta-llama/CodeLlama-${size}b-hf \
#         --num_gpu ${num_gpu} \
#         --dtype ${dtype} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --top_p 0.95 \
#         --swap_space ${swap_space} \
#         --code_type ${code_type} \
#         > log/inference/2shot_mc/cl${size}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo cl${size} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done    


# # DS
# num_gpu=4
# dtype=bfloat16
# num_icl_shot=2
# num_gen=50
# temperature=0.1
# swap_space=8
# code_type=modular

# for size in 33; do
#     for seed in 27 42 101 134 169; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python icl.py \
#         --seed ${seed} \
#         --model deepseek-ai/deepseek-coder-${size}b-base \
#         --num_gpu ${num_gpu} \
#         --dtype ${dtype} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --top_p 0.95 \
#         --swap_space ${swap_space} \
#         --code_type ${code_type} \
#         > log/inference/2shot_mc/ds${size}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ds${size} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done    


# num_gpu=4
# dtype=float16
# num_icl_shot=2
# num_gen=50
# temperature=0.6
# swap_space=8
# code_type=modular

# for size in 34; do
#     for seed in 27 42 101 134 169; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python icl.py \
#         --seed ${seed} \
#         --model meta-llama/CodeLlama-${size}b-hf \
#         --num_gpu ${num_gpu} \
#         --dtype ${dtype} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --top_p 0.95 \
#         --swap_space ${swap_space} \
#         --code_type ${code_type} \
#         > log/inference/2shot_mc/cl${size}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo cl${size} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done    

# # DS
# num_gpu=4
# dtype=bfloat16
# num_icl_shot=2
# num_gen=50
# temperature=0.6
# swap_space=8
# code_type=modular

# for size in 33; do
#     for seed in 27 42 101 134 169; do
#         CUDA_VISIBLE_DEVICES=0,1,2,3 python icl.py \
#         --seed ${seed} \
#         --model deepseek-ai/deepseek-coder-${size}b-base \
#         --num_gpu ${num_gpu} \
#         --dtype ${dtype} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --top_p 0.95 \
#         --swap_space ${swap_space} \
#         --code_type ${code_type} \
#         > log/inference/2shot_mc/ds${size}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ds${size} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done


# inference from ft checkpoint, pass@1(n=10)
seed=27
model=deepseek
num_gpu=2
dtype=float16
num_icl_shot=0
temperature=0.6
code_type=monolithic
swap_space=16
chkpt=_final
num_gen=50 #
debug_mode=0 #

# for low and high model simultaneously
degree=low
CUDA_VISIBLE_DEVICES=0,1 nohup python icl_ft.py \
--seed ${seed} \
--model /data/kdy20401/Workspace/Proj-Code-Generation/MC/tmp/${model}/${degree}/ \
--num_gpu ${num_gpu} \
--dtype ${dtype} \
--num_icl_shot ${num_icl_shot} \
--num_gen ${num_gen} \
--temperature ${temperature} \
--max_new_token 1024 \
--top_p 0.95 \
--swap_space ${swap_space} \
--code_type ${code_type} \
--degree ${degree} \
--debug_mode ${debug_mode} \
--chkpt ${chkpt} \
> log/inference/tmp/${model}_${degree}_mod_chkpt${chkpt}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1 &

degree=high
CUDA_VISIBLE_DEVICES=2,3 nohup python icl_ft.py \
--seed ${seed} \
--model /data/kdy20401/Workspace/Proj-Code-Generation/MC/tmp/${model}/${degree}/ \
--num_gpu ${num_gpu} \
--dtype ${dtype} \
--num_icl_shot ${num_icl_shot} \
--num_gen ${num_gen} \
--temperature ${temperature} \
--max_new_token 1024 \
--top_p 0.95 \
--swap_space ${swap_space} \
--code_type ${code_type} \
--degree ${degree} \
--debug_mode ${debug_mode} \
--chkpt ${chkpt} \
> log/inference/tmp/${model}_${degree}_mod_chkpt${chkpt}_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1 &
wait &&
echo done!