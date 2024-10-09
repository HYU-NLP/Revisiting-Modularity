#/bin/bash

# # CL
# num_icl_shot=0
# num_gen=1
# temperature=0.1
# k=1

# for size in 34; do
#     for seed in 27 42 101 134 169; do
#         python evaluate_.py \
#         --model meta-llama/CodeLlama-${size}b-hf \
#         --seed ${seed} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --k ${k} \
#         > log/evaluation/2shot_mc/cl${size}b_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo cl${size}b ${num_icl_shot} mc ${seed}seed evaluation ends
#     done
# done

# # DS
# num_icl_shot=2
# num_gen=50
# temperature=0.6
# k=10
# for size in 33; do
#     for seed in 27 42 101 134 169; do
#         python evaluate_.py \
#         --model deepseek-ai-deepseek-coder-${size}b-base \
#         --seed ${seed} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --k ${k} \
#         > log/evaluation/2shot_mc/ds${size}b_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ds${size}b ${num_icl_shot} mc ${seed}seed evaluation ends
#     done
# done


# inference after fine-tuning
num_icl_shot=0
num_gen=50
temperature=0.6
k=1
# degree=low
debug_mode=0
chkpt=_final

for degree in low high; do
    for seed in 27; do
        python evaluate_ft.py \
        --model meta-llama/CodeLlama-7b-hf \
        --seed ${seed} \
        --num_icl_shot ${num_icl_shot} \
        --num_gen ${num_gen} \
        --temperature ${temperature} \
        --k ${k} \
        --degree ${degree} \
        --chkpt ${chkpt} \
        > log/evaluation/tmp/CodeLlama_${degree}_mod_chkpt${chkpt}_${num_icl_shot}shot_${temperature}temp_${num_gen}gen_${seed}.log 2>&1 &
    done
done

# --model meta-llama/CodeLlama-7b-hf \
# --model /data/kdy20401/Workspace/Proj-Code-Generation/MC/tmp


# # gpt
# num_icl_shot=2
# num_gen=10
# temperature=0.1
# k=1

# for code_type in monolithic modular transformed_modular transformed_monolithic; do
#     for seed in 134; do
#         python evaluate_gpt.py \
#         --model gpt-4o-mini \
#         --code_type ${code_type} \
#         --seed ${seed} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --k ${k} \
#         >> log/evaluation/gpt/gpt-4o-mini_${code_type}_code_${num_icl_shot}shot_${temperature}temp_${num_gen}gen.log 2>&1
#     done
# done
