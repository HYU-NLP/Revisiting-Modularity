# lets go
model=gpt-4o-mini
num_icl_shot=2
num_gen=10
temperature=0.1
debug_mode=0

# for code_type in monolithic; do
#     for seed in 27 42 101 134; do
#         python icl_gpt.py \
#         --seed ${seed} \
#         --model ${model} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --code_type ${code_type} \
#         --debug_mode ${debug_mode} \
#         > log/inference/gpt/gpt-4o-mini_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ${model} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done

# for code_type in modular; do
#     for seed in 27 42 101 134 169; do
#         python icl_gpt.py \
#         --seed ${seed} \
#         --model ${model} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --code_type ${code_type} \
#         --debug_mode ${debug_mode} \
#         > log/inference/gpt/gpt-4o-mini_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ${model} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done

# for code_type in transformed_monolithic; do
#     for seed in 27 42 101 134 169; do
#         python icl_gpt.py \
#         --seed ${seed} \
#         --model ${model} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --code_type ${code_type} \
#         --debug_mode ${debug_mode} \
#         > log/inference/gpt/gpt-4o-mini_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ${model} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done

for code_type in transformed_modular; do
    for seed in 27 42 134 169; do
        python icl_gpt.py \
        --seed ${seed} \
        --model ${model} \
        --num_icl_shot ${num_icl_shot} \
        --num_gen ${num_gen} \
        --temperature ${temperature} \
        --max_new_token 1024 \
        --code_type ${code_type} \
        --debug_mode ${debug_mode} \
        > log/inference/gpt/gpt-4o-mini_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
        echo ${model} ${num_icl_shot}shot ${code_type} inference ${seed} ends
    done
done


# # # for test
# model=gpt-4o-mini
# num_icl_shot=2
# num_gen=10
# temperature=0.1
# debug_mode=0

# for code_type in transformed_modular; do
#     for seed in 101; do
#         python icl_gpt.py \
#         --seed ${seed} \
#         --model ${model} \
#         --num_icl_shot ${num_icl_shot} \
#         --num_gen ${num_gen} \
#         --temperature ${temperature} \
#         --max_new_token 1024 \
#         --code_type ${code_type} \
#         --debug_mode ${debug_mode} \
#         > log/inference/gpt/gpt-4o-mini_${num_icl_shot}shot_${code_type}_${temperature}temp_${num_gen}gen_${seed}.log 2>&1
#         echo ${model} ${num_icl_shot}shot ${code_type} inference ${seed} ends
#     done
# done