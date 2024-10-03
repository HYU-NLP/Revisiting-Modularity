#/bin/bash

num_icl_shot=2
code_type = $2

if [ $1 == deepseek ]; then
    model=deepseek-ai/deepseek-coder-6.7b-base
else
    model=meta-llama/CodeLlama-7b-hf
fi

task0() {
    local seed=$1
    llama_size=7
    llama_model=meta-llama/CodeLlama-${llama_size}b-hf
    CUDA_VISIBLE_DEVICES=0,1 python -u icl.py \
    --seed $seed --model ${llama_model} \
    --num_gpu ${num_gpu} --dtype float16 --num_icl_shot ${num_icl_shot} \
    --num_gen 10 --code_type mc \
    --temperature ${temperature} --max_new_token 1024 \
    --top_p 0.95 --modify original --swap_space ${swap_space} \
    > log/codellama_${llama_size}b_${num_icl_shot}shot_${temperature}temp_mc_${seed}.log 2>&1
    task1_completed $seed
}

task1() {
    local seed=$1
    llama_size=7
    llama_model=meta-llama/CodeLlama-${llama_size}b-hf
    CUDA_VISIBLE_DEVICES=0,1 python -u icl.py \
    --seed $seed --model ${llama_model} \
    --num_gpu ${num_gpu} --dtype float16 --num_icl_shot ${num_icl_shot} \
    --num_gen 10 --code_type mc \
    --temperature ${temperature} --max_new_token 1024 \
    --top_p 0.95 --modify original --swap_space ${swap_space} \
    > log/codellama_${llama_size}b_${num_icl_shot}shot_${temperature}temp_mc_${seed}.log 2>&1
    task1_completed $seed
}
task2() {
    local seed=$1
    deepseek_size=6.7
    deepseek_model=deepseek-ai/deepseek-coder-${deepseek_size}b-base
    CUDA_VISIBLE_DEVICES=0,1 python -u icl.py \
    --seed $seed --model ${deepseek_model} \
    --num_gpu ${num_gpu} --dtype bfloat16 --num_icl_shot ${num_icl_shot} \
    --num_gen 10 --code_type mc \
    --temperature ${temperature} --max_new_token 1024 \
    --top_p 0.95 --modify original --swap_space ${swap_space} \
    > log/deepseek_${deepseek_size}b_${num_icl_shot}shot_${temperature}temp_mc_${seed}.log 2>&1
    task2_completed $seed
}
task3() {
    local seed=$1
    llama_size=7
    llama_model=meta-llama/CodeLlama-${llama_size}b-hf
    CUDA_VISIBLE_DEVICES=2,3 python -u icl.py \
    --seed $seed --model ${llama_model} \
    --num_gpu ${num_gpu} --dtype float16 --num_icl_shot ${num_icl_shot} \
    --num_gen 10 --code_type sc \
    --temperature ${temperature} --max_new_token 1024 \
    --top_p 0.95 --modify original --swap_space ${swap_space} \
    > log/codellama_${llama_size}b_${num_icl_shot}shot_${temperature}temp_sc_${seed}.log 2>&1
    task3_completed $seed
}

task4() {
    local seed=$1
    deepseek_size=6.7
    deepseek_model=deepseek-ai/deepseek-coder-${deepseek_size}b-base
    CUDA_VISIBLE_DEVICES=3 python -u icl.py \
    --seed $seed --model ${deepseek_model} \
    --num_gpu ${num_gpu} --dtype bfloat16 --num_icl_shot ${num_icl_shot} \
    --num_gen 10 --code_type sc \
    --temperature ${temperature} --max_new_token 1024 \
    --top_p 0.95 --modify original --swap_space ${swap_space} \
    > log/deepseek_${deepseek_size}b_${num_icl_shot}shot_${temperature}temp_sc_${seed}.log 2>&1
    task4_completed $seed
}
task1_completed() {
    local seed=$1
    # Start task1 for the next seed
    next_seed=$(next_seed $seed)
    if [ -n "$next_seed" ]; then
        task1 $next_seed &
    fi
}

task2_completed() {
    local seed=$1
    # Start task2 for the next seed
    next_seed=$(next_seed $seed)
    if [ -n "$next_seed" ]; then
        task2 $next_seed &
    fi
}

task3_completed() {
    local seed=$1
    # Start task1 for the next seed
    next_seed=$(next_seed $seed)
    if [ -n "$next_seed" ]; then
        task3 $next_seed &
    fi
}

task4_completed() {
    local seed=$1
    # Start task2 for the next seed
    next_seed=$(next_seed $seed)
    if [ -n "$next_seed" ]; then
        task4 $next_seed &
    fi
}

next_seed() {
    local seed=$1
    case $seed in
        27) echo 42 ;;
        42) echo 101 ;;
        101) echo "" ;;
        169) echo "" ;;
    esac
}

temperature=0.1
num_gpu=2
swap_space=$((64/num_gpu))
num_icl_shot=2


# Start the first tasks
task0 134 &
# task1 27 &
# task3 27 &
# task2 27 &
# task4 27 &

# Wait for all background jobs to finish
wait