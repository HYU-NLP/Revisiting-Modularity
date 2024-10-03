#/bin/bash

temperature=0.1
code_type_=(mc sc tmc tsc)
model_name=$1
if [ ${model_name} == deepseek ]; then
    model=deepseek-ai/deepseek-coder-6.7b-base
else
    model=meta-llama/CodeLlama-7b-hf
fi

task() {
    local seed=$1
    for code_type in ${code_type_}; do
        python -u eval.py  --seed ${seed} \
        --model ${model} --num_icl_shot 2 \
        --num_gen 10 --code_type ${code_type} \
        --temperature ${temperature}  --modify original \
        > log/evaluation/meta-llama-CodeLlama-7b-hf_${codetype}_original_${num_icl_shot}shot_10gen_${temperature}temp_${seed}.log 2>&1
    done
    task_completed $seed
}

task_completed() {
    local seed=$1
    # Start task1 for the next seed
    next_seed=$(next_seed $seed)
    if [ -n "$next_seed" ]; then
        task1 $next_seed &
    fi
}


next_seed() {
    local seed=$1
    case $seed in
        27) echo 42 ;;
        42) echo 101 ;;
        101) echo 134 ;;
        134) echo 169 ;;
        169) echo "" ;;
    esac
}


# Start the first tasks
task 27 &

# Wait for all background jobs to finish
wait