(nohup python perplexity.py --gpu 0 --model meta-llama/CodeLlama-7b-hf --mod low --include_prompt > log/ppl_include_prompt/cl7b_low_mod.log 2>&1) &
(nohup python perplexity.py --gpu 1 --model meta-llama/CodeLlama-7b-hf --mod high --include_prompt > log/ppl_include_prompt/cl7b_high_mod.log 2>&1) &
(nohup python perplexity.py --gpu 2 --model deepseek-ai/deepseek-coder-6.7b-base --mod low --include_prompt > log/ppl_include_prompt/ds7b_low_mod.log 2>&1) &
(nohup python perplexity.py --gpu 3 --model deepseek-ai/deepseek-coder-6.7b-base --mod high --include_prompt > log/ppl_include_prompt/ds7b_high_mod.log 2>&1) &
wait &&
echo 7b model done!
(nohup python perplexity.py --gpu 0 --model meta-llama/CodeLlama-34b-hf --mod low --include_prompt > log/ppl_include_prompt/cl34b_low_mod.log 2>&1) &
(nohup python perplexity.py --gpu 1 --model meta-llama/CodeLlama-34b-hf --mod high --include_prompt > log/ppl_include_prompt/cl34b_high_mod.log 2>&1) &
(nohup python perplexity.py --gpu 2 --model deepseek-ai/deepseek-coder-33b-base --mod low --include_prompt > log/ppl_include_prompt/ds33b_low_mod.log 2>&1) &
(nohup python perplexity.py --gpu 3 --model deepseek-ai/deepseek-coder-33b-base --mod high --include_prompt > log/ppl_include_prompt/ds33b_high_mod.log 2>&1) &
echo 33b model in progress!