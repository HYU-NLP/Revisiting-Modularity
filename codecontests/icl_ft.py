import os
import random
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import torch
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils.utils import read_jsonl_to_dict, write_dict_to_jsonl, get_code_style_score
from utils.utils import get_code_modularity_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)     
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_solution(args, generation):
    if "CodeLlama" in args.model:
        # start_index = generation.find("```")
        # if start_index == -1:
        #     solution = ""
        # else:
        #     end_index = generation.find("```", start_index + len("```"))
        #     if start_index < end_index:
        #         solution = generation[start_index + len("```") : end_index]
        #     else:
        #         solution = ""
        idx = generation.find('```')
        if idx != -1:
            solution = generation[:idx]
        else:
            solution = generation.strip()

    elif "deepseek"  in args.model:
        idx = generation.find('```')
        if idx != -1:
            solution = generation[:idx]
        else:
            solution = generation.strip()

    return solution


def make_prompt(args, demonstration, test_data):
    instruction = (
        "Write a python code to solve the following coding problem "
        "that obeys the constraints and passes the example test cases. "
        "The output code needs to read from and write to standard IO. "
        "Please wrap your code answer using ```:"
    )
    
    if "CodeLlama" in args.model:
        # make zero-shot or few-shot prompt
        prompt = ""
        for i in range(args.num_icl_shot):
            prompt += "Q: " + instruction + "\n"
            prompt += demonstration["description"][i] + "\n"
            prompt += "A: " + "```" + demonstration["code"][i] + "```" + "\n"
        prompt += "Q: " + instruction + "\n"
        prompt += test_data["description"].strip() + "\n"
        prompt += "A: ```"
    elif "deepseek" in args.model:
        prompt = ""
        for i in range(args.num_icl_shot):
            prompt += "Q: " + instruction + "\n"
            prompt += demonstration["description"][i] + "\n"
            prompt += "A: " + "```" + demonstration["code"][i] + "```" + "\n"
        prompt += "Q: " + instruction + "\n"
        prompt += test_data["description"].strip() + "\n"
        prompt += "A: ```"
        
        # # make zero-shot or few-shot prompt
        # prompt = ""
        # prompt += instruction + "\n"
        # for i in range(args.num_icl_shot):
        #     prompt += "### Instruction:\n" + demonstration["description"][i] + "\n"
        #     prompt += (
        #         "### Response:\n" + "```" + demonstration["code"][i] + "```" + "\n"
        #     )
        # prompt += "### Instruction:\n" + test_data["description"].strip() + "\n"
        # prompt += "### Response:\n"

    return prompt


def extract_demonstration(train_dataset, shot, code_type):
    if 'transformed' not in code_type:
        problem_index_with_both_sc_and_mc = []
        for i, data in enumerate(train_dataset):
            num_sc = len(data['monolithic_codes']['monolithic_code'])
            num_mc = len(data['modular_codes']['modular_code']) 
            if num_sc > 0 and num_mc > 0:
                problem_index_with_both_sc_and_mc.append(i)

        demonstration = defaultdict(list)
        for i in random.sample(problem_index_with_both_sc_and_mc, shot):
            data = train_dataset[i]
            # modularity check
            # print(f'problem {i}')
            # tmp = []
            # for code in data['modular_codes']['modular_code']:
                # modularity = get_code_modularity_score(code)
                # tmp.append(modularity)
            # print(tmp)        
            if code_type == 'monolithic':
                demonstration['description'].append(data['problem_description'].strip())
                demonstration['code'].append(data['monolithic_codes']['monolithic_code'][0].strip()) # pick the first code
                # print(get_code_modularity_score(data['monolithic_codes']['monolithic_code'][0]))
            elif code_type == 'modular':
                demonstration['description'].append(data['problem_description'].strip())
                demonstration['code'].append(data['modular_codes']['modular_code'][0].strip())
                print(get_code_modularity_score(data['modular_codes']['modular_code'][0]))
                print(data['modular_codes']['modular_code'][0])

        return demonstration
    
    else:
        if code_type == 'transformed_modular':
            key = 'transformed_mc'
        elif code_type == 'transformed_monolithic':
            key = 'transformed_sc'

        demonstration = defaultdict(list)
        for i in range(shot):
            demonstration['description'].append(dataset['problem_description'][i].strip())
            demonstration['code'].append(dataset[key][i].strip())

        return demonstration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, default=42)
    parser.add_argument(
        "--model", type=str, required=True, default="meta-llama/CodeLlama-7b-hf"
    )
    parser.add_argument("--num_gpu", type=int, required=True, default=1, help="total number of gpus used")
    parser.add_argument("--dtype", type=str, required=True, default="float16")
    parser.add_argument("--num_icl_shot", type=int, required=True, default=2)
    parser.add_argument(
        "--num_gen",
        type=int,
        required=True,
        default=1,
        help="number of solutions generated per problem",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        default=0,
        help="0 means greedy decoding for vllm",
    )
    parser.add_argument("--max_new_token", type=int, required=True, default=1024)
    parser.add_argument("--top_p", type=float, required=True, default=0.95)
    parser.add_argument(
        "--swap_space",
        type=int,
        required=False,
        default=4,
        help="The size (GiB) of CPU memory per GPU to use as swap space",
    )
    parser.add_argument('--code_type', type=str, required=True, default='monolithic')
    parser.add_argument('--degree', type=str, required=True, default='low')
    parser.add_argument('--debug_mode', type=int, required=True, default=0)
    parser.add_argument('--chkpt', type=str, required=True, default=0)
    # additional arguments candidiates:
    # max_model_len
    # stop
    # start_token, end_token
    args = parser.parse_args()

    # load model
    # when initializing VLLM engine, random.seed() is called internally.
    # so, set_seed() should be called after initializing VLLM engine.
    model = LLM(
        model=args.model,
        tensor_parallel_size=args.num_gpu,
        dtype=args.dtype,
        max_model_len=8192,
        swap_space=args.swap_space,
    )
    
    # all models are fine-tuned with "Q:,, A:,," format
    stop = ["Q:", "A:"]
        
    sampling_params = SamplingParams(
        n=args.num_gen,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_token,
        stop=stop,
    )

    # load code contest test dataset
    test_dataset = load_dataset(
        "deepmind/code_contests",
        split="test",
    )
        
    # set seed
    set_seed(args.seed)

    base_directory = os.path.dirname(__file__)
    
    # monolithic(sc) or modular(mc) demonstration
    if 'transformed' not in args.code_type: 
        dataset = read_jsonl_to_dict(os.path.join(os.path.dirname(__file__), 'data', 'my_code_contests_divided_train.jsonl'))
    # transformed monolithic(tsc) or transformed modular(tmc) demonstration
    else:
        dataset = read_jsonl_to_dict(os.path.join(os.path.dirname(__file__), 'data', f'monolithic_2shot_demonstration_{args.seed}seed.jsonl'))[0]
    
    demonstration = extract_demonstration(dataset, args.num_icl_shot, args.code_type)
    
    if "CodeLlama" in args.model:
        file_name = f"CodeLlama_{args.degree}_mod_chkpt{args.chkpt}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.seed}seed_icl_result.jsonl"
    elif "deepseek" in args.model:
        file_name = f"DeepSeek_{args.degree}_mod_chkpt{args.chkpt}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.seed}seed_icl_result.jsonl"

    
    if os.path.exists(os.path.join(base_directory, "result/ft", file_name)):
        print(f'{file_name} already exists.')
        return

    # make prompt for each test data
    if args.debug_mode:
        test_dataset = list(test_dataset)[:10] # for test

    prompts = []
    for test_data in test_dataset:
        prompt = make_prompt(args, demonstration, test_data)
        prompts.append(prompt)
        
    # inference using vllm
    generations = []
    solutions = []
    
    # generate solution code using vllm
    outputs = model.generate(
        prompts, sampling_params=sampling_params, use_tqdm=True
    )
    for idx, output in enumerate(outputs):
        # for each input in the prompts, args.gen_num number of outputs are generated
        generations_ = [outs.text.strip() for outs in output.outputs]
        assert len(generations_) == args.num_gen
        # extract solution code from generated code
        solutions_ = [
            extract_solution(args, generation) for generation in generations_
        ]
        if args.debug_mode:
            print(f'problem {idx}, prompt:')
            print(prompts[idx])
            print('-' * 100)
            print('generation:')
            print(generations_[0].strip())
            print('-' * 100)
            print('solution:')
            print(solutions_[0].strip())
            print('*' * 100)
        
        # save generated solutions (list)
        generations.append(generations_)
        solutions.append(solutions_)
        
    # save generated solutions
    result = []
    for i, test_data in enumerate(test_dataset):
        result.append(
            {
                "name": test_data["name"],
                "description": test_data["description"],
                "public_tests": test_data["public_tests"],
                "private_tests": test_data["private_tests"],
                "difficulty": test_data["difficulty"],
                "cf_rating": test_data["cf_rating"], # difficulty level
                "generated_solutions": generations[i], # list of generated solutions
                "extracted_solutions": solutions[i],
                "prompt": prompts[i],
                "demonstration": demonstration, # contains code and its description
            }
        )
    
    if not args.debug_mode:
        write_dict_to_jsonl(result, os.path.join(base_directory, "result/ft", file_name))
    print(f'program ends.')


if __name__ == "__main__":
    main()
