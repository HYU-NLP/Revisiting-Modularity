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
    if "meta-llama/CodeLlama" in args.model:
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            start_index = generation.find("```")
            if start_index == -1:
                solution = ""
            else:
                end_index = generation.find("```", start_index + len("```"))
                if start_index < end_index:
                    solution = generation[start_index + len("```") : end_index]
                else:
                    solution = ""

    elif "deepseek-ai/deepseek-coder" in args.model:
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            start_index = generation.find("```")
            if start_index == -1:
                solution = ""
            else:
                end_index = generation.find("```", start_index + len("```"))
                if start_index < end_index:
                    solution = generation[start_index + len("```") : end_index]
                else:
                    solution = ""

    return solution


def make_prompt(args, demonstration, test_data):
    instruction = (
        "Write a python code to solve the following coding problem "
        "that obeys the constraints and passes the example test cases. "
        "The output code needs to read from and write to standard IO. "
        "Please wrap your code answer using ```:"
    )
    
    if "meta-llama/CodeLlama" in args.model:
        # make zero-shot or few-shot prompt
        prompt = ""
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            for i in range(args.num_icl_shot):
                prompt += "Q: " + instruction + "\n"
                prompt += demonstration["description"][i] + "\n"
                prompt += "A: " + "```" + demonstration["code"][i] + "```" + "\n"
            prompt += "Q: " + instruction + "\n"
            prompt += test_data["description"] + "\n"
            prompt += "A: "
    elif "deepseek-ai/deepseek-coder" in args.model:
        # make zero-shot or few-shot prompt
        prompt = ""
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            prompt += instruction + "\n"
            for i in range(args.num_icl_shot):
                prompt += "### Instruction:\n" + demonstration["description"][i] + "\n"
                prompt += (
                    "### Response:\n" + "```" + demonstration["code"][i] + "```" + "\n"
                )
            prompt += "### Instruction:\n" + test_data["description"] + "\n"
            prompt += "### Response:\n"

    return prompt


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
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        default='style',
        help="code metric (e.g., style or modularity)",
    )
    # additional arguments candidiates:
    # max_model_len
    # stop
    # start_token, end_token
    args = parser.parse_args()
    
    # this code is impelemented for only 1-shot ICL
    assert args.num_icl_shot == 1
    
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
    
    if "meta-llama/CodeLlama" in args.model:
        stop = ["Q:", "A:"]
    elif "deepseek-ai/deepseek-coder" in args.model:
        stop = ["### Instruction", "### Response"]
        
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
        cache_dir="/data/huggingface/datasets",
    )
        
    # set seed
    set_seed(args.seed)

    base_directory = os.path.dirname(__file__)
    
    # demonstration pool constructed by style or modularity
    demonstration_dataset = read_jsonl_to_dict(
        os.path.join(
            base_directory,
            "data",
            f"{args.metric}_demonstration.jsonl",
        )  
    )
    assert len(demonstration_dataset) == 100
    
    # iterate over codes in the demonstration
    # make 1-shot prompt using the code and estimate pass@k
    for code_idx, data in enumerate(demonstration_dataset):
        if data['var_len'] < 5:
            continue
        print(f'average variable length: {data["var_len"]}')
        file_name = f"{args.model.replace('/', '-')}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.metric}_{code_idx}code_icl_result.jsonl"
        if not os.path.exists(os.path.join(base_directory, "result", file_name)):
            print(file_name)
            description = data['description']
            code = data['code']
            score_style = data['score_style'] # 'score_pep8', 'score_var', 'score_style'
            score_modularity = data['score_modularity']
            
            # make demonstration for each code (1-shot)
            demonstration = defaultdict(list)
            demonstration['description'].append(description.strip())
            demonstration["code"].append(code.strip())
            demonstration['score_style'].append(score_style)
            demonstration['score_modularity'].append(score_modularity)
            demonstration['var_len'].append(data['var_len'])
            
            # make prompt for each test data
            prompts = []
            # test_dataset = list(test_dataset)[:1] # for test
            for test_data in test_dataset:
                prompt = make_prompt(args, demonstration, test_data)
                prompts.append(prompt)

            # inference using vllm
            generations = []
            solutions = []
            # generate solution code using vllm
            print(f'<inference with {code_idx}th demonstration code starts>')
            
            outputs = model.generate(
                prompts, sampling_params=sampling_params, use_tqdm=True
            )
            for output in outputs:
                # for each input in the prompts, args.gen_num number of outputs are generated
                generations_ = [outs.text for outs in output.outputs]
                assert len(generations_) == args.num_gen
                # extract solution code from generated code
                solutions_ = [
                    extract_solution(args, generation) for generation in generations_
                ]
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
                        "demonstration": demonstration, # contains code and its metric scores
                    }
                )
                
            write_dict_to_jsonl(result, os.path.join(base_directory, "result", file_name))
            
    print(f'program ends.')


if __name__ == "__main__":
    main()
