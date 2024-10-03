import os
import random
import argparse
from tqdm import tqdm

import numpy as np
from collections import defaultdict

import torch

from datasets import load_dataset, Dataset

from vllm import LLM, SamplingParams

from utils import read_jsonl_to_dict, write_dict_to_jsonl, get_avg_cc

import sys


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


def get_transformed_demonstration(args, data):
    demonstration = defaultdict(list)

    for i in range(args.num_icl_shot):
        if "sc" in args.code_type:
            instruction = data["sc_instruction"][i]
        else:
            instruction = data["mc_instruction"][i]

        if "transformed" in args.code_type:
            code = data[args.code_type][i][0].strip()
        else:
            code = data[args.code_type][i].strip()

        demonstration["problem_id"].append(data["problem_id"][i])
        demonstration["description"].append(data["problem_description"][i].strip())
        demonstration["instruction"].append(instruction)
        demonstration["starter_code"].append(data["starter_code"][i])
        demonstration["code"].append(code)
        demonstration["code_cc"].append(get_avg_cc(data[args.code_type][i]))

    return demonstration


def extract_solution(args, generation):
    if args.num_icl_shot > 0:
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
    if test_data["starter_code"] == "":
        question_guide = "read from and write to standard IO"
    else:
        question_guide = "use the provided function signature"

    if "sc" in args.code_type:
        instruction = (
            "Write a python code to solve the following coding problem "
            "that obeys the constraints and passes the example test cases. "
            f"The output code needs to {question_guide}. "
            "Please wrap your code answer using ```:"
        )
    elif "mc" in args.code_type:
        instruction = (
            "Write a python code to solve the following coding problem "
            "that obeys the constraints and passes the example test cases. "
            f"The output code needs to {question_guide}. "
            "Ensure modularity of the python code by dividing the code into smaller, "
            "useful functions to solve the given problem. "
            "Please wrap your code answer using ```:"
        )

    # instruction of CodeLlama for APPS
    if "meta-llama/CodeLlama" in args.model:
        # make zero-shot or few-shot prompt
        prompt = ""
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            for i in range(args.num_icl_shot):
                prompt += "Q: " + demonstration["instruction"][i] + "\n"
                prompt += demonstration["description"][i] + "\n"
                prompt += demonstration["starter_code"][i] + "\n"
                prompt += "A: " + "```" + demonstration["code"][i] + "```" + "\n"
            prompt += "Q: " + instruction + "\n"
            prompt += test_data["question"] + "\n"
            prompt += test_data["starter_code"] + "\n"
            prompt += "A: "

    # instruction of DeepseekCoder for APPS
    elif "deepseek-ai/deepseek-coder" in args.model:
        # make zero-shot or few-shot prompt
        prompt = ""
        if args.num_icl_shot == 0:
            assert ()  # not implemented yet
        elif args.num_icl_shot > 0:
            for i in range(args.num_icl_shot):
                prompt += demonstration["instruction"][i] + "\n"
                prompt += "### Instruction:\n" + demonstration["description"][i] + "\n"
                prompt += demonstration["starter_code"][i] + "\n"
                prompt += (
                    "### Response:\n" + "```" + demonstration["code"][i] + "```" + "\n"
                )
            prompt += instruction + "\n"
            prompt += "### Instruction:\n" + test_data["question"] + "\n"
            prompt += test_data["starter_code"] + "\n"
            prompt += "### Response:\n"

    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/CodeLlama-7b-hf")
    parser.add_argument("--num_gpu", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_icl_shot", type=int, default=2)
    parser.add_argument(
        "--num_gen",
        type=int,
        default=1,
        help="number of solutions generated per problem",
    )
    parser.add_argument("--code_type", type=str, default="sc")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="0 means greedy decoding for vllm",
    )
    parser.add_argument("--max_new_token", type=int, default=1024)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--modify",
        type=str,
        default="original",
        help="modification method of the demonstration code",
    )
    parser.add_argument(
        "--swap_space",
        type=int,
        default=4,
        help="The size (GiB) of CPU memory per GPU to use as swap space",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    base_directory = os.path.dirname(__file__)
    if not os.path.exists(os.path.join(base_directory, "result")):
        os.makedirs(os.path.join(base_directory, "result"))
    file_name = f"{args.model.replace('/', '-')}_{args.code_type}_{args.modify}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.seed}seed_icl_result.jsonl"

    data = Dataset.from_json(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            f"2shot_demonstration_{args.seed}seed_reduced2.json",
        )
    )

    demonstration = get_transformed_demonstration(args, data)

    # load apps test dataset
    test_dataset = load_dataset("codeparrot/apps", split="test", trust_remote_code=True)
    # filtering for specific platforms
    words = ["codeforces", "atcoder", "codechef"]
    test_dataset = test_dataset.filter(
        lambda x: any(word in x["url"] for word in words)
    )

    prompts = []
    for test_data in test_dataset:
        prompt = make_prompt(args, demonstration, test_data)
        prompts.append(prompt)

    if os.path.exists(os.path.join(base_directory, "result", file_name)):
        results = read_jsonl_to_dict(os.path.join(base_directory, "result", file_name))
        start_index = len(results)
        if not start_index == len(prompts):
            prompts = prompts[start_index:]
        else:
            print("All problems are already solved.")
            sys.exit()

    # load model
    # when initializing VLLM engine, random.seed() is called internally.
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

    # inference using vllm
    generations = []
    solutions = []
    for idx, prompt in enumerate(tqdm(prompts)):
        outputs = model.generate(
            prompt, sampling_params=sampling_params, use_tqdm=False
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

        result.append(
            {
                "problem_id": test_dataset[idx]["problem_id"],
                "description": test_dataset[idx]["question"],
                "difficulty": test_dataset[idx]["difficulty"],
                "starter_code": test_dataset[idx]["starter_code"],
                "generated_solutions": generations_,
                "extracted_solutions": solutions_,
                "prompt": prompt,
                "demonstration": demonstration,
            }
        )

        write_dict_to_jsonl(result, os.path.join(base_directory, "result", file_name))

    print("program ends.")


if __name__ == "__main__":
    main()
