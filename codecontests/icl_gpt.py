import os
import random
import argparse
from tqdm import tqdm

import numpy as np
from collections import defaultdict

import torch

from datasets import load_dataset

from openai import OpenAI

from utils.utils import read_jsonl_to_dict, write_dict_to_jsonl


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


# def extract_demonstration(train_dataset, shot, code_type):
#     problem_with_both_sc_and_mc = []
#     for i, data in enumerate(train_dataset):
#         num_sc = len(data['monolithic_codes']['monolithic_code'])
#         num_mc = len(data['modular_codes']['modular_code']) 
#         if num_sc > 0 and num_mc > 0:
#             problem_with_both_sc_and_mc.append(i)

#     problem_index = random.sample(problem_with_both_sc_and_mc, shot)
#     demonstration = defaultdict(list)

#     for i in problem_index:
#         data = train_dataset[i]

#         if code_type == 'monolithic':
#             demonstration['problem_description'].append(data['problem_description'])
#             demonstration['code'].append(data['monolithic_codes']['monolithic_code'][0]) # pick the first code
#             demonstration['code_cc'].append(data['monolithic_codes']['monolithic_code_cc'][0])
#         elif code_type == 'modular':
#             demonstration['problem_description'].append(data['problem_description'])
#             demonstration['code'].append(data['modular_codes']['modular_code'][0])
#             demonstration['code_cc'].append(data['modular_codes']['modular_code_cc'][0])
    
#     # demonstration consists of shot-number of (problem, code, cc of code)
#     return demonstration

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
                # print(get_code_modularity_score(data['modular_codes']['modular_code'][0]))
                # print(data['modular_codes']['modular_code'][0])

        return demonstration
    
    else:
        if code_type == 'transformed_modular':
            key = 'transformed_mc'
        elif code_type == 'transformed_monolithic':
            key = 'transformed_sc'

        demonstration = defaultdict(list)
        for i in range(shot):
            demonstration['description'].append(train_dataset['problem_description'][i].strip())
            demonstration['code'].append(train_dataset[key][i].strip())

        return demonstration
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, default=42)
    parser.add_argument('--model', type=str, required=True, default='gpt')
    parser.add_argument('--num_icl_shot', type=int, required=True, default=2)
    parser.add_argument('--num_gen', type=int, required=True, default=1, help='number of solutions generated per problem')
    parser.add_argument('--code_type', type=str, required=True, default='monolithic')
    parser.add_argument('--temperature', type=float, required=True, default=0, help='0 means greedy decoding for vllm')
    parser.add_argument('--max_new_token', type=int, required=True, default=1024)
    parser.add_argument('--debug_mode', type=int, required=True, default=0)
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # load model
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    # use train dataset to extract demonstrations (problem, code)
    # if args.num_icl_shot is 0, demonstration is not made
    # train_dataset = read_jsonl_to_dict(os.path.join(os.path.dirname(__file__), 'data', 'my_code_contests_divided_train.jsonl'))
    # demonstration = extract_demonstration(train_dataset, args.num_icl_shot, args.code_type)
    
    base_directory = os.path.dirname(__file__)
    
    # monolithic(sc) or modular(mc) demonstration
    if 'transformed' not in args.code_type: 
        dataset = read_jsonl_to_dict(os.path.join(os.path.dirname(__file__), 'data', 'my_code_contests_divided_train.jsonl'))
    # transformed monolithic(tsc) or transformed modular(tmc) demonstration
    else:
        dataset = read_jsonl_to_dict(os.path.join(os.path.dirname(__file__), 'data', f'monolithic_2shot_demonstration_{args.seed}seed.jsonl'))[0]
    
    demonstration = extract_demonstration(dataset, args.num_icl_shot, args.code_type)
    
    test_dataset = load_dataset("deepmind/code_contests", split='test')

    # # old instruction format
    # sc_instruction =  (    
    #     "Write a python code to solve the following coding problem "
    #     "that obeys the constraints and passes the example test cases.\n"
    # )
    # mc_instruction = (
    #     "Write a python code to solve the following coding problem "
    #     "that obeys the constraints and passes the example test cases.\n"
    #     "Follow  the guidelines\n"
    #     "* Your code must be modular with smaller and meaningful helper functions.\n"
    #     "* Your code must include helper functions that will be used in your code and then write your code using them."
    # )

    # new instruction format
    sc_instruction = (    
        "Write a python code to solve the following coding problem "
        "that obeys the constraints and passes the example test cases. "
    )
    mc_instruction = (
        "Write a python code to solve the following coding problem "
        "that obeys the constraints and passes the example test cases. "
        "Ensure modularity of the python code by dividing the code into smaller, useful functions to solve the given problem. "
    )
    
    def make_gpt_chat_messsage(role, content):
        return {'role': role, 'content': content}
    
    # generated_solutions = {}
    # extracted_solutions = {}
    
    generations = []
    solutions = []
        
    test_dataset = list(test_dataset) # for test
    if args.debug_mode:
        test_dataset = test_dataset[:2]
        
    for test_data in tqdm(test_dataset):
        messages = []
        messages.append(make_gpt_chat_messsage('system', "You are an AI programming assistant."))
        if 'monolithic' in args.code_type: # monolithic, transformed monolithic
            messages.append(make_gpt_chat_messsage('user', sc_instruction))
            # zero-shot prompting
            if args.num_icl_shot == 0:
                messages.append(make_gpt_chat_messsage('user', test_data['description'])) # instruction, problem description
            # few-shot prompting
            elif args.num_icl_shot > 0:
                for i in range(args.num_icl_shot):
                    messages.append(make_gpt_chat_messsage('user', demonstration['description'][i])) # instruction, problem description
                    messages.append(make_gpt_chat_messsage('assistant', demonstration['code'][i])) # code
                messages.append(make_gpt_chat_messsage('user', test_data['description'].strip()))
        elif 'modular' in args.code_type: # modular, transformed modular
            messages.append(make_gpt_chat_messsage('user', mc_instruction))
            if args.num_icl_shot == 0:
                messages.append(make_gpt_chat_messsage('user', test_data['description']))
            elif args.num_icl_shot > 0:
                for i in range(args.num_icl_shot):
                    messages.append(make_gpt_chat_messsage('user', demonstration['description'][i]))
                    messages.append(make_gpt_chat_messsage('assistant', demonstration['code'][i]))
                messages.append(make_gpt_chat_messsage('user', test_data['description'].strip()))

        if args.debug_mode:
            print(f'\ncode type: {args.code_type}')
            print('demonstration code:')
            for msg in messages:
                if msg['role'] == 'assistant':
                    print(msg['content'])
                    print('-' * 100)
            
            print('chat message:')
            for msg in messages:
                print(msg)
                print('-' * 100)
                
        generations_ = []
        solutions_ = []
        
        # generate
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            n=args.num_gen,
            messages=messages,
            max_tokens=args.max_new_token,
            stop=["\n\n\n\n", "###", "####", "----"],
            temperature=args.temperature,
        )
        
        for choice in completion.choices:
            response = choice.message.content
            
            if args.debug_mode:
                print(f'gpt response:\n{response}')
                print('=' * 100)

            # original response
            generations_.append(response)

            # if solution is wrapped by backticks            
            start_index = response.find('```python')
            if start_index != -1:
                end_index = response.find('```', start_index + len('```python'))
                if end_index != -1:
                    response = response[start_index + len('```python'): end_index]
                else:
                    response = response[start_index + len('```python'):]

            if args.debug_mode:
                print(f'gpt answer:\n{response}')
                print('=' * 100)
                
            # preprocessed response                
            solutions_.append(response.strip())
        
        # generated_solutions[test_data['name']] = generations
        # extracted_solutions[test_data['name']] = solutions
        
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
                "demonstration": demonstration, # contains code and its description
            }
        )

    base_directory = os.path.dirname(__file__)
    file_name = f"gpt-4o-mini_{args.code_type}_code_{args.num_icl_shot}shot_{args.num_gen}gen_{args.seed}seed_icl_result.jsonl"
    
    write_dict_to_jsonl(result, os.path.join(base_directory, 'result/gpt', file_name))
        
    print('program ends.')

if __name__ == '__main__':
    main()
