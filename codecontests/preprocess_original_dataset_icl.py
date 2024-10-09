import re
import os
from datasets import load_dataset
from utils.utils_evaluate import safe_eval_answer_from_agent
from radon.complexity import cc_visit


# delete all solutions in another language except python in the dataset
def leave_python_solution(example):
    solutions = example['solutions']['solution']
    language_index = example['solutions']['language']

    python_solution = []
    for i, lang in enumerate(language_index):
        if lang == 3: # python3
            python_solution.append(solutions[i])

    example['solutions']['solution'] = python_solution
    del example['solutions']['language']
    return example


# remove annotated parts in the code
def remove_annotation(example):
    def remove_annotation_(input_string):
        modified_string = re.sub(r"#.*?(?=\n)", '', input_string)
        modified_string = re.sub(r"'''.*?'''", '', modified_string, flags=re.DOTALL)
        modified_string = re.sub(r'""".*?"""', '', modified_string, flags=re.DOTALL)
        return modified_string

    for i in range(len(example['solutions']['solution'])):
        example['solutions']['solution'][i] = remove_annotation_(example['solutions']['solution'][i])

    return example


# calculate cc and module list of code and add them to dataset
def add_cc_and_modules(example):
    ccs = []
    modules = []

    for code in example['solutions']['solution']:
        cc, module_name = get_avg_cc_and_module(code)
        ccs.append(cc)
        modules.append(module_name)

    example['solutions']['cc'] = ccs
    example['solutions']['modules'] = modules

    return example


# calculate average cc of each solution code and add it
def get_avg_cc_and_module(code):
    try:
        module_name = []
        visitor = cc_visit(code)

        # 1. average cc of modules
        total_module_complexity = 0
        num_module = 0
        for module in visitor.blocks:
            # only consider function or method of class as module
            if module.__class__.__name__ == 'Function': 
                module_name.append(module.name)
                total_module_complexity += module.complexity
                num_module += 1

        # 2. cc of body code
        body_complexity = visitor.complexity

        # 3. average cc of the program
        avg_cc = (total_module_complexity + body_complexity) / (num_module + 1)
    except:
        # cc_visit fails to return because the input code has some errors
        avg_cc = 0
        module_name = []
    
    return avg_cc, module_name


def start(split):
    base_dir = os.path.dirname(__file__)
    
    # load original dataset
    dataset = load_dataset("deepmind/code_contests", cache_dir='/data/huggingface/datasets')
    dataset = dataset[split]
    # 1. filter questions without any python solution
    dataset = dataset.filter(lambda example: 3 in example['solutions']['language'])
    # 2. retain only python solutions in problem
    dataset = dataset.map(leave_python_solution, num_proc=2)
    # 3: mark each python solution passed or not by running the test cases
    dataset = dataset.map(safe_eval_answer_from_agent, num_proc=1)
    # 4. remove annotation parts of code
    dataset = dataset.map(remove_annotation)
    # 5. add cc and modules names contained in the code to the dataset
    dataset = dataset.map(add_cc_and_modules, num_proc=16)
    # 6. save
    dataset.to_json(os.path.join(base_dir, 'data', f'my_code_contests_{split}.jsonl'))


start('test')
start('valid')
start('train')