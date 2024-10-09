import re
import os
from datasets import load_dataset
from utils.utils_evaluate import safe_eval_answer_from_agent_ft
from utils.utils import get_code_modularity_score


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


def calculate_mos(example):
    scores = []
    for code in example['solutions']['solution']:
        try:
            modularity_score = get_code_modularity_score(code.strip())
        except:
            modularity_score = -1
            
        scores.append(modularity_score)
        
    example['solutions']['modularity'] = scores

    return example


def start(split):
    base_dir = os.path.dirname(__file__)
    
    # load original dataset
    dataset = load_dataset("deepmind/code_contests")
    dataset = dataset[split]
    # dataset = dataset[split].select(range(5)) # for test
    print(f'len(dataset): {len(dataset)}')
    # 1. filter questions without any python solution
    print('1')
    dataset = dataset.filter(lambda example: 3 in example['solutions']['language'])
    # 2. retain only python solutions in problem
    print('2')
    dataset = dataset.map(leave_python_solution, num_proc=16)
    # 3. remove annotation in the code
    print('3')
    dataset = dataset.map(remove_annotation, num_proc=16)
    # 4. retain only python solutions that pass the test cases
    print('4')
    dataset = dataset.map(safe_eval_answer_from_agent_ft, num_proc=16)
    # 5. calculate MoS score of code
    print('5')
    dataset = dataset.map(calculate_mos, num_proc=16)
    # 6. save
    dataset.to_json(os.path.join(base_dir, 'data/ft', f'my_code_contests_{split}.jsonl'))


# start('test')
start('valid')
# start('train')