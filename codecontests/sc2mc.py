import os
import argparse

from utils.utils_evaluate import verify_code_official

from utils.utils import read_jsonl_to_dict, write_dict_to_jsonl

from openai import OpenAI

import multiprocessing


problem_description = '''\
QUESTION:
Given a permutation $p$ of length $n$, find its subsequence $s_1$, $s_2$, $\ldots$, $s_k$ of length at least $2$ such that:  $|s_1-s_2|+|s_2-s_3|+\ldots+|s_{{k-1}}-s_k|$ is as big as possible over all subsequences of $p$ with length at least $2$.  Among all such subsequences, choose the one whose length, $k$, is as small as possible. 

If multiple subsequences satisfy these conditions, you are allowed to find any of them.

A sequence $a$ is a subsequence of an array $b$ if $a$ can be obtained from $b$ by deleting some (possibly, zero or all) elements.

A permutation of length $n$ is an array of length $n$ in which every element from $1$ to $n$ occurs exactly once.


-----Input-----

The first line contains an integer $t$ ($1 \le t \le 2 \cdot 10^4$) — the number of test cases. The description of the test cases follows.

The first line of each test case contains an integer $n$ ($2 \le n \le 10^5$) — the length of the permutation $p$.

The second line of each test case contains $n$ integers $p_1$, $p_2$, $\ldots$, $p_{{n}}$ ($1 \le p_i \le n$, $p_i$ are distinct) — the elements of the permutation $p$.

The sum of $n$ across the test cases doesn't exceed $10^5$.


-----Output-----

For each test case, the first line should contain the length of the found subsequence, $k$. The second line should contain $s_1$, $s_2$, $\ldots$, $s_k$ — its elements.

If multiple subsequences satisfy these conditions, you are allowed to find any of them.


-----Example-----
Input
2
3
3 2 1
4
1 3 4 2

Output
2
3 1 
3
1 4 2 



-----Note-----

In the first test case, there are $4$ subsequences of length at least $2$:  $[3,2]$ which gives us $|3-2|=1$.  $[3,1]$ which gives us $|3-1|=2$.  $[2,1]$ which gives us $|2-1|=1$.  $[3,2,1]$ which gives us $|3-2|+|2-1|=2$. 

So the answer is either $[3,1]$ or $[3,2,1]$. Since we want the subsequence to be as short as possible, the answer is $[3,1]$.\
'''

sc = '''\
ANSWER:
```python
import sys
for _ in range(int(input())):
    n = int(input())
    data = list(map(int, input().split()))
    ans = [data[0]]
    for i in range(1, n - 1):
        if data[i - 1] < data[i] > data[i + 1] or data[i - 1] > data[i] < data[i + 1]:
            ans += [data[i]]
    print(len(ans) + 1)
    print(*ans, data[-1])
```\
'''

mc = '''\
```python
import sys

def ii():
    return sys.stdin.readline().strip()

def idata():
    return [int(x) for x in ii().split()]

def solve_of_problem():
    n = int(ii())
    data = idata()
    ans = [data[0]]
    for i in range(1, n - 1):
        if data[i - 1] < data[i] > data[i + 1] or data[i - 1] > data[i] < data[i + 1]:
            ans += [data[i]]
    print(len(ans) + 1)
    print(*ans, data[-1])
    return

if __name__ == '__main__':
    for ______ in range(int(ii())):
        solve_of_problem()
```\
'''

sc2mc_instruction = '''\
Refactor the above python program following the question. Follow the guidelines
* make the program more modular with smaller and meaningful helper functions
* good descriptive names for the helper functions
* have an entry function called ‘main()’ 
* 'main()' is called inside 'if __name__ == '__main__''

Do not change the original semantics of the program significantly and no need to perform optimizations. \
Enclose the program within backticks as shown above\
'''

mc2sc_instruction = '''\
Refactor the above program. Follow the guidelines
* make the program monolithic without helper functions
* transform the program with multiple functions into a single piece of code
* do not copy the given code exactly as it is
* eliminate any modular structures such as separate functions or classes, merging them into a continuous, unified script

Do not change the original semantics of the program significantly and no need to perform optimizations. \
Enclose the program within backticks as shown above\
'''

sc2mc_demonstration = {
    'problem_description': problem_description,
    'sc': sc,
    'mc': mc,
    'instruction': sc2mc_instruction
}

mc2sc_demonstration = {
    'problem_description': problem_description,
    'sc': sc,
    'mc': mc,
    'instruction': mc2sc_instruction
}

def make_gpt_chat_messsage(role, content):
    return {'role': role, 'content': content}


def make_sc2mc_prompt(demonstration, input, shot):
    problem_description = demonstration['problem_description']
    sc = demonstration['sc']
    mc = demonstration['mc']
    instruction = demonstration['instruction']
    
    input_problem_description = input['problem_description']
    input_code = input['code']
    
    messages = []
    
    # zero-shot prompt for sc -> mc
    if shot == 0:
        messages.append(make_gpt_chat_messsage('system', "You are an AI programming assistant."))
        messages.append(make_gpt_chat_messsage('user', 'QUESTION:\n' + input_problem_description + '\n' + 'ANSWER:\n```python\n' + input_code + '\n```\n' + instruction))
    # 1 shot prompt for sc -> mc
    else:
        messages.append(make_gpt_chat_messsage('system', "You are an AI programming assistant."))
        messages.append(make_gpt_chat_messsage('user', problem_description + '\n' + sc + '\n' + instruction))
        messages.append(make_gpt_chat_messsage('assistant', mc))
        messages.append(make_gpt_chat_messsage('user', 'QUESTION:\n' + input_problem_description + '\n' + 'ANSWER:\n```python\n' + input_code + '\n```\n' + instruction))
    
    return messages


def make_mc2sc_prompt(demonstration, input, shot):
    problem_description = demonstration['problem_description']
    sc = demonstration['sc']
    mc = demonstration['mc']
    instruction = demonstration['instruction']
    
    input_problem_description = input['problem_description']
    input_code = input['code']
    
    messages = []
    
    # zero-shot prompt for mc -> sc
    if shot == 0:
        messages.append(make_gpt_chat_messsage('system', "You are an AI programming assistant."))
        messages.append(make_gpt_chat_messsage('user', 'QUESTION:\n' + input_problem_description + '\n' + 'ANSWER:\n```python\n' + input_code + '\n```\n' + instruction))
    # 1 shot prompt for sc -> mc
    else:
        messages.append(make_gpt_chat_messsage('system', "You are an AI programming assistant."))
        messages.append(make_gpt_chat_messsage('user', problem_description + '\n' + mc + '\n' + instruction))
        messages.append(make_gpt_chat_messsage('assistant', sc))
        messages.append(make_gpt_chat_messsage('user', 'QUESTION:\n' + input_problem_description + '\n' + 'ANSWER:\n```python\n' + input_code + '\n```\n' + instruction))
    
    return messages


def check_correctness(code, tests):
    GLOBAL_TIMEOUT = 10
    
    def _temp_run(code, tests, result):
        try:
            flag, outcomes = verify_code_official(tests, code)
            result.append(flag)
        except Exception as e:
            pass
        
    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(code, tests, result))
    p.start()
    p.join(timeout=GLOBAL_TIMEOUT + 1)
    if p.is_alive():
        p.kill()
    if not result:
        result = [-1]
    if result[0] == True:
        return True
    else:
        return False

def main():
    # seeds = [27, 42, 101, 134, 169]
    # seeds = [42, 101, 134, 169]
    seeds = [101]
    code_type = 'monolithic'
    client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

    for seed in seeds:
        base_directory = os.getcwd()
        file_name = f"{code_type}_2shot_demonstration_{seed}seed.jsonl"
        data = read_jsonl_to_dict(os.path.join(base_directory, 'data', file_name))[0]

        transformed_code = []
        passed = []
        # 2 examples are in demonstration
        for i in range(2):    
            problem_description = data['problem_description'][i]
            input_code = data['code'][i]
            input = {'problem_description': problem_description, 'code': input_code}
            messages = make_sc2mc_prompt(sc2mc_demonstration, input, shot=1)
            
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=1024,
                stop=["\n\n\n\n", "####", "----"],
                temperature=0,
            )
            response = completion.choices[0].message.content
            
            start_index = response.find('```python')
            if start_index != -1:
                end_index = response.find('```', start_index + len('```python'))
                if end_index != -1:
                    response = response[start_index + len('```python'): end_index]
                else:
                    response = response[start_index + len('```python'):]
            transformed_code.append(response)
            
            ## correctness check
            tests = {'inputs': [], 'outputs': []}
            tests['inputs'].extend(data['public_tests'][i]['input'])
            tests['inputs'].extend(data['private_tests'][i]['input'])
            tests['outputs'].extend(data['public_tests'][i]['output'])
            tests['outputs'].extend(data['private_tests'][i]['output'])

            if check_correctness(response, tests) == True:
                print('pass')
                passed.append(True)
            else:
                print('not passed')
                passed.append(False)
                
        data['transformed_code'] = transformed_code
        data['passed'] = passed
        write_dict_to_jsonl([data], os.path.join(base_directory, 'data', file_name))
main()