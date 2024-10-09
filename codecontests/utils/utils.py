import json 
from glob import glob 
import pdb 
from tqdm import tqdm 
from multiprocessing import Pool
from functools import partial
import numpy as np
import re
import argparse, pprint 
import datasets
import os 
from collections import Counter
import pandas as pd
import random 
import tokenize
from io import BytesIO
from collections import deque
from scipy.stats import describe 
import copy 
import torch 

import ast
import builtins
from radon.complexity import cc_visit


def extract_code_segment(result, keyword, all_segments=True):
    regex = '\`\`\`\s*{}((?:.|\n)*?)\`\`\`'.format(keyword)
    codes = re.findall(regex, result)
    if len(codes)==0: 
        regex = '\`\`\`\s*{}'.format(keyword)
        indices = [(m.start(0), m.end(0)) for m in re.finditer(regex, result)]
        if len(indices) == 0:
            return ''
        last_end_index = indices[-1][1]
        code = result[last_end_index:]
        return code 
    if all_segments:
        return '\n'.join(codes)
    return codes[-1]

def extract_code(result, keyword='python'):
    code = extract_code_segment(result, keyword)
    if len(code)==0: 
        code = extract_code_segment(result, '', False)

    return code 

def extract_func(result, keyword='module'): 
    regex = '\`\`\`\s*{}((?:.|\n)*?)\`\`\`'.format(keyword)
    codes = re.findall(regex, result)
    
    if len(codes)==0: 
        if '\nSTEP 2' in result:
            index = result.index('\nSTEP 2')
            regex = '\`\`\`\s*python((?:.|\n)*?)\`\`\`'
            codes = re.findall(regex, result[:index])
    
    codes = [o for o in codes if 'class ' not in o and 'def main(' not in o]
        
    new_outputs = [] 
    for output in codes: 
        indices = [m.start() for m in re.finditer('def ', output)]
        if len(indices)>1: 
            funcs = [] 
            for i, index in enumerate(indices[:-1]):
                func = output[index: indices[i+1]]
                funcs.append(func)
            func = output[indices[-1]:]
            funcs.append(func)
            new_outputs += funcs 
        elif len(indices)==0: 
            continue 
        else:
            new_outputs.append(output)
    
    return new_outputs 

def get_func_codes(code_string):
    code_string = code_string.strip()
    file = BytesIO(code_string.encode())
    tokens = None 
    try:
        tokens = deque(tokenize.tokenize(file.readline))
    except Exception as e: 
        print("Error parsing function code: " + str(e)) 
        pass 
    if tokens is None: 
        return []
    lines = []
    while tokens:
        token = tokens.popleft()
        if token.type == tokenize.NAME and token.string == 'def':
            start_line, _ = token.start
            last_token = token
            while tokens:
                token = tokens.popleft()
                if token.type == tokenize.NEWLINE:
                    break
                last_token = token
            if last_token.type == tokenize.OP and last_token.string == ':':
                indents = 0
                while tokens:
                    token = tokens.popleft()
                    if token.type == tokenize.NL:
                        continue
                    if token.type == tokenize.INDENT:
                        indents += 1
                    elif token.type == tokenize.DEDENT:
                        indents -= 1
                        if not indents:
                            break
                    else:
                        last_token = token
            lines.append((start_line, last_token.end[0]))
    code_lines = code_string.split('\n')
    outputs = [] 
    for line in lines: 
        start, end = line 
        function = '\n'.join(code_lines[start-1:end])
        if len(function.strip())>0:
            outputs.append(function)
    return outputs 

def is_in_final_code(funcs, func_codes): 
    output_funcs = [] 
    output_funcs_codes = [] 
    for func in funcs: 
        lines = func.split('\n')
        for line in lines: 
            if 'def ' in line: 
                for func_code in func_codes:
                    if line.strip() in func_code:
                        output_funcs.append(func)
                        output_funcs_codes.append(func_code)
                        break 
                break 
    assert len(output_funcs) == len(output_funcs_codes)
    return output_funcs, output_funcs_codes 

def get_embedding(output_embed_file, data, embedding_model, func_type='func'): 
    from embedding.encoder import CodeBERT, StarEncoder, CodeT5Plus  
    
    if func_type == 'func':
        seqs = [] 
        for x, y in zip(data['func'].tolist(), data['func_code'].tolist()):
            if x != 'No docstring':
                seqs.append(x + '\n' + y)
            else:
                seqs.append(y)
                
    elif func_type == 'centroid':
        seqs = []
        clusters = [] 
        docs = [] 
        codes = [] 
        for row in data.iterrows(): 
            cluster = row[1]['cluster']
            if cluster not in clusters:
                clusters.append(cluster)
                doc = row[1]['centroid']
                code = row[1]['centroid_code']
                docs.append(doc)
                codes.append(code)
                if doc != 'No docstring':
                    seqs.append(doc + '\n' + code)
                else:
                    seqs.append(y)
                
    if os.path.exists(output_embed_file): 
        print("Loading embedding from {}".format(output_embed_file))
        embeds = np.load(output_embed_file)
        print("Embedding of shape {}".format(embeds.shape))
    else:
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        MAX_INPUT_LEN = 10000
        MAX_TOKEN_LEN = 512 if embedding_model == 'codebert' else 1024 

        if embedding_model == 'codebert':
            encoder = CodeBERT(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        elif embedding_model == 'starencoder': 
            encoder = StarEncoder(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        elif embedding_model == 'codet5': 
            encoder = CodeT5Plus(DEVICE, MAX_INPUT_LEN, MAX_TOKEN_LEN)
        
        print("Obtain embeddings...")
        embeddings = encoder.encode(seqs)
        embeds = np.stack(embeddings, axis=0)
        print("Embedding of shape {}".format(embeds.shape))
        np.save(output_embed_file, embeds)
        print("Saved embedding to {}".format(output_embed_file))
    
    if func_type == 'centroid':
        return embeds, docs, codes 
    
    return embeds 

def create_func_prompt(doc, code):
    if doc == 'No docstring': 
        return code 
    else:
        code_lines = code.split('\n')
        cutoff_line_idx = -1 
        for line_idx, line in enumerate(code_lines): 
            if 'def ' in line: 
                cutoff_line_idx = line_idx
                break 
        code = '\n'.join(code_lines[cutoff_line_idx+1:])
        return doc + '\n' + code

def get_util_functions_self_cluster(data, num_clusters=1, problem_id_type=int):
    outputs = {}     
    for row in data.iterrows():
        file = row[1]['file']
        problem_id = problem_id_type(file.split('/')[-1].replace('.json', ''))
        centroid = row[1]['centroid_code']
        centroid_doc = row[1]['centroid']

        if problem_id not in outputs: 
            outputs[problem_id] = []
        
        func_str = create_func_prompt(centroid_doc, centroid)
        if func_str not in outputs[problem_id]:
            outputs[problem_id].append(func_str)    
        
    new_outputs = {} 
    for k,v in outputs.items(): 
        sampled = random.sample(v, min(num_clusters, len(v)))
        new_outputs[k] = sampled
    
    lens = [len(i) for i in new_outputs.values()]
    print("Distribution of number of utils:")
    print(describe(lens))
    print(Counter(lens))
    return new_outputs 

def udpate_code_by_all_past_results(results, past_results, files):
    new_files = [] 
    for k,v in past_results.items():
        past_result = v['result']
        
        if k in results: 
            curr_result = results[k]['result']            
        
            # if no passed code in this round, check past results 
            if True not in curr_result and True in past_result: 
                results[k] = past_results[k]
        
        elif True in past_result: 
            results[k] = past_results[k]
            
        if k in results:
            new_files.append(results[k]['file'])        
    
    return results, new_files


#################### my utils below ####################
# ex, functions related to calculate cc, helper function, etc.
def write_dict_to_jsonl(dictionary, filename):
    import json

    with open(filename, 'w') as file:
        for item in dictionary:
            json.dump(item, file)
            file.write('\n')


def read_jsonl_to_dict(filename):
    import json
    
    result = []
    with open(filename, 'r') as file:
        for line in file:
            item = json.loads(line.strip())
            result.append(item)
    return result


def count_module_written(code, module):
    indices = []
    index = -1
    # find all parts starting with module name in the code
    while True:
        index = code.find(module, index + 1)
        if index == -1:
            break
        indices.append(index)

    # filter 
    permit_left_char = [' ', '(', ':', '+', '-', '*', '/', '//', '%', '=', '<', '>', '!', '~', '&', '|', '^'] + ['.', '{', '\n']
    permit_right_char = [' ', '(']
    cnt = 0
    for index in indices:
        if code[index - 1] in permit_left_char and code[index + len(module)] in permit_right_char:
            cnt += 1
    # print(f'module: {module}, cnt: {cnt}, len(index): {len(indices)}')
    return cnt


# calculate average cc of code
def get_avg_cc(code):
    try:
        visitor = cc_visit(code)

        # 1. average cc of modules
        total_module_complexity = 0
        num_module = 0
        for module in visitor.blocks:
            # only consider function or method of class as module
            if module.__class__.__name__ == 'Function': 
                total_module_complexity += module.complexity
                num_module += 1

        # 2. cc of body code
        body_complexity = visitor.complexity

        # 3. average cc of the program
        avg_cc = (total_module_complexity + body_complexity) / (num_module + 1)
    except:
        # cc_visit fails to return because the input code has some errors
        avg_cc = -1
    
    return avg_cc


# return the number of lines of code
def get_loc(code):
    from radon.raw import analyze
    
    return analyze(code).loc


# helper function to get the average length of variables
class VariableVisitor(ast.NodeVisitor):
    def __init__(self):
        self.variables = set()
    
    def visit_Name(self, node):
        # Collect variable names that are identifiers, not just used as constants
        if isinstance(node.ctx, ast.Store) and node.id not in dir(builtins):
            self.variables.add(node.id)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        # Collect argument names from function definitions
        for arg in node.args.args:
            if arg.arg not in dir(builtins):
                self.variables.add(arg.arg)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        # Collect variable names from assignments
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id not in dir(builtins):
                self.variables.add(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name) and elt.id not in dir(builtins):
                        self.variables.add(elt.id)
        self.generic_visit(node)
    
    def visit_For(self, node):
        # Collect variable names from for-loops
        if isinstance(node.target, ast.Name) and node.target.id not in dir(builtins):
            self.variables.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name) and elt.id not in dir(builtins):
                    self.variables.add(elt.id)
        self.generic_visit(node)

    def visit_comprehensions(self, node):
        # Collect variable names from comprehensions
        for generator in node.generators:
            if isinstance(generator.target, ast.Name) and generator.target.id not in dir(builtins):
                self.variables.add(generator.target.id)
            elif isinstance(generator.target, ast.Tuple):
                for elt in generator.target.elts:
                    if isinstance(elt, ast.Name) and elt.id not in dir(builtins):
                        self.variables.add(elt.id)
        self.generic_visit(node)


def soft_step_function(x):
    if x < 0:
        return 0
    elif x >= 0 and x < 10:
        return x / 10
    else:
        return 1
    
    
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


def get_average_length_of_variables(code):
    try:
        code = code.strip()
        tree = ast.parse(code)
        visitor = VariableVisitor()
        visitor.visit(tree)
        avg_len = sum(len(var) for var in visitor.variables) / len(visitor.variables)
    except ZeroDivisionError:
        # no variables are used in the code
        avg_len = 0
    except:
        # ast.parse fails to return because the input code has some errors
        raise Exception('AST parsing error at get_average_length_of_variables function')
    
    return avg_len

    
def get_pep8_style_following_score(code):
    code = code.strip()  
    pid = os.getpid() # necessary when executing this function by multiprocessing  
    tmp_file = os.path.join(os.getcwd(), f'tmp{pid}.py') 
    with open(tmp_file, 'w') as f:
        f.write(code)
    
    ignore = ['W292'] # no newline at end of file (W292)
    command = f'flake8 --extend-ignore {",".join(ignore)} {tmp_file}'
    stream = os.popen(command)
    messages = stream.read().split('\n') # get all error messages
    
    # for test
    # for msg in messages:
        # print(msg)
        
    error_lines = set()
    total_lines = get_loc(code)
    for msg in messages:
        if len(msg) == 0:
            continue
        error_line = int(msg.split(':')[1])
        error_lines.add(error_line)
    
    # more lines which contain errors, the lower the score    
    score_pep8 = 1 - len(error_lines) / total_lines
    
    # remove tmp file
    if os.path.isfile(tmp_file):
        os.remove(tmp_file)
        
    return score_pep8

    
# code style metric based on variable name length and pep8 style guide
def get_code_style_score(code: str):
    # variable name score
    score_var = soft_step_function(get_average_length_of_variables(code)) # [0, 1]
    # pep8 style score
    score_pep8 = get_pep8_style_following_score(code) # [0, 1]
    # style score
    score_style = (score_var + score_pep8) / 2
        
    return {
        'score_var': score_var,
        'score_pep8': score_pep8,
        'score_style': score_style
    }


def count_num_module_calls(code):
    try:
        code = code.strip()
        
        visitor = cc_visit(code)
        modules = []
        for module in visitor.blocks:
            # only consider function or method of class as module
            if module.__class__.__name__ == 'Function': 
                modules.append(module.name)
        # print(modules)
        written_counts = [count_module_written(code, module) for module in modules]
        # print(written_counts)
        num_module_calls = sum([count - 1 for count in written_counts if count >= 2])
        
        return num_module_calls
    except:
        # cc_visit fails to return because the input code has some errors
        raise Exception('cc visiting error at get_code_modularity_score function')


# code modularity metric(MoS) based on cyclomatic complexity and number of modules
# if the complexity of the code is more than 5, the code is considered to be modularized
# NOTE: change 'cc_visit_ast(code2ast(code), **kwargs).blocks' in complexity.py in radon
# to 'cc_visit_ast(code2ast(code), **kwargs)' for execution
def get_code_modularity_score(code: str):
    try:
        code = code.strip()
        
        # calculate total cyclomatic complexity: module complexity + body complexity
        visitor = cc_visit(code)
        total_complexity = 0
        modules = []
        for module in visitor.blocks:
            # only consider function or method of class as module
            if module.__class__.__name__ == 'Function': 
                total_complexity += module.complexity
                modules.append(module.name)
        total_complexity += visitor.complexity
        
        used_modules = [module for module in modules if count_module_written(code, module) >= 2]
        cc_limit = 5
        if total_complexity >= cc_limit:
            num_ideal_modules = int(total_complexity / cc_limit)
            score_modularity = min(1.0, len(used_modules) / num_ideal_modules)
        else:
            # if the code has modules even if the complexity is less than 5,
            # consider the modularity of the code to be 1
            if len(used_modules) >= 1:
                score_modularity = 1.0
            # consider the modularity of simple monolithic code to be 0
            else:
                score_modularity = 0.0
    
        return score_modularity
    except:
        # cc_visit fails to return because the input code has some errors
        raise Exception('cc visiting error at get_code_modularity_score function')