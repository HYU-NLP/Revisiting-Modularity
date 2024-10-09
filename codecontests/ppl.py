import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.utils import read_jsonl_to_dict
from tqdm import tqdm
import argparse
import random
import numpy as np
import os


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
    
    
set_seed(42)
    
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, default=0)
parser.add_argument("--model", type=str, required=True, default="meta-llama/CodeLlama-7b-hf")
parser.add_argument("--include_prompt", action='store_true')
parser.add_argument("--mod", type=str, required=True)
args = parser.parse_args()
    
device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(args.model)

if 'CodeLlama' in args.model:
    dtype = torch.float16
elif 'deepseek' in args.model:
    dtype = torch.bfloat16
    
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
model = model.to(device)
tokenizer.pad_token = tokenizer.eos_token

path = f'/home/kdy20401/Workspace/Proj-Code-Generation/MC/data/my_code_contests_train_{args.mod}_mod.jsonl'
dataset = read_jsonl_to_dict(path)

losses = []
perplexity = []
# length = []
problems = []
for j, data in enumerate(dataset):
    description = data['description']
    code = data['code']
    
    if args.include_prompt == True:
        instruction = (
            "Write a python code to solve the following coding problem "
            "that obeys the constraints and passes the example test cases. "
            "The output code needs to read from and write to standard IO. "
            "Please wrap your code answer using ```:"
        )
        if 'CodeLlama' in args.model:
            prefix = ""
            prefix += "Q: " + instruction + "\n"
            prefix += description + "\n"
            prefix += "A: "
        elif 'deepseek' in args.model:
            prefix = ""
            prefix += instruction + '\n'
            prefix += "### Instruction:\n" + description + "\n"
            prefix += "### Response:\n"
            
        prompt = prefix + code
        all_tokens = tokenizer(prompt, return_tensors="pt", max_length=8192, truncation=True).to(device)
        prefix_tokens = tokenizer(prefix, return_tensors="pt", max_length=8192, truncation=True).to(device)
        code_start_index = len(prefix_tokens['input_ids'][0])
        labels = all_tokens['input_ids'].clone()
        labels[:, :code_start_index] = -100 # ignore loss of prefix
    else:
        prompt = code
        all_tokens = tokenizer(prompt, return_tensors="pt", max_length=8192, truncation=True).to(device)
        labels = all_tokens['input_ids']
        
    # problem
    problems.append(data['name'])
    with torch.no_grad():
        outputs = model(all_tokens['input_ids'], labels=labels)
        loss = outputs.loss
        # loss 
        losses.append(loss)
        
        ppl = torch.exp(outputs.loss).item()
        if ppl != torch.nan:
            perplexity.append(ppl)
        else:
            print('nan!')


# print(min(length), max(length))
print(f'model: {args.model}')
print(f'dataset of {args.mod} modularity')
# print(f'average nll: {torch.stack(losses).mean()}')
print(f'average ppl: {sum(perplexity) / len(perplexity)}')
