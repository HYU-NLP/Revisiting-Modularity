import random
import numpy as np
import torch
import os
from datasets import Dataset
from collections import defaultdict


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


def extract_demonstration(train_dataset):
    demonstration = defaultdict(list)
    for i in sorted(random.sample(list(range(len(train_dataset))), 2)):
        data = train_dataset[i]
        if data["starter_code"] != "":
            question_guide = "use the provided function signature"
        else:
            question_guide = "read from and write to standard IO"
        sc_instruction = (
            "Write a python code to solve the following coding problem "
            "that obeys the constraints and passes the example test cases. "
            f"The output code needs to {question_guide}. "
            "Please wrap your code answer using ```:"
        )
        mc_instruction = (
            "Write a python code to solve the following coding problem "
            "that obeys the constraints and passes the example test cases. "
            f"The output code needs to {question_guide}. "
            "Ensure modularity of the python code by dividing the code into smaller, "
            "useful functions to solve the given problem. "
            "Please wrap your code answer using ```:"
        )

        demonstration["problem_id"].append(data["problem_id"])
        demonstration["problem_description"].append(data["question"].strip())
        demonstration["starter_code"].append(data["starter_code"])
        demonstration["sc_instruction"].append(sc_instruction)
        demonstration["mc_instruction"].append(mc_instruction)
        demonstration["sc"].append(data["sc"][0].strip())
        demonstration["sc_cc"].append(data["sc_cc"][0])
        demonstration["mc"].append(data["mc"][0].strip())
        demonstration["mc_cc"].append(data["mc_cc"][0])

    return demonstration


for seed in [27, 42, 101, 134, 169]:
    set_seed(seed)
    dataset = Dataset.from_json("data/filtered_APPS.json")
    demonstration = extract_demonstration(dataset)
    Dataset.from_dict(demonstration).to_json(
        f"data/2shot_demonstration_{seed}seed.json"
    )
