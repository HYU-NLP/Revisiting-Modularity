import json
import re
from radon.complexity import cc_visit

def write_dict_to_jsonl(dictionary, filename):
    import json

    with open(filename, "a") as file:
        for item in dictionary:
            json.dump(item, file)
            file.write("\n")


def read_jsonl_to_dict(filename):
    import json

    result = []
    with open(filename, "r") as file:
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
    permit_left_char = [
        " ",
        "(",
        ":",
        "+",
        "-",
        "*",
        "/",
        "//",
        "%",
        "=",
        "<",
        ">",
        "!",
        "~",
        "&",
        "|",
        "^",
    ]
    permit_right_char = [" ", "("]
    cnt = 0
    for index in indices:
        if (
            code[index - 1] in permit_left_char
            and code[index + len(module)] in permit_right_char
        ):
            cnt += 1

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
            if module.__class__.__name__ == "Function":
                total_module_complexity += module.complexity
                num_module += 1

        # 2. cc of body code
        body_complexity = visitor.complexity

        # 3. average cc of the program
        avg_cc = (total_module_complexity + body_complexity) / (num_module + 1)
    except:
        # cc_visit fails to return because the input code has some errors
        avg_cc = -10

    return avg_cc


def process_text(input_string):
    modified_string = re.sub(r"#.*?(?=\n)", "", input_string)
    modified_string = re.sub(r"'''.*?'''", "", modified_string, flags=re.DOTALL)
    modified_string = re.sub(r'""".*?"""', "", modified_string, flags=re.DOTALL)
    return modified_string


def make_solution_column(dataset):
    solution = []
    for problem in dataset:
        solution.append(json.loads(problem["solutions"]))
    dataset = dataset.add_column("solution", solution)
    return dataset