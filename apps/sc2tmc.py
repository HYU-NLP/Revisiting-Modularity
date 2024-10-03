from openai import OpenAI
from datasets import Dataset
from utils import *
from eval.apps_metric import apps_metric

from filter import *


def mc_transform(question, sc):
    client = OpenAI(api_key=your_key)
    try:
        messages = [
            {
                "role": "system",
                "content": "You are an AI programming assistant.",
            },
            {
                "role": "user",
                "content": f"""QUESTION:
{question}

ANSWER:
```python
{sc}
```
Refactor the above program. Follow the guidelines
* make the program more modular with smaller and meaningful helper functions
* good descriptive names for the helper functions
* have an entry function called 'main()'
* 'main()' is called inside 'if __name__ == '__main__''

Do not change the original semantics of the program significantly and no need to perform optimizations. Enclose the program within backticks as shown above.""",
            },
        ]

        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1024,
            stop=["\n\n\n\n", "QUESTION:", "ANSWER:"],
            temperature=0.6,
            n=20,
        )

        response = []
        for choice in completion.choices:
            content = choice.message.content
            response.append(extract_solution(content))
        return response

    except:
        return None


def extract_solution(code):
    start_index = code.find("```python")
    if start_index == -1:
        solution = code
    else:
        end_index = code.find("```", start_index + len("```python"))
        if start_index < end_index:
            solution = code[start_index + len("```python") : end_index]
        else:
            solution = code[start_index + len("```python") :]
    return solution


def main():
    eval_apps = apps_metric()
    # for seed in [27, 42, 101, 134, 169]:
    for seed in [27]:
        data = Dataset.from_json(f"data/2shot_demonstration_{seed}seed.json")
        dataset = data.map(
            lambda x: {"tmc": mc_transform(x["problem_description"], x["sc"])}
        )
        results, _ = eval_apps._compute(
            dataset, k_list=[1], split="train", column_name="tmc"
        )
        transformed_mc = []
        for index in results:
            passed_code = []
            for i, result in enumerate(results[index]):
                code = dataset["tmc"][int(index)][i]
                print(code)
                if all(x == True for x in result):
                    visit = cc_visit(code)
                    count = [
                        count_module_written(code, func.name)
                        for func in visit.functions
                    ]
                    TF = all(x >= 2 for x in count)
                    if len(count) >= 3 and TF:
                        passed_code.append([code])
                        break
            if not len(passed_code) > 0:
                # raise ValueError("No code passed the criteria")
                break
            else:
                transformed_mc.append(passed_code[0])
        if len(transformed_mc) == len(dataset):
            dataset = dataset.remove_columns(["tmc"])
            dataset = dataset.add_column("transformed_mc", transformed_mc)
            dataset.to_json(
                f"data/2shot_demonstration_{seed}seed.json"
            )


if __name__ == "__main__":
    main()
