import os
import json
from eval.apps_metric import apps_metric
from eval.utils import get_results
import argparse
from datasets import Dataset
import re
import os
import argparse

from tqdm import tqdm

from utils import read_jsonl_to_dict, write_dict_to_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="meta-llama/CodeLlama-7b-hf")
    parser.add_argument("--num_icl_shot", type=int, default=2)
    parser.add_argument(
        "--num_gen",
        type=int,
        default=10,
        help="number of solutions generated per problem",
    )
    parser.add_argument("--code_type", type=str, default="sc")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="0 means greedy decoding for vllm",
    )
    parser.add_argument("--k", type=int, default=1, help="k of pass@k")
    parser.add_argument(
        "--modify",
        type=str,
        default="original",
        help="modification method of the demonstration code",
    )
    parser.add_argument(
        "--level", type=str, default="all", help="level of the evaluation"
    )

    args = parser.parse_args()

    base_directory = os.path.dirname(__file__)
    file_name = f"{args.model.replace('/', '-')}_{args.code_type}_{args.modify}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.seed}seed_icl_result.jsonl"
    
    data = read_jsonl_to_dict(os.path.join(base_directory, "result", file_name))
    data = Dataset.from_list(data)

    if not os.path.exists(
        os.path.join(base_directory, "tf", file_name.replace("result.jsonl", "tf.json"))
    ):
        eval_apps = apps_metric()
        results, metrics = eval_apps._compute(
            data,
            k_list=[1, 5],
            level=args.level,
            split="test",
            column_name="extracted_solutions",
        )
        json.dump(
            results,
            open(
                os.path.join(
                    base_directory, "tf", file_name.replace("result.jsonl", "tf.json")
                ),
                "w",
            ),
        )
    else:
        results = json.load(open(os.path.join(base_directory, "tf", file_name.replace("result.jsonl", "tf.json")),"r"))
        print("\n\n\nResults: pass@k on all level")
        get_results(
            data,
            k_list=[1, 5],
        )

        
    results_list = [results[index] for index in results]
    passed_list = []
    for results in results_list:
        for result in results:
            passed = []
            for element in result:
                passed.append([int(element)])
        passed_list.append(passed)
    data = data.add_column("passed", passed_list)
    for difficulty in ["introductory", "interview", "competition"]:
        print(f"\n\n\nResults: pass@k on {difficulty} level")
        get_results(
            data.filter(lambda x: x["difficulty"] == difficulty)["passed"],
            k_list=[1, 5],
        )


if __name__ == "__main__":
    main()
