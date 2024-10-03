from datasets import load_dataset, Dataset, concatenate_datasets
import json
from utils import *
from radon.complexity import cc_visit
from eval.apps_metric import apps_metric
import os

def filtering(dataset):
    words = ["codeforces", "atcoder", "codechef"]
    dataset = dataset.filter(lambda x: any(word in x["url"] for word in words))
    dataset = make_solution_column(dataset)

    if os.path.exists(
        "data/apps_results.json"
    ):
        results = json.load(
            open(
                "data/apps_results.json",
                "r",
            )
        )
    else:
        eval_apps = apps_metric()
        results, _ = eval_apps._compute(
            dataset, k_list=[1], split="train", column_name="solution"
        )
        json.dump(
            results,
            open(
                "data/apps_results.json",
                "w",
            ),
        )

    data = []
    for index in results:
        sc = []
        sc_cc = []
        mc = []
        mc_cc = []
        cc_criteria = 10
        for i, result in enumerate(results[index]):
            try:
                code = process_text(dataset[int(index)]["solution"][i])
                code_cc = get_avg_cc(code)
                if all(x == True for x in result):
                    if code_cc >= cc_criteria:
                        sc.append(code)
                        sc_cc.append(code_cc)
                    else:
                        visit = cc_visit(code)
                        count = [
                            count_module_written(code, func.name)
                            for func in visit.functions
                        ]
                        TF = all(x >= 2 for x in count)
                        if len(count) >= 3 and TF:
                            mc.append(code)
                            mc_cc.append(code_cc)
            except:
                pass
        data.append({"mc": mc, "mc_cc": mc_cc, "sc": sc, "sc_cc": sc_cc})

    final_data = concatenate_datasets([dataset, Dataset.from_list(data)], axis=1)
    final_data = final_data.filter(
        lambda x: x["sc"] != []
        and x["mc"] != []
        and -10 not in x["sc_cc"]
        and -10 not in x["mc_cc"]
    )

    return final_data


def main():

    dataset_name = "codeparrot/apps"

    dataset = load_dataset(
        dataset_name,
        trust_remote_code=True,
        split="train",
    )

    filtered_dataset = filtering(dataset)
    filtered_dataset.to_json(
        f"data/filtered_APPS.json"
    )


if __name__ == "__main__":
    main()
