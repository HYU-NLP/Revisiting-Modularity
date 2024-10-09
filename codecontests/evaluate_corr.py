import os
import argparse
import multiprocessing
import time

from tqdm import tqdm

from utils.utils_evaluate import compute_pass_at_ks, verify_code_official
from utils.utils import read_jsonl_to_dict, write_dict_to_jsonl

from datasets import load_dataset

from scipy import stats


def _temp_run(code, tests, passed):
    try:
        flag, _ = verify_code_official(tests, code)
        passed.append(flag)
    except Exception as e:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, required=True, default="meta-llama/CodeLlama-7b-hf"
    )
    parser.add_argument("--num_icl_shot", type=int, required=True, default=2)
    parser.add_argument(
        "--num_gen",
        type=int,
        required=True,
        default=1,
        help="number of solutions generated per problem",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        required=True,
        help="0 means greedy decoding for vllm",
    )
    parser.add_argument("--k", type=int, required=True, default=1, help="k of pass@k")
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        default='style',
        help="code metric (e.g., style or modularity)",
    )
    args = parser.parse_args()

    base_directory = os.path.dirname(__file__)
    test_dataset = load_dataset(
        "deepmind/code_contests", split="test", cache_dir="/data/huggingface/datasets"
    )
    
    performances = []
    metrics = []
    
    for code_idx in tqdm(range(100)):
        result_file = f"{args.model.replace('/', '-')}_{args.num_icl_shot}shot_{args.num_gen}gen_{args.temperature}temp_{args.metric}_{code_idx}code_icl_result.jsonl"
        if not os.path.exists(os.path.join(base_directory, "result", result_file)):
            continue
        
        if os.path.exists(os.path.join(base_directory, "result", "corr_exp_evaluation_result", result_file)):
            continue
        
        result_data = read_jsonl_to_dict(os.path.join(base_directory, "result", result_file))
        assert len(result_data) == 165
        
        start = time.time()
        passed_results = []
        for i, data in enumerate(result_data):
            # make test cases for each problem
            tests = {"inputs": [], "outputs": []}
            tests["inputs"].extend(data["public_tests"]["input"])
            tests["inputs"].extend(data["private_tests"]["input"])
            tests["outputs"].extend(data["public_tests"]["output"])
            tests["outputs"].extend(data["private_tests"]["output"])
            assert len(tests["inputs"]) == len(tests["outputs"])
            
            time_limit = test_dataset[i]["time_limit"]["seconds"]
            passed = []
            for code in data["extracted_solutions"]:
                manager = multiprocessing.Manager()
                manager_list = manager.list()
                p = multiprocessing.Process(
                    target=_temp_run, args=(code, tests, manager_list)
                )
                p.start()
                p.join(timeout=time_limit + 1)
                
                if p.is_alive():
                    p.kill()
                if not manager_list:
                    passed.append(0)
                else:
                    if manager_list[0] == True:
                        passed.append(1)
                    else:
                        passed.append(0)
                        
            result_data[i]["passed"] = passed  # new data
            passed_results.append(passed)
            
        print(f"time: {time.time() - start:.2f}s")
        ks = [args.k]
        performance = compute_pass_at_ks(passed_results, ks)
        print(f"pass@{ks[0]}: {performance}")
        # statistics for one dot in the correlation figure
        performances.append(performance)
        metrics.append(result_data[0]['demonstration'])
        # add pass information to result_data and save
        write_dict_to_jsonl(result_data, os.path.join(base_directory, "result", "corr_exp_evaluation_result", result_file))
        print(f'{result_file} saved.')
    
    print('program ends.')
    
    # # compute correlation
    # pass_at_k = [e[args.k] for e in performances]
    
    # if args.metric == 'style':
    #     style = [m['score_style'][0]['score_pep8'] for m in metrics]
    #     print(stats.pearsonr(style, pass_at_k))
    #     print(stats.spearmanr(style, pass_at_k))
    # elif args.metric == 'modularity':
    #     modularity = [m['score_modularity'][0] for m in metrics]
    #     print(stats.pearsonr(modularity, pass_at_k))
    #     print(stats.spearmanr(modularity, pass_at_k))
        



if __name__ == "__main__":
    main()
