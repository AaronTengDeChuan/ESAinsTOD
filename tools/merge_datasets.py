# coding: utf-8

import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import chain

def merge_files(file_paths, output_file, train_set=False):
    merged_data = []
    file2num = defaultdict(list) if train_set else {}
    for file in file_paths:
        with open(file, "r", encoding="utf-8") as fin:
            data = json.load(fin)
            if file in file2num and not train_set:
                continue
            if train_set:
                file2num[file].append(len(data))
            else:
                file2num[file] = len(data)
            merged_data.extend(data)
    with open(output_file, "w", encoding="utf-8") as fout:
        json.dump(merged_data, fout, ensure_ascii=False, indent=2)
    dial_tokens, dial_lengths = [], []
    # get stats
    for sample in merged_data:
        dial_tokens.append(sample["total_length"])
        dial_lengths.append(max(1, len(sample["conversations"]) - 1))
    stats = pd.DataFrame({"tokens": np.array(dial_tokens), 'turns': np.array(dial_lengths)})
    stats['token_per_turn'] = stats['tokens'] / stats['turns']
    stats_dict = stats.describe().to_dict()
    stats_dict['total_tokens'] = stats_dict['tokens']['count'] * stats_dict['tokens']['mean']
    stats_dict['total_tokens (B)'] = stats_dict['total_tokens'] / 1e9
    stats_dict['train_tokens (B)'] = len(merged_data) * model_max_length / 1e9

    return stats_dict, file2num


if __name__ == '__main__':
    model_max_length = 4096
    # data_root = "sft_data/v1.0"
    data_root = "sft_data/qwen25"
    suffix = "-ddb"
    num_dataset = 7

    suffixes = ["-raw_uttr", "-ddb"]
    # suffixes = ["-raw_uttr", "-ddb", "-turn_schema"]
    concat_text_fn = lambda texts, exclude: "".join([text for i, text in enumerate(texts) if text not in exclude])
    suffix = concat_text_fn(suffixes, [])
    # num_dataset = 11
    # num_dataset = 5
    merge_prefix = f"merged_{num_dataset}-irs-mix_schema"
    # merge_prefix = f"merged_{num_dataset}-irs-mix_schema-no_woz"
    dataset_dirs = [
        os.path.join(data_root, f"camrest{concat_text_fn(suffixes, [])}"),
        os.path.join(data_root, f"camrest{concat_text_fn(suffixes, [])}-no_sys_act"),
        os.path.join(data_root, f"kvret{concat_text_fn(suffixes, [])}"),
        os.path.join(data_root, f"multiwoz2.1{concat_text_fn(suffixes, [])}"),
        os.path.join(data_root, "sgd", f"mix_1{concat_text_fn(suffixes, ['-raw_uttr'])}"),
        os.path.join(data_root, f"frames{concat_text_fn(suffixes, ['-raw_uttr', '-ddb'])}-no_sys_act"),
        os.path.join(data_root, f"bitod{concat_text_fn(suffixes, ['-raw_uttr'])}"),
        os.path.join(data_root, f"star{concat_text_fn(suffixes, ['-raw_uttr', '-no_con_resp'])}"),
        os.path.join(data_root, f"star{concat_text_fn(suffixes, ['-raw_uttr', '-no_con_resp'])}-no_sys_act"),
        # os.path.join(data_root, "single_turn", f"hwu-turn_level-idr_0.33-dup_2"),
        # os.path.join(data_root, "single_turn", f"clinc-turn_level-idr_0.33-dup_2"),
        # os.path.join(data_root, "single_turn", f"banking-turn_level-idr_0.33-dup_2"),
        # os.path.join(data_root, "single_turn", f"snips-turn_level")
    ]

    # suffixes = ["-pptod", "-raw_uttr", "-ddb"]
    # concat_text_fn = lambda texts, exclude: "".join([text for i, text in enumerate(texts) if text not in exclude])
    # suffix = concat_text_fn(suffixes, [])
    # num_dataset = 7
    # merge_prefix = f"merged_{num_dataset}"
    # dataset_dirs = [
    #     os.path.join(data_root, f"camrest{concat_text_fn(suffixes, [])}"),
    #     os.path.join(data_root, f"kvret{concat_text_fn(suffixes, [])}"),
    #     os.path.join(data_root, f"multiwoz2.1{concat_text_fn(suffixes, [])}"),
    #     os.path.join(data_root, "sgd", f"mix_1{concat_text_fn(suffixes, ['-raw_uttr'])}"),
    #     os.path.join(data_root, f"frames{concat_text_fn(suffixes, ['-raw_uttr', '-ddb'])}-no_sys_act"),
    #     os.path.join(data_root, f"bitod{concat_text_fn(suffixes, ['-raw_uttr'])}"),
    #     os.path.join(data_root, f"star{concat_text_fn(suffixes, ['-raw_uttr', '-no_con_resp'])}"),
    # ]

    train_files = [os.path.join(dataset_dir, "train.json") for dataset_dir in dataset_dirs]
    valid_files = [os.path.join(dataset_dir, "dev.json") for dataset_dir in dataset_dirs]
    test_files = [os.path.join(dataset_dir, "test.json") for dataset_dir in dataset_dirs]

    train_stats, train2num = merge_files(
        train_files, os.path.join(data_root, f"{merge_prefix}_train{suffix}.json"), train_set=True)
    valid_stats, valid2num = merge_files(valid_files, os.path.join(data_root, f"{merge_prefix}_dev{suffix}.json"))
    test_stats, test2num = merge_files(test_files, os.path.join(data_root, f"{merge_prefix}_test{suffix}.json"))

    print(f"train:\n{json.dumps(train_stats, indent=2)}")
    print(f"dev:\n{json.dumps(valid_stats, indent=2)}")
    print(f"test:\n{json.dumps(test_stats, indent=2)}", end="\n\n")

    source2split_num = {}
    longest_path = max(map(len, dataset_dirs))
    max_digit_num = 7
    format_fn = lambda nums: f"{nums[0]:>{max_digit_num}} {'' if len(nums) == 1 else f'x {len(nums)}':>3}" \
        if isinstance(nums, list) else f"{nums:>{max_digit_num}}"
    for dataset_dir in dataset_dirs:
        source2split_num[dataset_dir] = (
            f"{' ' * (longest_path - len(dataset_dir))}"
            f"train={format_fn(train2num[os.path.join(dataset_dir, 'train.json')])} ; "
            f"dev={format_fn(valid2num[os.path.join(dataset_dir, 'dev.json')]):>{max_digit_num}} ; "
            f"test={format_fn(test2num[os.path.join(dataset_dir, 'test.json')]):>{max_digit_num}}"
        )
    source2split_num["TOTAL"] = (
        f"{' ' * (longest_path - 5)}"
        f"train={sum(chain(*train2num.values())):>{len(format_fn([1]))}} ; "
        f"dev={sum(valid2num.values()):>{len(format_fn(1))}} ; "
        f"test={sum(test2num.values()):>{len(format_fn(1))}}"
    )

    print(f"Number of 'chunk={model_max_length}' in each data file:")
    print(json.dumps(source2split_num, indent=2), end="\n\n")

    print(f"Merged dataset saved to '{data_root}/{merge_prefix}_*{suffix}.json'")

    # TODO: add related data paths into the stats file
    with open(os.path.join(data_root, f"{merge_prefix}_stats{suffix}.json"), "w", encoding="utf-8") as fout:
        json.dump({
            "num_dataset": num_dataset,
            "chunk_size": model_max_length,
            "source2split_num": source2split_num,
            "train": train_stats,
            "dev": valid_stats,
            "test": test_stats
        }, fout, indent=2)
