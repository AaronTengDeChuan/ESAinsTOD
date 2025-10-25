import os
import json
import glob
import random
import argparse
from collections import defaultdict, OrderedDict


def collect_dialogues(variant_path):
    splits = ["train", "dev", "test"]
    split2schema = {}
    chunk2dials = defaultdict(list)
    for split in splits:
        split2schema[split] = json.load(open(os.path.join(variant_path, split, "schema.json"), 'r', encoding="utf-8"))
        for chunk_path in glob.glob(os.path.join(variant_path, split, "dialogues_*.json")):
            chunk_name = os.path.basename(chunk_path)
            chunk2dials[os.path.join(split, chunk_name)] = json.load(open(chunk_path, 'r', encoding="utf-8"))
    return split2schema, chunk2dials


def merge_schema(variant_schemas):
    merged_schema = defaultdict(list)
    splits = ["train", "dev", "test"]
    for split in splits:
        for variant in variant_schemas:
            merged_schema[split].extend(variant[split])
    return merged_schema


def merge_dialogues(variant_dials, num_epochs):
    variant_names, variant_dials = zip(*variant_dials)
    merged_dials = defaultdict(list)
    chunk_names = variant_dials[0].keys()
    variant_counter = defaultdict(lambda: defaultdict(int))
    for chunk_name in chunk_names:
        split, chunk_file = os.path.split(chunk_name)
        dial_ids = [dial["dialogue_id"] for dial in variant_dials[0][chunk_name]]
        for vi in range(1, len(variant_dials)):
            assert dial_ids == [dial["dialogue_id"] for dial in variant_dials[vi][chunk_name]]
        for idx, dial_id in enumerate(dial_ids):
            random_idx = random.sample(range(len(variant_names)), num_epochs)
            if split in ["dev", "test"]:
                random_idx = random_idx[:1]
            for vi, vname in sorted(zip(random_idx, [variant_names[_] for _ in random_idx]), key=lambda x: x[1]):
                dial_src = variant_dials[vi][chunk_name][idx]
                assert dial_src["dialogue_id"] == dial_id
                dial_src["dialogue_id"] = f"{vname}-{dial_id}"
                merged_dials[chunk_name].append(dial_src)
                variant_counter[split][vname] += 1
    return merged_dials, {split: OrderedDict(sorted(counter.items(), key=lambda x: x[0])) for split, counter in variant_counter.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="raw_data/sgd")
    parser.add_argument("--output_dir", type=str, default="raw_data/sgd")
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    variant_paths = glob.glob(os.path.join(args.input_dir, "v*"))
    print(f"variant_paths: {variant_paths}")
    variant_schemas, variant_dials = [], []
    for variant_path in variant_paths:
        var_name = os.path.basename(variant_path)
        var_schema, var_dials = collect_dialogues(variant_path)
        variant_schemas.append(var_schema)
        variant_dials.append((var_name, var_dials))
        print(f"Read '{sum(map(len, var_dials.values()))}' dialogues from '{variant_path}'.")

    merged_schema = merge_schema(variant_schemas)
    merged_dials, variant_counter = merge_dialogues(variant_dials, args.num_epochs)
    output_dir = os.path.join(args.output_dir, f"mix_{args.num_epochs}")
    num_origin, num_merged = 0, 0
    for chunk_name, dials in merged_dials.items():
        num = len(variant_dials[0][1][chunk_name])
        output_subdir, chunk_name = os.path.split(chunk_name)
        output_path = os.path.join(output_dir, output_subdir)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, chunk_name), 'w', encoding="utf-8") as fout:
            json.dump(dials, fout, indent=2, ensure_ascii=False)
        print(f"{output_subdir}: {chunk_name}, dial_num: {num} -> {len(dials)}")
        num_origin += num
        num_merged += len(dials)
    for split, schema in merged_schema.items():
        with open(os.path.join(output_dir, split, "schema.json"), 'w', encoding="utf-8") as fout:
            json.dump(schema, fout, indent=2, ensure_ascii=False)
        print(f"split: {split}, schema_num: {len(schema)}")
    print(f"num_origin: {num_origin}, num_merged: {num_merged}")
    print(f"variant_counter: {json.dumps(variant_counter, indent=2)}")
