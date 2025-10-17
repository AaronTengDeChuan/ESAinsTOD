import os
import re
import json
import difflib
from tqdm import tqdm

from restore_text_pattern import normalize_and_restore


def is_same_message(text1, text2, ratio_threshold=0.6):
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    toks1 = re.split(r'\W+', text1)
    toks2 = re.split(r'\W+', text2)
    # fuzzy match
    seq = difflib.SequenceMatcher(None, toks1, toks2)
    return seq.ratio() > ratio_threshold, seq.ratio()


if __name__ == '__main__':
    origin_file = "multiwoz2.1_released/data.json"
    target_file = "multiwoz2.1/data_for_space.json"

    origin_data = json.load(open(origin_file, 'r', encoding='utf-8'))
    origin_data = {k.lower(): v for k, v in origin_data.items()}
    target_data = json.load(open(target_file, 'r', encoding='utf-8'))

    ratio_threshold = 0.6
    num_diff = 0
    inserted, total = 0, 0
    for dialog_id, dialog in tqdm(target_data.items(), desc="Processing dialogs"):
        total += 1
        dialog_id = dialog_id.lower()
        assert dialog_id in origin_data, f"{dialog_id} not in origin data '{origin_file}'."

        ori_logs = origin_data[dialog_id]["log"]
        exchange_turn_ids = []
        prev_role = None
        for i, log in enumerate(ori_logs):
            role = "system" if log["metadata"] else "user"
            if role != prev_role:
                exchange_turn_ids.append(i)
                prev_role = role
        ori_logs = [ori_logs[i] for i in exchange_turn_ids]
        ori_len = len(ori_logs)

        tgt_logs = dialog["log"]
        tgt_len = len(tgt_logs) * 2

        is_user = not ori_logs[0]["metadata"]
        assert is_user, f"The first turn should be user turn in {dialog_id} in '{origin_file}'."

        if ori_len != tgt_len:
            print(f"Length mismatch for {dialog_id}: origin {ori_len}, target {tgt_len}")
            print(f"\tOrigin roles: {[ 'system' if log['metadata'] else 'user' for log in ori_logs ]}")
            continue

        for i, tgt_log in enumerate(tgt_logs):
            ori_user_msg = ori_logs[i * 2]["text"].strip()
            ori_sys_msg = ori_logs[i * 2 + 1]["text"].strip()
            tgt_log["original_user"] = ori_user_msg
            same_flag, ratio = is_same_message(ori_user_msg, tgt_log["user"], ratio_threshold)
            if not same_flag:
                print(f"User message mismatch in {dialog_id} at turn {i} with ratio {ratio:.2f}:")
                print(f"\tOrigin: {ori_user_msg}")
                print(f"\tTarget: {tgt_log['user']}")
                num_diff += 0.5
            tgt_log["restored_resp"] = normalize_and_restore(ori_sys_msg, tgt_log["resp"].strip())
            tgt_log["original_nodelx_resp"] = ori_sys_msg

            same_flag, ratio = is_same_message(ori_sys_msg, tgt_log["nodelx_resp"], ratio_threshold)
            if not same_flag:
                print(f"System message mismatch in {dialog_id} at turn {i} with ratio {ratio:.2f}:")
                print(f"\tOrigin: {ori_sys_msg}")
                print(f"\tTarget: {tgt_log['nodelx_resp']}")
                num_diff += 0.5

        inserted += 1

    print(f"Inserted {inserted} / {total} dialogs.")
    print(f"Total {num_diff} turn differences found with ratio threshold {ratio_threshold}.")

    with open(target_file, 'w', encoding='utf-8') as fout:
        json.dump(target_data, fout, ensure_ascii=False, indent=2)
