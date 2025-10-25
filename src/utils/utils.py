# coding: utf-8
import os
import re
import logging
import json
import hashlib
import numpy as np
import pandas as pd
import bisect
from datetime import datetime
import pytz
import difflib
import tempfile
import portalocker
import html
from collections import OrderedDict


def set_to_list(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def clean_replace(s, r, t, forward=True, backward=False):
    def clean_replace_single(s, r, t, forward, backward, sidx=0):
        # idx = s[sidx:].find(r)
        idx = s.find(r)
        if idx == -1:
            return s, -1
        idx_r = idx + len(r)
        if backward:
            while idx > 0 and s[idx - 1]:
                idx -= 1
        elif idx > 0 and s[idx - 1] != ' ':
            return s, -1

        if forward:
            while idx_r < len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
                idx_r += 1
        elif idx_r != len(s) and (s[idx_r].isalpha() or s[idx_r].isdigit()):
            return s, -1
        return s[:idx] + t + s[idx_r:], idx_r

    # source, replace, target = s, r, t
    # count = 0
    sidx = 0
    while sidx != -1:
        s, sidx = clean_replace_single(s, r, t, forward, backward, sidx)
        # count += 1
        # print(s, sidx)
        # if count == 20:
        #     print(source, '\n', replace, '\n', target)
        #     quit()
    return s


def py2np(list):
    return np.array(list)


def write_dict(fn, dic):
    with open(fn, 'w') as f:
        json.dump(dic, f, indent=2)


def f1_score(label_list, pred_list):
    tp = len([t for t in pred_list if t in label_list])
    fp = max(0, len(pred_list) - tp)
    fn = max(0, len(label_list) - tp)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


def split_long_dial(turn_lengths, model_max_length, history_ratio=0.5):
    header_length = turn_lengths[0]
    turn_lengths = turn_lengths[1]
    max_token_limit = model_max_length - header_length
    history_token_limit = int(max_token_limit * history_ratio)

    short_spans = []
    start_index = 0
    prev_ed = 0
    while True:
        end_index = bisect.bisect_right(np.cumsum(turn_lengths[start_index:]), max_token_limit)
        end_index += start_index

        end_index = max(end_index, prev_ed + 1)

        st = bisect.bisect_right(np.cumsum(turn_lengths[:end_index][::-1]), max_token_limit)
        st = end_index - st

        short_spans.append((st, prev_ed, end_index))
        prev_ed = end_index
        if end_index == len(turn_lengths):
            break
        # start_index = bisect.bisect_right(np.cumsum(turn_lengths[:end_index][::-1]),
        #                                   max(history_token_limit, max_token_limit - new_tokens))
        start_index = bisect.bisect_right(np.cumsum(turn_lengths[:end_index][::-1]),
                                          history_token_limit)
        start_index = end_index - max(start_index, 1)
    return short_spans, history_token_limit


def evenly_split_list(input_list, num_groups):
    avg = len(input_list) // num_groups
    remainder = len(input_list) % num_groups

    groups = []
    index = 0

    for _ in range(num_groups):
        group_size = avg + (1 if remainder > 0 else 0)
        groups.append(input_list[index:index + group_size])
        index += group_size
        remainder -= 1

    return groups


def divide_into_groups(dials, N):
    dials = sorted(dials, key=lambda x: len(x), reverse=True)  # 按照对话长度降序排列
    chunks = [[] for _ in range(N)]  # 创建N个空组
    sums = [0] * N  # 用于存储每个组的总和

    for dial in dials:
        min_sum_index = sums.index(min(sums))  # 找到总和最小的组的索引
        chunks[min_sum_index].append(dial)  # 将数字添加到总和最小的组
        sums[min_sum_index] += len(dial)  # 更新总和

    print(f"Total {sum(sums)} turns -> {sum(sums) / N} turns per group")
    for i, chunk in enumerate(chunks):
        print(f"Group {i} has {len(chunk)} dials: {list(map(len, chunk))} -> {sums[i]}")

    return chunks


def sort_by_ids(origin_ids, unsorted_data, key_func):
    hash_table = {key_func(d): d for d in unsorted_data}
    sorted_data = [hash_table[id] for id in origin_ids]
    return sorted_data


def seconds_to_hms(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


def get_progress(passed_time, passed_turns, total_turns):
    secs_per_turn = passed_time / passed_turns
    hours, minutes, seconds = seconds_to_hms(passed_time)
    passed_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    total_time = total_turns * secs_per_turn
    hours, minutes, seconds = seconds_to_hms(total_time - passed_time)
    remain_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"[{passed_time_str}<{remain_time_str}, {secs_per_turn:.2f}s/it]"


def highlight(text, color=None):
    match color:
        case "red":
            return f"\033[1;31;40m{text}\033[0m"
        case "green":
            return f"\033[1;32;40m{text}\033[0m"
        case "yellow":
            return f"\033[1;33;40m{text}\033[0m"
        case _:
            return f"{text}"


def compute_entroy(counts):
    counts = np.array(counts)
    probs = counts / np.sum(counts)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def count_and_entropy(stats):
    key2entropy = {}
    for key, v_list in stats.items():
        if len(v_list) == 0:
            continue
        counter = pd.Series(v_list).value_counts()
        entropy = compute_entroy(counter)
        key2entropy[key] = [counter.to_dict(), entropy]
    return key2entropy


def get_md5_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def get_sha256_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_domain_and_slot(domain_or_slot, entire_constraint):
    if '-' in domain_or_slot:
        print(highlight(f"WARNING: '{domain_or_slot}' contains '-', which is not allowed.", "red") + f"\nCurrent constraint: {entire_constraint}")
    return domain_or_slot.replace('-', '_')


def messup_dataset_info(dataset_info):
    # mess up the dataset info
    new_dataset_info = {}
    for k, v in dataset_info.items():
        if isinstance(v, dict):
            element_type = list(v.values())[0]
            if isinstance(element_type, list):
                pass
            elif isinstance(element_type, set):
                new_dataset_info[k] = {kk: list(vv) for kk, vv in v.items()}
            else:
                new_dataset_info[k] = dict(sorted(v.items(), key=lambda x: x[1], reverse=True))
        elif isinstance(v, list):
            pass
        elif isinstance(v, int):
            new_dataset_info[k] = v
        elif isinstance(v, set):
            new_dataset_info[k] = list(v)
    return new_dataset_info


def get_ip_address():
    import socket
    return socket.gethostbyname(socket.gethostname())


def get_current_time(timezone="Asia/Shanghai"):
    tz = pytz.timezone(timezone)
    return datetime.now(tz)


def get_current_time_str(timezone="Asia/Shanghai"):
    return get_current_time(timezone).strftime("%Y-%m-%d %H:%M:%S")


def merge_overlapping_segments(segments):
    # 将差异片段索引列表按起始位置排序
    sorted_segments = sorted(segments)

    # 初始化合并后的差异片段索引列表
    merged_segments = []

    # 遍历差异片段索引列表
    for start, end in sorted_segments:
        # 如果当前差异片段与上一个合并后的差异片段有重叠部分，合并它们
        if merged_segments and start <= merged_segments[-1][1]:
            merged_segments[-1] = (merged_segments[-1][0], max(end, merged_segments[-1][1]))
        else:
            # 否则，将当前差异片段添加到合并后的差异片段列表中
            merged_segments.append((start, end))

    return merged_segments


def find_diff_segments(strings):
    # 初始化差异片段索引字典，用于存储每个字符串的差异片段索引列表
    diff_segments_dict = {}

    # 遍历字符串列表
    for i, reference_string in enumerate(strings):
        # 初始化差异片段索引列表
        diff_segments = []

        # 遍历其他字符串，与参考字符串比较
        for j, string in enumerate(strings):
            # 跳过与自身的比较
            if i == j:
                continue

            # 使用 difflib 模块的 SequenceMatcher 类比较两个字符串
            matcher = difflib.SequenceMatcher(None, reference_string, string)

            # 获取差异块
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                # 如果是替换或插入操作，将差异片段的索引加入到结果列表中
                if tag != 'equal':
                    diff_segments.append((i1, i2))

        # 将差异片段索引列表中的重叠部分合并
        merged_diff_segments = merge_overlapping_segments(diff_segments)
        # 将差异片段索引列表添加到字典中
        diff_segments_dict[reference_string] = merged_diff_segments

    return diff_segments_dict


def atomic_json_update(file_path, new_data):
    backup_file_path = file_path + '.bak'
    with open(file_path, 'r+', encoding='utf-8') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        data = json.load(file)
        backup_data = data.copy()
        with open(backup_file_path, 'w', encoding='utf-8') as backup_file:
            json.dump(backup_data, backup_file, ensure_ascii=False, indent=2)
        data.append(new_data)
        file.seek(0)
        file.truncate()
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.flush()
        portalocker.unlock(file)


def extract_values_in_concrete_resp(original_text, delexicalized_text):
    original_text = original_text.strip() + " $$$$"
    delexicalized_text = delexicalized_text.strip() + " $$$$"
    not_founds = []
    # 定义正则表达式来匹配槽位和值
    window_size = 10
    slot_pattern = re.compile(r'\[value_(.*?)\]')

    # 找出去词汇化文本中的槽位
    slots = slot_pattern.findall(delexicalized_text)

    # 为了提取值，我们需要构造正则表达式来匹配原始文本中的值
    values = {}
    current_pos_in_original_text = 0
    current_pos_in_delexicalized_text = 0

    for slot in slots:
        slot_placeholder = f"[value_{slot}]"
        # 找到槽位在去词汇化文本中的位置
        slot_pos = delexicalized_text.find(slot_placeholder, current_pos_in_delexicalized_text)

        # 找到对应槽位前后的文本
        # start_text = delexicalized_text[current_pos_in_delexicalized_text:slot_pos]
        start_text = delexicalized_text[max(current_pos_in_delexicalized_text, slot_pos - window_size):slot_pos]
        next_slot_pos = delexicalized_text.find('[value_', slot_pos + len(slot_placeholder))
        current_pos_in_delexicalized_text = slot_pos + len(slot_placeholder)
        if next_slot_pos == -1:
            next_slot_pos = len(delexicalized_text)
        # end_text = delexicalized_text[current_pos_in_delexicalized_text:next_slot_pos]
        end_text = delexicalized_text[current_pos_in_delexicalized_text: min(current_pos_in_delexicalized_text + window_size, next_slot_pos)]

        # 构造正则表达式来匹配原始文本中的值
        value_pattern = re.escape(start_text) + r'(.+?)' + re.escape(end_text)
        match = re.search(value_pattern, original_text[current_pos_in_original_text:])

        if match:
            values[slot] = match.group(1).strip()
            current_pos_in_original_text = original_text.find(match.group(1), current_pos_in_original_text) + len(match.group(1))
        else:
            not_founds.append(slot)

    return values, not_founds


def deduplicate_string(text: str) -> str:
    """
    对字符串进行深度去重，处理重复的短语、单词和字符。

    Args:
        text: 需要处理的原始字符串。

    Returns:
        去重后的干净字符串。
    """
    if not text:
        return ""

    # --- 1. 预处理：清理多余的空白和特殊字符 ---
    # 将特殊字符替换为空格，然后合并所有连续的空白为一个空格
    processed_text = re.sub(r'\s+', ' ', text.replace('│', ' ')).strip()

    # --- 2. 去除长短语重复 ---
    # 模式 r'(.{10,})\1+' 解释:
    # (.{10,}) : 捕获一个长度至少为10的任意字符序列（作为第1组）
    # \1+       : 匹配前面第1组捕获内容一次或多次
    # 这个循环确保了像 "abcabcabc" 这样的情况能被彻底简化为 "abc"
    previous_text = ""
    while previous_text != processed_text:
        previous_text = processed_text
        processed_text = re.sub(r'(.{10,})(\1)+', r'\1', processed_text)

    # --- 3. 去除单词重复 ---
    # 模式 r'(\b\w+\b)(?:\s+\1)+' 解释:
    # (\b\w+\b) : 捕获一个独立的单词（\b是单词边界）
    # (?:\s+\1)+: 匹配一个或多个空格后跟相同的单词，重复一次或多次
    processed_text = re.sub(r'(\b\w+\b)(?:\s+\1)+', r'\1', processed_text)

    # --- 4. 去除长串的单个字符重复 ---
    # 模式 r'(.)\1{4,}' 解释:
    # (.)  : 捕获任意单个字符
    # \1{4,}: 匹配该字符额外重复4次或更多次（总共5次以上）
    # 将长串重复（如 '1111111'）压缩为单个字符（'1'）
    processed_text = re.sub(r'(.)\1{4,}', r'\1', processed_text)

    return processed_text


def fuzzy_match_domain(domain2possible, domain):
    res_doms = []
    if not domain2possible:
        return res_doms

    domain = domain.strip().lower()
    for key, values in domain2possible.items():
        for val in values:
            if val.lower() in domain:
                res_doms.append(key)
                break
    return res_doms


if __name__ == '__main__':
    original_text = "sorry , i am the supervisor . i have booked you 5 more tickets . the extra cost is 300.39 pounds , payable at the station . the reference number is 0g76cz16 . anything else ? is 07218068540"
    delexicalized_text = "[value_general] , i am the supervisor . i have booked you [value_people] more tickets . the extra cost is [value_price] , payable at the station . the reference number is [value_reference] . anything else ? is [value_phone]"
    values = extract_values_in_concrete_resp(original_text, delexicalized_text)
    print(values)

    exit(0)

    match_results = [
        {
            "address": "G4 Cambridge Leisure Park Clifton Way Cherry Hinton",
            "area": "south",
            "food": "italian",
            "location": "52.190176,0.13699",
            "phone": "01223 323737",
            "pricerange": "moderate",
            "postcode": "C.B 1, 7 D.Y",
            "type": "restaurant",
            "id": "19196",
            "name": "pizza hut cherry hinton"
        },
        {
            "address": "64 Cherry Hinton Road Cherry Hinton",
            "area": "south",
            "food": "indian",
            "location": "52.188747,0.138941",
            "phone": "01223 412299",
            "pricerange": "expensive",
            "postcode": "C.B 1, 7 A.A",
            "type": "restaurant",
            "id": "19191",
            "name": "taj tandoori"
        },
        {
            "address": "152 - 154 Hills Road",
            "area": "south",
            "food": "modern european",
            "location": "52.18889,0.13589",
            "phone": "01223 413000",
            "pricerange": "moderate",
            "postcode": "C.B 2, 8 P.B",
            "type": "restaurant",
            "id": "14731",
            "name": "restaurant alimentum"
        },
        {
            "address": "529 Newmarket Road Fen Ditton",
            "area": "east",
            "food": "chinese",
            "location": "52.212992,0.157569",
            "phone": "01223 248882",
            "pricerange": "expensive",
            "postcode": "C.B 5, 8 P.A",
            "type": "restaurant",
            "id": "19273",
            "name": "yu garden"
        },
    ]
    # max_display = 2
    # stats = {
    #     "food": ['chinese', 'japanese', 'vietnamese', 'italian', 'italian'],
    #     "area": ['east', 'east', 'north', 'east', 'north', 'east'],
    #     "pricerange": ['cheap', 'cheap', 'cheap', 'cheap', 'cheap', 'cheap'],
    # }
    # key2entropy = compute_entropy(stats)
    # print(json.dumps(key2entropy, indent=2))
    # texts = []
    # # TODO: control the number of slots to display
    # for slot, stat in sorted(key2entropy.items(), key=lambda x: x[1][1], reverse=True):
    #     texts.append(f"{len(stat[0])} distinct [{slot}] " + ' '.join(
    #         [f"{v} ({c} items)" for v, c in sorted(stat[0].items(), key=lambda x: x[1], reverse=True)[:5]]))
    # summary = f"statistics for each attribute - " + ' ; '.join(texts)
    # items_str = '\n        '.join([f"[{idx + 1} of {len(match_results)}] {json.dumps(item)}" for idx, item in
    #                                enumerate(match_results[:max_display])])
    # summary += f"\n        {items_str}"
    # print(summary)
    #
    # exit(0)

    turn_lengths = [162, [305, 298, 370, 390, 411, 422, 421, 429, 452, 478, 496, 503, 520, 560, 580]]
    model_max_length = 4096
    history_ratio = 0
    print(f"{turn_lengths} -> {sum([turn_lengths[0], *turn_lengths[1]])}")
    for hr in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        short_spans, history_token_limit = split_long_dial(turn_lengths, model_max_length, hr)
        print(f"\nhistory_ratio: {hr}, history_token_limit: {history_token_limit}")
        prev_ed = 0
        for st, sp, ed in short_spans:
            history_tot = sum(turn_lengths[1][st:sp])
            tot = sum(turn_lengths[1][st:ed])
            curr_tot = sum(turn_lengths[1][sp:ed])
            print(f"({st}, {sp}, {ed}): {turn_lengths[1][st:sp]}={history_tot} ; {turn_lengths[1][sp:ed]}={curr_tot} -> {tot}, "
                  f"{model_max_length - tot - turn_lengths[0] - 30}")
            prev_ed = ed