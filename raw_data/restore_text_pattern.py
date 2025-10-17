import re
import json
from collections import defaultdict, Counter
from typing import List, Dict, Tuple


# --- 步驟 1: 定義簡寫映射關係 (與第一個問題相同) ---
# 這是我們的基礎知識庫，用於識別 Text A 中的簡寫
CONTRACTED_TO_EXPANDED: Dict[str, List[str]] = {
    "i'm": ["i", "am"], "i'd": ["i", "would"], "you'd": ["you", "would"], "he'd": ["he", "would"], "she'd": ["she", "would"], "it'd": ["it", "would"], "we'd": ["we", "would"], "they'd": ["they", "would"],
    "i'll": ["i", "will"], "you'll": ["you", "will"], "that'll": ["that", "will"], "he'll": ["he", "will"], "she'll": ["she", "will"], "it'll": ["it", "will"], "we'll": ["we", "will"], "they'll": ["they", "will"], "there'll": ["there", "will"],
    "i've": ["i", "have"], "we've": ["we", "have"], "they've": ["they", "have"], "you've": ["you", "have"],
    "you're": ["you", "are"], "we're": ["we", "are"], "they're": ["they", "are"],
    "it's": ["it", "is"], "there's": ["there", "is"], "here's": ["here", "is"], "that's": ["that", "is"], "what's": ["what", "is"], "where's": ["where", "is"], "how's": ["how", "is"],
    "there're": ["there", "are"], "what're": ["what", "are"], "where're": ["where", "are"],
    "can't": ["can", "not"], "cannot": ["can", "not"], "couldn't": ["could", "not"],
    "don't": ["do", "not"], "doesn't": ["does", "not"], "didn't": ["did", "not"], "hasn't": ["has", "not"], "haven't": ["have", "not"], "hadn't": ["had", "not"],
    "isn't": ["is", "not"], "aren't": ["are", "not"], "wasn't": ["was", "not"], "weren't": ["were", "not"], "won't": ["will", "not"], "wouldn't": ["would", "not"], "shouldn't": ["should", "not"], "mightn't": ["might", "not"], "mustn't": ["must", "not"], "needn't": ["need", "not"], "shan't": ["shall", "not"],
    "let's": ["let", "us"],
    "restaurants": ["restaurant", "-", "s"], "hotels": ["hotel", "-", "s"], "laptops": ["laptop", "-", "s"], "dinners": ["dinner", "-", "s"], "cheaper": ["cheap", "-", "er"], "lunches": ["lunch", "-", "s"], "breakfasts": ["breakfast", "-", "s"],
    "prices": ["price", "-", "s"], "places": ["place", "-", "s"], "areas": ["area", "-", "s"], "locations": ["location", "-", "s"], "types": ["type", "-", "s"], "stars": ["star", "-", "s"], "venues": ["venue", "-", "s"], "ranges": ["range", "-", "s"], "meals": ["meal", "-", "s"], "policies": ["policy", "-", "s"], "kids": ["kid", "-", "s"], "adults": ["adult", "-", "s"], "kidfriendly": ["kid", "friendly"], "cards": ["card", "-", "s"], "inches": ["inch", "-", "s"], "uses": ["use", "-", "s"], "dimensions": ["dimension", "-", "s"], "driverange": ["drive", "range"], "includes": ["include", "-", "s"],
    "computers": ["computer", "-", "s"], "machines": ["machine", "-", "s"], "families": ["family", "-", "s"], "colleges": ["college", "-", "s"], "ratings": ["rating", "-", "s"], "constraints": ["constraint", "-", "s"], "pricerange": ["price", "range"], "batteryrating": ["battery", "rating"], "requirements": ["requirement", "-", "s"], "drives": ["drive", "-", "s"], "specifications": ["specification", "-", "s"], "weighrange": ["weight", "range"], "harddrive": ["hard", "drive"], "batterylife": ["battery", "life"], "businesses": ["business", "-", "s"], "hours": ["hour", "-", "s"],
    "cheaply": ["cheap", "-", "ly"], "moderately": ["moderate", "-", "ly"], "expensively": ["expensive", "-", "ly"],
    "anywhere": ["any", "where"], "somewhere": ["some", "where"], "nowhere": ["no", "where"],
    "gonna": ["gon", "na"], "wanna": ["wan", "na"], "gotta": ["got", "ta"],
    "children": ["child", "-", "s"], "children's": ["child", "-", "s", "'s"],
}

EXPANDED_TO_CONTRACTEDS: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
for contracted, expanded in CONTRACTED_TO_EXPANDED.items():
    EXPANDED_TO_CONTRACTEDS[tuple(expanded)].append(contracted)
print(f"Expands with multiple contractions:")
for k, v in EXPANDED_TO_CONTRACTEDS.items():
    if len(v) > 1:
        print(f"  {k} -> {v}")


def smart_tokenize(text: str) -> List[str]:
    """將單詞、數字、簡寫和標點符號分開。"""
    # return re.findall(r"[\w']+|[^\s\w]", text)
    return re.findall(r"[\w']+|\[value_\w+\]|[^\s\w]", text)


def build_contraction_map_from_text_a(tokens_a: List[str]) -> Dict[Tuple[str, ...], str]:
    """
    掃描 Text A 的分詞列表，構建一個從擴展形式到原始簡寫形式的映射。
    例如: {'(i, am)': "I'm", '(can, not)': "can't"}
    """

    def simple_split(token: str) -> List[List[str]]:
        # more possible splits
        # i'd -> i 'd, can't -> can 't, can't -> ca n't
        splits = []
        if "'" in token:
            splits.append(token.replace("'", " '").split())
        if "n't" in token:
            splits.append(token.replace("n't", " n't").split())
        return splits

    contraction_map = {}
    for token in tokens_a:
        lookup_token = token.lower()
        if lookup_token in CONTRACTED_TO_EXPANDED:
            # 鍵是小寫的擴展形式元組，值是 Text A 中帶有原始大小寫的簡寫
            expanded_key = tuple(CONTRACTED_TO_EXPANDED[lookup_token])
            contraction_map[expanded_key] = token
            for split in simple_split(lookup_token):
                if tuple(split) != expanded_key:
                    contraction_map[tuple(split)] = token
    return contraction_map


def restore_contractions_from_map(tokens_b: List[str], contraction_map: Dict[Tuple[str, ...], str]) -> List[str]:
    """
    (第一階段) 遍歷 Text B 的分詞，使用從 Text A 構建的地圖來恢復簡寫。
    """
    result_tokens = []
    i = 0
    # 將 map 的鍵按長度排序，優先匹配長的（例如 "i would have" vs "i would"）
    sorted_keys = sorted(contraction_map.keys(), key=len, reverse=True)

    while i < len(tokens_b):
        found_match = False
        for key in sorted_keys:
            key_len = len(key)
            # 檢查 Text B 從當前位置開始的片段是否與 map 中的鍵匹配
            if i + key_len <= len(tokens_b) and tuple(t.lower() for t in tokens_b[i:i + key_len]) == key:
                # 找到了匹配！添加恢復後的簡寫
                result_tokens.append(contraction_map[key])
                i += key_len  # 跳過被合併的詞
                found_match = True
                break

        if not found_match:
            # 沒有找到匹配的簡寫，直接添加當前詞
            result_tokens.append(tokens_b[i])
            i += 1

    return result_tokens


def normalize_final_text(tokens: List[str]) -> str:
    """
    (第二階段) 對已經恢復了簡寫的分詞列表進行通用規範化。
    """
    # 1. 智能拼接，處理標點空格
    text = " ".join(tokens)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # 2. 句子首字母大寫
    sentences = re.split(r'([.!?]\s*)', text)
    capitalized_sentences = []
    sentence_pairs = [sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else '')
                      for i in range(0, len(sentences), 2)]

    for s in sentence_pairs:
        s = s.strip()
        if s:
            capitalized_sentences.append(s[0].upper() + s[1:])

    text = ' '.join(capitalized_sentences)

    # 3. 處理獨立的 ' i ' (雖然簡寫恢復可能已經處理了 I'm, I'd, 但 I can... 這種情況仍需處理)
    text = re.sub(r'\bi\b', 'I', text)

    # 4. 處理所有者 's 的空格問題 (e.g., "Kohinoor 's" -> "Kohinoor's")
    text = re.sub(r" 's\b", "'s", text)

    # 5. 處理 [value_{slot}] 的后缀问题 (e.g., "[value_type] - s" -> "[value_type]", "[value_pricerange] - ly" -> "[value_pricerange]")
    text = re.sub(r'\]\s+-\s*s\b', ']', text)
    text = re.sub(r'\]\s+-\s*ly\b', ']', text)

    return text


def normalize_and_restore(text_a: str, text_b: str, verbose=False) -> str:
    """
    主函數：規範化 Text B，並根據 Text A 恢復簡寫。
    """
    # 如果 Text B 中没有 [value_{slot}] 模式, 直接返回 Text A
    if not re.search(r'\[value_\w+\]', text_b):
        return text_a

    # 預處理：對 A 和 B 進行分詞
    tokens_a = smart_tokenize(text_a)
    tokens_b = smart_tokenize(text_b)

    # 從 Text A 構建簡寫地圖
    contraction_map = build_contraction_map_from_text_a(tokens_a)

    # --- 第一階段 ---
    tokens_b_with_contractions = restore_contractions_from_map(tokens_b, contraction_map)

    # --- 第二階段 ---
    final_text = normalize_final_text(tokens_b_with_contractions)

    if verbose:
        print("Original Text A:", text_a)
        print("Original Text B:", text_b)
        print(f"Tokens A: {tokens_a}")
        print(f"Tokens B: {tokens_b}")
        print(f"Contraction Map: {contraction_map}")
        print(f"Tokens B with Contractions: {tokens_b_with_contractions}")
        print(f"Final Restored Text: {final_text}")
        print("-" * 50)

    return final_text


def inverse_normalize(text: str) -> str:
    # strip, lowercase, expand contractions, add spaces around punctuation
    text = text.strip().lower()
    tokens = smart_tokenize(text)
    expanded_tokens = []
    for token in tokens:
        if token in CONTRACTED_TO_EXPANDED:
            expanded_tokens.extend(CONTRACTED_TO_EXPANDED[token])
        else:
            expanded_tokens.append(token)
    text = " ".join(expanded_tokens)
    # add space before remaining "'"
    text = re.sub(r"(?<!\s)'", " '", text)
    return text


def bleu_between_raw_and_processed(raw: List[str], processed: List[str], verbose=False) -> List[str]:
    from src.utils.eval import BLEUScorer

    bleu_scorer = BLEUScorer()
    wrap_generated = [[_] for _ in raw]
    wrap_truth = [[_] for _ in processed]
    sc = bleu_scorer.score(zip(wrap_generated, wrap_truth))
    return sc


if __name__ == '__main__':
    data_file = "raw_data/multiwoz2.1/data_for_space.json"
    data = json.load(open(data_file, 'r'))

    raw_concrete_column = "original_nodelx_resp"
    for raw_column, processed_column in [
        ["original_nodelx_resp", "nodelx_resp"],
        ["restored_resp", "resp"]
    ]:
        max_pairs = 10000
        raw, processed = [], []
        for dialog_id, dialog in data.items():
            if max_pairs <= 0:
                break
            for turn in dialog["log"]:
                if max_pairs <= 0:
                    break
                max_pairs -= 1
                if raw_column in turn and processed_column in turn:
                    restored_processed = normalize_and_restore(turn[raw_concrete_column], turn[processed_column])
                    # raw_temp = turn[raw_column].strip().lower()
                    raw_temp = inverse_normalize(turn[raw_column])
                    # prc_temp = turn[processed_column].strip().lower()
                    prc_temp = inverse_normalize(restored_processed)
                    sc = bleu_between_raw_and_processed([raw_temp], [prc_temp], verbose=False)
                    if max_pairs < 20:
                        print(f"[RAW] Before: {turn[raw_column]}\n[RAW] After:  {raw_temp}\n[PRC] Before: {turn[processed_column]}\n[PRC] After:  {prc_temp}\nBLEU:    {sc:.2f}\n" + "-" * 20)
                    raw.append(raw_temp)
                    processed.append(prc_temp)

        bleu_score = bleu_between_raw_and_processed(raw, processed, verbose=False)
        print(f"BLEU score between {len(raw)} pairs of '{raw_column}' and '{processed_column}': {bleu_score:.2f}", end="\n\n")

    exit(0)

    sentence_pairs = [
        ("I've already set, i like! It's a [value_name] with a moderately priced restaurants, located at [value_address]..",
         "I 've already set , i like ! it   is a [value_name] with a [value_pricerange] priced restaurant -s, located at [value_address] ."),
        ("i'm sorry, I can't find any place with Corsica food. Do you have a second choice?",
         "i 'm sorry , i ca n't find any place   with corsica food . do you have a second choice ?"),
        ("Kohinoor's address is 74 Mill Road City Centre and the phone number is 01223 323639. That'll sd. May I help you with anything else?",
         "[value_name] 's address is [value_address] and the phone number is [value_phone_number] . that will  that 'll . may i help you with anything else ?"),
        ("You're welcome. Goodbye.", "you 're welcome . goodbye .")
    ]
    for text_a, text_b in sentence_pairs:
        restored_text = normalize_and_restore(text_a, text_b, verbose=True)
