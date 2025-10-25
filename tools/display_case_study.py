# coding: utf-8

import os
import sys
import json


def highlight(text, color=None):
    match color:
        case "red":
            return f"\033[1;31;40m{text}\033[0m"
        case "green":
            return f"\033[1;32m{text}\033[0m"
        case "yellow":
            return f"\033[1;33m{text}\033[0m"
        case _:
            return f"{text}"


keywords = [
    "Domain Schema", "Intent Schema", "system\n", "user\n", "assistant\n", "AI assistant\n", "informable slots", "requestable slots", "required_slots", "optional_slots", "result_slots", "intents",
]
green_words = ["identify the domains", "Select the correct intent", "maintain the user's needs", "summarize the actions", "Generate delexicalized", "Generate concrete"]
special_tokens_dict, _ = json.load(open("raw_data/special_tokens.json", "r", encoding="utf-8"))
keywords += [word for word in special_tokens_dict.values() if word.startswith("<|beginof")]
special_tokens_dict, _ = json.load(open("raw_data/qwen25_special_tokens.json", "r", encoding="utf-8"))
keywords += [word for word in special_tokens_dict.values() if word.startswith("<sos_") or word.startswith("<eos_")]
keywords = set(keywords)

def highlight_keywords(text):
    for keyword in green_words:
        text = text.replace(keyword, highlight(keyword, "green"))
    for keyword in keywords:
        text = text.replace(keyword, highlight(keyword, "yellow"))
    return text


if __name__ == '__main__':
    meta_file = "dial_meta.json"
    with open(meta_file, "r", encoding="utf-8") as fin:
        meta = json.load(fin)

    out_file = os.path.join(os.path.dirname(meta_file), f"{os.path.basename(meta_file).split('.')[0]}_display.txt")
    with open(out_file, "w", encoding="utf-8") as fout:
        for dial_id, dial_meta in meta.items():
            fout.write(f"------------------ {dial_id} ------------------\n")
            fout.write(f"Prompt + Turn Prompt:\n")
            fout.write(f"{dial_meta['prompt']}{dial_meta['turn_prompt']}\n")
            if "history_turns" in dial_meta:
                fout.write(f"Full History:\n")
                fout.write(f"{''.join(dial_meta['history_turns'])}\n")
            if "prev_turn_text" in dial_meta:
                ptt = dial_meta["prev_turn_text"]
                ptt = ''.join(ptt) if isinstance(ptt, list) else ptt
                fout.write(f"Previous Turn Text:\n")
                fout.write(f"{ptt}\n")
            fout.write(f"\n")
    if "history_turns" in dial_meta:
        print(highlight_keywords(f"{''.join(dial_meta['history_turns'])}"))
        print(f"-------------------------------------------\n")
    print(f"Last Turn Prompt:\n")
    print(highlight_keywords(f"{dial_meta['prompt']}{dial_meta['turn_prompt']}"))
    if "prev_turn_text" in dial_meta:
        ptt = dial_meta["prev_turn_text"]
        ptt = ''.join(ptt) if isinstance(ptt, list) else ptt
        print(f"-------------------------------------------\n")
        print(f"Previous Turn Text:\n")
        print(highlight_keywords(ptt))
