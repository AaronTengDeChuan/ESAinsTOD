import os
import re
import json
from collections import defaultdict, Counter


API_MAP = {
    'chat': 'chat',
    "restaurants_en_US_search": "restaurant_search",
    "restaurants_en_US_booking": "restaurant_booking",
    "hotels_en_US_search": "hotel_search",
    "hotels_en_US_booking": "hotel_booking",
    "attractions_en_US_search": "attraction_search",
    "weathers_en_US_search": "weather_search",
    "HKMTR_en": "HKMTR"
}

transactional_apis = ["restaurant_booking", "hotel_booking"]


def read_ontology(dataset_dir):
    # NOTE: manually process ontology files in 'knowledgebase/apis' folder by selecting "example values" or "possible values" for each input slot
    api_path = "knowledgebase/new_apis"
    omit_result_slots = ["description"]
    schema = {}
    domain2slots = defaultdict(set)
    for fn in os.listdir(os.path.join(dataset_dir, api_path)):
        api_name = fn.replace(".json", "")
        api_name = API_MAP.get(api_name, None)
        if api_name is None:
            continue
        with open(os.path.join(dataset_dir, api_path, fn), 'r', encoding="utf-8") as f:
            try:
                ontology = json.load(f)
            except Exception as e:
                print(f"[{api_name}]: {e}")
                exit()
            schema[api_name] = {}
            input_slots = set([slot["Name"] for slot in ontology["input"]])
            # process input slots
            input_schema = {}
            for slot in ontology["input"]:
                slot_name = slot["Name"]
                input_schema[slot_name] = {}
                if "Categories" in slot and len(slot["Categories"]) > 0:
                    values_type = "possible_values" if slot["Type"] == "Categorical" else "example_values"
                    input_schema[slot_name][values_type] = slot["Categories"]
                elif slot["Type"] == "Integer":
                    min_value, max_value = slot["Min"], slot["Max"]
                    all_integers = range(min_value, max_value + 1)
                    num_display = 4
                    if len(all_integers) <= num_display:
                        input_schema[slot_name]["possible_values"] = list(all_integers)
                    else:
                        half_display = num_display // 2
                        input_schema[slot_name]["example_values"] = list(all_integers[:half_display]) + list(all_integers[-half_display:])

            if len(ontology["required"]) > 0:
                # require_slots
                required_slots = set(ontology["required"])
                assert required_slots.issubset(input_slots), f"{api_name}: {required_slots - input_slots}"
                print(f"{api_name}: input_slots ({len(input_slots)}) - required_slots ({len(required_slots)}) = {input_slots - required_slots}")
                schema[api_name]["required_slots"] = input_schema
            else:
                schema[api_name]["informable slots"] = input_schema
            result_slots = [slot["Name"] for slot in ontology["output"] if slot["Name"] not in omit_result_slots]
            if api_name.startswith("HKMTR"):
                assert len(result_slots) == 0, f"{api_name}: {result_slots}"
                result_slots = ["shortest_path", "price", "estimated_time"]
            schema[api_name]["result_slots"] = result_slots
            print(f"{api_name}:\n\t{len(input_slots)} input slots - {input_slots},"
                  f"\n\t{len(result_slots)} result slots - {result_slots}")
            domain2slots[api_name].update(input_slots)
            domain2slots[api_name].update(result_slots)
            # special dst format
    # schema["slot_format_in_belief_state"] = {
    #     "relations": ["equal_to", "at_least", "not", "one_of"],
    #     "slot_1": {
    #         "relation": "one_of",
    #         "value": ["value_11", "value_12"]
    #     },
    #     "slot_2": {
    #         "relation": "equal_to",
    #         "value": ["value_21"]
    #     }
    # }
    schema["format_for_slot_value"] = {
        "format": {
            "relations": ["equal_to", "at_least", "not", "one_of"],
            "{slot_name}": "{relation}({' , '.join(slot_value_list)})"
        },
        "examples": {
            "slot_1": "one_of(value_11 , value_12)",
            "slot_2": "equal_to(value_21)"
        }
    }
    return schema, domain2slots


def dict_diff(dict1, dict2):
    # recursively compare two dicts
    diff = {}
    relations = []
    for k, v in dict1.items():
        if k == "relation":
            relations.append(v)
        if k not in dict2:
            if "value" in v and v["value"] == ["don't care"]:
                continue
            diff[k] = f"{v} (v.s.) None"
        else:
            if isinstance(v, dict):
                temp_diff, temp_relations = dict_diff(v, dict2[k])
                relations.extend(temp_relations)
                if len(temp_diff) > 0:
                    diff[k] = temp_diff
            else:
                if v != dict2[k]:
                    diff[k] = f"{v} (v.s.) {dict2[k]}"
    for k, v in dict2.items():
        if k not in dict1:
            if "relation" in v:
                relations.append(v["relation"])
            if "value" in v and v["value"] == ["don't care"]:
                continue
            diff[k] = f"None (v.s.) {v}"
    return diff, relations


def clean_belief_state(belief_state):
    # TODO: simply state to '{"attraction_search": {"location": "equal_to(Central District)", "type": "one_of(Events , Museums)", "name": "not(Shia Wong Hip)"}, "hotel_search": {"rating": "at_least(4)"}}'
    belief_state = {API_MAP[k]: v for k, v in belief_state.items()}
    simplified_state = {}
    for domain, state in belief_state.items():
        simplified_state[domain] = {}
        for slot, slot_dict in state.items():
            slot_dict['value'] = [str(v).replace("don't care", 'dont care') for v in slot_dict['value']]
            simplified_state[domain][slot] = f"{slot_dict['relation']}({' , '.join(slot_dict['value'])})"
    return simplified_state


ordered_system_actions = ["offer", "request", "inform", "confirm", "affirm", "notify_success", "greeting", "goodbye", "request_more", "notify_fail", "request_update"]
slot_following_actions = ["offer", "request", "inform", "confirm", "request_update"]


def textualize_system_action(domain, actions):
    if not actions:
        return ""
    sorted_actions = sorted(actions, key=lambda x: ordered_system_actions.index(x['act']))
    action_strs = [f"[{domain}]"]
    prev_act = None
    for action in sorted_actions:
        cur_act = action['act'].lower()
        if action['act'] in slot_following_actions and prev_act == action['act']:
            action_strs.append(action['slot'])
        elif prev_act == action['act']:
            print(f"[Consecutive Action] action '{action['act']}' should not appear twice")
        else:
            action_strs.append(f"[{cur_act}]")
            if action['act'] in slot_following_actions:
                if not action['slot'] and action['act'] != "confirm":
                    print(f"[Empty Slot] action '{action['act']}' should have a slot")
                else:
                    action_strs.append(action['slot'])
        prev_act = action['act']
    if set(action_strs[1:]).issubset({'[request_more]', '[greeting]', '[goodbye]'}):
        del action_strs[0]
    assert len(action_strs) > 0, f"Empty action: {actions}"
    return " ".join(action_strs)


def rephrase_system_action(sys_act):
    new_sys_act = sys_act.replace("[offer]", "[recommend]")
    return new_sys_act



def delexicalize_utterance(text, actions, dataset_info):
    expanded_text = f" {text} "
    corrected_text = expanded_text
    for act in sorted([act for act in actions if len(act["value"]) >= 1], key=lambda x: len(str(x["value"][0])), reverse=True):
        slot, value = act["slot"], str(act["value"][0])
        # consider word boundary and eacape special characters
        # exist = re.search(r"(?<=\s)" + re.escape(value) + r"(?=\s|\W)", expanded_text)
        matches = re.findall(r"(?<=\s|\W)" + re.escape(value) + r"(?=\s|\W)", expanded_text)
        if len(matches) > 1:
            # print(f"{act} occurs multiple times in '{text}'")
            dataset_info["value_multi_occurrence"][act['slot']] += 1
            continue
        elif len(matches) == 1:
            expanded_text = re.sub(r"(?<=\s|\W)" + re.escape(value) + r"(?=\s|\W)", f"[value_{slot}]", expanded_text)
            continue

        # head_exist = re.search(r"(?<=\s|\W)" + re.escape(value), expanded_text)
        head_matches = re.findall(r"(?<=\s|\W)" + re.escape(value), expanded_text)
        # tail_exist = re.search(re.escape(value) + r"(?=\s|\W)", expanded_text)
        tail_matches = re.findall(re.escape(value) + r"(?=\s|\W)", expanded_text)
        if len(head_matches + tail_matches) == 0:
            # print(f"{act} not in '{expanded_text}'")
            dataset_info["value_absent"][act['slot']] += 1
        elif len(head_matches + tail_matches) > 1:
            # print(f"{act} occurs multiple times in '{expanded_text}'")
            dataset_info["value_multi_occurrence"][act['slot']] += 1
        else:
            siamese = "tail" if len(head_matches) > 0 else "head"
            legal_suffixes = ["ing", "y", "st", "nd", "rd", "th", ]
            if siamese == "head":
                expanded_text = re.sub(re.escape(value) + r"(?=\s|\W)", f" [value_{slot}]", expanded_text)
                corrected_text = re.sub(re.escape(value) + r"(?=\s|\W)", f" {value}", corrected_text)
            else:
                end_idx = re.search(r"(?<=\s|\W)" + re.escape(value), expanded_text).end()
                suffix = expanded_text[end_idx:]
                for legal_suffix in legal_suffixes:
                    if suffix.startswith(legal_suffix):
                        expanded_text = re.sub(r"(?<=\s|\W)" + re.escape(value) + re.escape(legal_suffix), f"[value_{slot}]", expanded_text)
                        break
                else:
                    expanded_text = re.sub(r"(?<=\s|\W)" + re.escape(value), f"[value_{slot}] ", expanded_text)
                    corrected_text = re.sub(r"(?<=\s|\W)" + re.escape(value), f"{value} ", corrected_text)
                    # print(f"corrected_text: [{slot}={value}] {corrected_text}")
            # print(f"{act} occurs once in '{expanded_text}'")
            dataset_info["value_half_occurrence"][act['slot']] += 1
    return expanded_text.strip(), corrected_text.strip()
