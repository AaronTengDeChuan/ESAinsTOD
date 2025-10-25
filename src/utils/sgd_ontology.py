import re
import json
from src.utils import utils

def simplify_schema(schema, disable_intent):
    simplified_schema = {}
    intent_schema = {}
    for service_name, service in schema.items():
        simplified_schema[service_name] = {
            "slots": {},
        }
        if not disable_intent:
            simplified_schema[service_name]["intents"] = [intent["name"] for intent in service["intents"]]
        intent_schema[service_name] = {}
        for slot in service["slots"]:
            simplified_schema[service_name]["slots"][slot["name"]] = {}
            if slot["possible_values"]:
                simplified_schema[service_name]["slots"][slot["name"]]["possible_values"] = slot["possible_values"]
        for intent in service["intents"]:
            intent_schema[service_name][intent["name"]] = {
                "required_slots": intent["required_slots"],
                "optional_slots": intent["optional_slots"],
                "result_slots": intent["result_slots"],
            }
    return simplified_schema, intent_schema


def get_schema_with_informable_and_requestable_slots(schema, requested_slots, disable_intent):
    plain_schema = {}
    for service_name, service in schema.items():
        all_slots = {}
        for slot in service['slots']:
            all_slots[slot["name"]] = {}
            if slot["possible_values"]:
                all_slots[slot["name"]]["possible_values"] = slot["possible_values"]
        informable_slots = set()
        for intent in service["intents"]:
            informable_slots.update(intent["required_slots"])
            informable_slots.update(intent["optional_slots"])
        requestable_slots = requested_slots.get(service_name, set())
        requestable_slots.update(set(all_slots) - informable_slots - requestable_slots)
        plain_schema[service_name] = {
            "informable slots": {sn: all_slots[sn] for sn in informable_slots},
            "requestable slots": list(requestable_slots),
        }
        if not disable_intent:
            plain_schema[service_name]["intents"] = [intent["name"] for intent in service["intents"]]
    return plain_schema


all_system_actions = ["OFFER", "REQUEST", "INFORM", "CONFIRM", "NOTIFY_SUCCESS", "GOODBYE", "INFORM_COUNT", "REQ_MORE", "OFFER_INTENT", "NOTIFY_FAILURE"]

ordered_system_actions = ["INFORM", "NOTIFY_SUCCESS", "NOTIFY_FAILURE", "INFORM_COUNT", "OFFER", "REQUEST", "CONFIRM", "REQ_MORE", "GOODBYE", "OFFER_INTENT"]

slot_following_actions = ["OFFER", "REQUEST", "INFORM", "CONFIRM", "OFFER_INTENT"]


def add_value_into_mapping(value_mapping, domain, slot, canonical_value, value):
    # if canonical_value == value:
    #     return
    # if domain not in value_mapping:
    #     value_mapping[domain] = {}
    # if slot not in value_mapping[domain]:
    #     value_mapping[domain][slot] = {}
    # if canonical_value not in value_mapping[domain][slot]:
    #     value_mapping[domain][slot][canonical_value] = []
    # if value not in value_mapping[domain][slot][canonical_value]:
    #     value_mapping[domain][slot][canonical_value].append(value)
    if canonical_value not in value_mapping:
        value_mapping[canonical_value] = []
    if canonical_value == value:
        return
    if value not in value_mapping[canonical_value]:
        value_mapping[canonical_value].append(value)
    else:
        # put the value to the end of the list
        value_mapping[canonical_value].remove(value)
        value_mapping[canonical_value].append(value)


skip_case = ["dontcare", "dont care", "don't care", "do n't care", "do not care", "any", "none", "not mentioned"]


def normalize_constraint(constraint, value_mapping):
    normalized_constraint = {}
    for domain, slots in constraint.items():
        domain = utils.check_domain_and_slot(domain, constraint).strip()
        normalized_constraint[domain] = {}
        assert isinstance(slots, dict), f"slots of '{domain}' should be a dict, but got {constraint}"
        for slot, value in slots.items():
            slot = utils.check_domain_and_slot(slot, constraint).strip()
            value = value.strip()
            if value.lower() in skip_case:
                continue
            normalized_constraint[domain][slot] = value
            can_values = []
            for canon_value, variant_values in value_mapping.items():
                if value in variant_values:
                    can_values.append(canon_value)
                    normalized_constraint[domain][slot] = canon_value
            assert len(can_values) <= 1, \
                f"Multiple canonical values '{can_values}' for '{domain}-{slot}={value}': {value_mapping}"
    return normalized_constraint


def is_constraint_consistent_with_db_item(db_item, constraint, skip_slots=None, exact_match=True):
    skip_slots = skip_slots or []
    def is_time(slot, value):
        match_status = re.fullmatch(r"([0-9]{1,2}:[0-9]{2})", value) is not None
        assert "time" not in slot or match_status, f"Time slot {slot} has invalid value {value}"
        return match_status
    return all([db_item[slot] == value for slot, value in constraint.items()
                if slot not in skip_slots and slot in db_item and (exact_match or not is_time(slot, db_item[slot]))])


def accumulate_belief_state(user_turn, asst_turn, value_mapping, belief_state, idx):
    def update_value_mapping(actions, valid_acts, domain):
        for act in actions:
            if act['act'] in valid_acts and act['canonical_values'] and act['values']:
                assert len(act['canonical_values']) == len(act['values']), \
                    f"{idx}: canonical values and values should have the same length: {act}"
                for i in range(len(act['canonical_values'])):
                    add_value_into_mapping(
                        value_mapping, domain, act['slot'], act['canonical_values'][i], act['values'][i])

    current_state = {}

    for frame in user_turn['frames']:
        # update value mapping
        service, dom = frame['service'], frame['service'].split('_')[0]
        dom = service
        update_value_mapping(frame['actions'], ['INFORM', 'SELECT'], dom)
        state = frame['state']['slot_values']
        assert dom not in current_state, f"{idx}: Duplicate domain {dom} in {user_turn['frames']}"
        current_state[dom] = {}
        for slot, values in state.items():
            if len(values) == 1:
                current_state[dom][slot] = values[0]
            else:
                assert value_mapping, \
                    f"{idx}: Cannot find value mapping for '{service}-{slot}: {values}' in {value_mapping}"
                for canon_value, variant_values in value_mapping.items():
                    if canon_value in values:
                        current_state[dom][slot] = canon_value
                        break
                    elif set(values) & set(variant_values):
                        # find the last value in the variant_values
                        current_state[dom][slot] = sorted(
                            [(variant_values.index(value), value) for value in values if value in variant_values],
                            key=lambda x: x[0], reverse=True)[0][1]
                        break
                else:
                    raise ValueError(f"{idx}: Cannot determine the latest value for '{service}-{slot}' in {values} "
                                     f"based on {value_mapping}")

    for frame in asst_turn['frames']:
        # update value mapping
        service, dom = frame['service'], frame['service'].split('_')[0]
        dom = service
        update_value_mapping(frame['actions'], ['REQUEST', 'OFFER', 'CONFIRM'], dom)

    belief_state.update(current_state)

    return current_state


def textualize_system_action(domain, actions):
    if not actions:
        return ""
    sorted_actions = sorted(actions, key=lambda x: ordered_system_actions.index(x['act']))
    action_strs = [f"[{domain}]"]
    prev_act = None
    for action in sorted_actions:
        cur_act = action['act'].lower()
        if action['act'] in slot_following_actions and prev_act == action['act']:
            if action['act'] == "OFFER_INTENT":
                assert len(action['canonical_values']) == 1, \
                    f"OFFER_INTENT action should have only one canonical value, but got {action['canonical_values']}"
                action_strs.append(action['canonical_values'][0])
            else:
                action_strs.append(action['slot'])
        elif prev_act == action['act']:
            assert action['act'] not in slot_following_actions, f"action {action['act']} should not appear twice"
        else:
            action_strs.append(f"[{cur_act}]")
            if action['act'] in slot_following_actions and action['act'] != "OFFER_INTENT":
                action_strs.append(action['slot'])
            elif action['act'] == "OFFER_INTENT":
                assert len(action['canonical_values']) == 1, \
                    f"OFFER_INTENT action should have only one canonical value, but got {action['canonical_values']}"
                action_strs.append(action['canonical_values'][0])
        prev_act = action['act']
    if set(action_strs[1:]).issubset({'[req_more]', '[goodbye]'}):
        del action_strs[0]
    assert len(action_strs) > 0, f"Empty action: {actions}"
    return " ".join(action_strs)


def rephrase_system_action(sys_act):
    new_sys_act = sys_act.replace("[offer]", "[recommend]")
    return new_sys_act


def delexicalize_utterance(text, slot_spans):
    delex_intervals = []
    prev_end = 0
    for slot_span in sorted(slot_spans, key=lambda x: x['start']):
        assert prev_end <= slot_span['start'], f"{prev_end} > {slot_span['start']}"
        delex_intervals.extend([text[prev_end:slot_span['start']], f"[value_{slot_span['slot']}]"])
        prev_end = slot_span['exclusive_end']
    delex_intervals.append(text[prev_end:])
    return ''.join(delex_intervals)
