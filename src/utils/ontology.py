import os
import json
from collections import defaultdict, Counter

all_domains = ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'police', 'hospital']
db_domains = ['restaurant', 'hotel', 'attraction', 'train']

normlize_slot_names = {
    "car type": "car",
    "entrance fee": "price",
    "duration": "time",
    "leaveat": 'leave',
    'arriveby': 'arrive',
    'trainid': 'id'
}

requestable_slots = {
    "taxi": ["car", "phone"],
    "police": ["postcode", "address", "phone"],
    "hospital": ["address", "phone", "postcode"],
    "hotel": ["address", "postcode", "internet", "phone", "parking", "type", "pricerange", "stars", "area",
              "reference"],
    "attraction": ["price", "type", "address", "postcode", "phone", "area", "reference"],
    "train": ["time", "leave", "price", "arrive", "id", "reference"],
    "restaurant": ["phone", "postcode", "address", "pricerange", "food", "area", "reference"]
}
all_reqslot = ["car", "address", "postcode", "phone", "internet", "parking", "type", "pricerange", "food",
               "stars", "area", "reference", "time", "leave", "price", "arrive", "id"]

booking_result_slots = {
    "taxi": ["car", "phone"],
    "police": ["address", "postcode", "phone"],
    "hospital": ["address", "postcode", "phone"],
    "hotel": ["address", "postcode", "phone", "reference"],
    "attraction": ["address", "postcode", "phone", "reference"],
    "train": ["id", "reference"],
    "restaurant": ["address", "postcode", "phone", "reference"]
}

categorical_slots = {
    "taxi": [],
    "police": [],
    "hospital": [],
    "hotel": ["parking", "internet", "type", "pricerange", "area", "stars", "stay", "day", "people"],
    "attraction": ["area"],
    "train": ["day", "people"],
    "restaurant": ["pricerange", "area", "day", "people"]
}

informable_slots = {
    "taxi": ["leave", "destination", "departure", "arrive"],
    "police": [],
    "hospital": ["department"],
    "hotel": ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name"],
    "attraction": ["area", "type", "name"],
    "train": ["destination", "day", "arrive", "departure", "people", "leave"],
    "restaurant": ["food", "pricerange", "area", "name", "time", "day", "people"]
}
all_infslot = ["type", "parking", "pricerange", "internet", "stay", "day", "people", "area", "stars", "name",
               "leave", "destination", "departure", "arrive", "department", "food", "time"]

all_slots = all_reqslot + ["stay", "day", "people", "name", "destination", "departure", "department"]
get_slot = {}
for s in all_slots:
    get_slot[s] = 1

# mapping slots in dialogue act to original goal slot names
da_abbr_to_slot_name = {
    'addr': "address",
    'fee': "price",
    'post': "postcode",
    'ref': 'reference',
    'ticket': 'price',
    'depart': "departure",
    'dest': "destination",
}

dialog_acts = {
    'restaurant': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'hotel': ['inform', 'request', 'nooffer', 'recommend', 'select', 'offerbook', 'offerbooked', 'nobook'],
    'attraction': ['inform', 'request', 'nooffer', 'recommend', 'select'],
    'train': ['inform', 'request', 'nooffer', 'offerbook', 'offerbooked', 'select'],
    'taxi': ['inform', 'request'],
    'police': ['inform', 'request'],
    'hospital': ['inform', 'request'],
    # 'booking': ['book', 'inform', 'nobook', 'request'],
    'general': ['bye', 'greet', 'reqmore', 'welcome'],
}
all_acts = []
for acts in dialog_acts.values():
    for act in acts:
        if act not in all_acts:
            all_acts.append(act)

dialog_act_params = {
    'inform': all_slots + ['choice', 'open'],
    'request': all_infslot + ['choice', 'price'],
    'nooffer': all_slots + ['choice'],
    'recommend': all_reqslot + ['choice', 'open'],
    'select': all_slots + ['choice'],
    # 'book': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'nobook': ['time', 'people', 'stay', 'reference', 'day', 'name', 'choice'],
    'offerbook': all_slots + ['choice'],
    'offerbooked': all_slots + ['choice'],
    'reqmore': [],
    'welcome': [],
    'bye': [],
    'greet': [],
}

dialog_act_all_slots = all_slots + ['choice', 'open']

# special slot tokens in belief span
# no need of this, just covert slot to [slot] e.g. pricerange -> [pricerange]
slot_name_to_slot_token = {}

# special slot tokens in responses
# not use at the momoent
slot_name_to_value_token = {
    # 'entrance fee': '[value_price]',
    # 'pricerange': '[value_price]',
    # 'arriveby': '[value_time]',
    # 'leaveat': '[value_time]',
    # 'departure': '[value_place]',
    # 'destination': '[value_place]',
    # 'stay': 'count',
    # 'people': 'count'
}


# eos tokens definition
eos_tokens = {
    'user': '<eos_u>', 'user_delex': '<eos_u>',
    'resp': '<eos_r>', 'resp_gen': '<eos_r>', 'pv_resp': '<eos_r>',
    'bspn': '<eos_b>', 'bspn_gen': '<eos_b>', 'pv_bspn': '<eos_b>',
    'bsdx': '<eos_b>', 'bsdx_gen': '<eos_b>', 'pv_bsdx': '<eos_b>',
    'qspn': '<eos_q>', 'qspn_gen': '<eos_q>', 'pv_qspn': '<eos_q>',
    'aspn': '<eos_a>', 'aspn_gen': '<eos_a>', 'pv_aspn': '<eos_a>',
    'dspn': '<eos_d>', 'dspn_gen': '<eos_d>', 'pv_dspn': '<eos_d>'}

# sos tokens definition
sos_tokens = {
    'user': '<sos_u>', 'user_delex': '<sos_u>',
    'resp': '<sos_r>', 'resp_gen': '<sos_r>', 'pv_resp': '<sos_r>',
    'bspn': '<sos_b>', 'bspn_gen': '<sos_b>', 'pv_bspn': '<sos_b>',
    'bsdx': '<sos_b>', 'bsdx_gen': '<sos_b>', 'pv_bsdx': '<sos_b>',
    'qspn': '<sos_q>', 'qspn_gen': '<sos_q>', 'pv_qspn': '<sos_q>',
    'aspn': '<sos_a>', 'aspn_gen': '<sos_a>', 'pv_aspn': '<sos_a>',
    'dspn': '<sos_d>', 'dspn_gen': '<sos_d>', 'pv_dspn': '<sos_d>'}

# db tokens definition
db_tokens = ['<sos_db>', '<eos_db>',
             '[book_nores]', '[book_fail]', '[book_success]',
             '[db_nores]', '[db_0]', '[db_1]', '[db_2]', '[db_3]']


# understand tokens definition
def get_understand_tokens(prompt_num_for_understand):
    understand_tokens = []
    for i in range(prompt_num_for_understand):
        understand_tokens.append(f'<understand_{i}>')
    return understand_tokens


# policy tokens definition
def get_policy_tokens(prompt_num_for_policy):
    policy_tokens = []
    for i in range(prompt_num_for_policy):
        policy_tokens.append(f'<policy_{i}>')
    return policy_tokens


# all special tokens definition
def get_special_tokens(other_tokens):
    special_tokens = ['<go_r>', '<go_b>', '<go_a>', '<go_d>',
                      '<eos_u>', '<eos_r>', '<eos_b>', '<eos_a>', '<eos_d>', '<eos_q>',
                      '<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<sos_d>', '<sos_q>'] \
                     + db_tokens \
                     + other_tokens
    return special_tokens

def save_schema_file(slot2values, save_path):
    # count slot value
    merged_slot2values = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0])))
    split2idx = {"train": 0, "dev": 1, "test": 2}
    for i, (split, dic) in enumerate(slot2values.items()):
        idx = split2idx[split]
        for dom in dic:
            for slot in dic[dom]:
                counter = Counter(dic[dom][slot])
                merged_slot2values[dom][slot]["NUM_UNIQUE"][idx] = len(counter)
                for value, count in counter.most_common():
                    merged_slot2values[dom][slot][value][idx] = count
                    merged_slot2values[dom][slot][value][3] += count
                merged_slot2values[dom][slot]["NUM_UNIQUE"][3] = len(merged_slot2values[dom][slot]) - 1

    schema = {}
    for dom in informable_slots:
        schema[dom] = {
            "informable slots": {},
            "requestable slots": requestable_slots[dom],
        }
        dom_slot2values = merged_slot2values.get(dom, {})
        for slot in informable_slots[dom]:
            schema[dom]["informable slots"][slot] = {}
            if slot in categorical_slots[dom]:
                value2nums = dom_slot2values.get(slot, {})
                valid_values = set(value2nums) - {"NUM_UNIQUE", "none", "dontcare", "do n't care", "dont care"}
                sorted_values = []
                for value, _ in sorted([(v, value2nums[v][0]) for v in valid_values], key=lambda x: x[1], reverse=True):
                    if '|' in value or '<' in value or '>' in value:
                        break
                    sorted_values.append(value)

                schema[dom]["informable slots"][slot]["possible_values"] = sorted(sorted_values)

    with open(os.path.join(save_path, f'schema.json'), 'w', encoding='utf-8') as fout:
        json.dump(schema, fout, ensure_ascii=False, indent=2)

    # convert to string
    for dom in merged_slot2values:
        for slot in merged_slot2values[dom]:
            for value, nums in merged_slot2values[dom][slot].items():
                if value == "NUM_UNIQUE":
                    merged_slot2values[dom][slot][value] = "{0} | {1} | {2} = {3}".format(*nums)
                else:
                    merged_slot2values[dom][slot][value] = "{0} + {1} + {2} = {3}".format(*nums)
    with open(os.path.join(save_path, f'slot2values.json'), 'w',
              encoding='utf-8') as fout:
        json.dump(merged_slot2values, fout, ensure_ascii=False, indent=2)