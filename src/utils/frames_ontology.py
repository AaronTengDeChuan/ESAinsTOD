token_map_dict = {
    'arr_time_dst':'arrive time destination',
    'arr_time_or':'arrive time origin',
    'budget_ok':'budget ok',
    'count_amenities':'count amenities',
    'count_category':'count category',
    'count_dst_city':'count destination city',
    'count_name':'count name',
    'count_seat':'count seat',
    'dep_time_dst':'departure time destination',
    'dep_time_or':'departure time origin',
    'dst_city':'destination city',
    'dst_city_ok':'destination city ok',
    'end_date':'end date',
    'end_date_ok':'end date ok',
    'gst_rating':'guest rating',
    'impl_anaphora':'impl anaphora',
    'max_duration':'max duration',
    'min_duration':'min duration',
    'n_adults':'number of adults',
    'n_adults_ok':'number of adults ok',
    'n_children':'number of children',
    'or_city':'origin city',
    'ref_anaphora':'ref anaphora',
    'seat_ok':'seat ok',
    'str_date':'start date',
    'str_date_ok':'start date ok',
    'flex': 'dates flexible',
    'name': 'hotel name'
}


informable_slots = ["budget", "or_city", "dst_city", "min_duration", "max_duration", "str_date", "end_date", "n_adults", "n_children", "flex"]


def map_slot_name(slot_name):
    return token_map_dict.get(slot_name, slot_name)


def format_slot_name(slot_name):
    return '_'.join(slot_name.split())


database_without = ['impl_anaphora', 'action', 'intent', ]
amenities = ['breakfast', 'parking', 'wifi', 'gym', 'spa']
vicinities = ['park', 'museum', 'beach', 'shopping', 'market', 'airport', 'university', 'mall', 'cathedral', 'downtown', 'palace', 'theatre']


def get_requested_slots(requested_slots):
    return [slot for slot in requested_slots if
    not slot.endswith("_ok") and not slot.startswith("count") and slot not in database_without + amenities + vicinities]


def update_user_belief_state(prev_bs_dict, prev_bs_name_list, usr_dict, turn_info):
    # e.g. usr_dict = data[1]['turns'][0]
    res_bs_dict, res_bs_name_list = prev_bs_dict.copy(), prev_bs_name_list.copy()
    # print (res_bs_dict)
    assert usr_dict['author'] == 'user'
    user_frames = usr_dict['labels']["frames"]
    active_frame_id = usr_dict['labels']["active_frame"]
    turn_info["num_frames"] = len(user_frames)
    turn_info["active_frame"] = active_frame_id
    for user_frame in user_frames:
        if user_frame["frame_id"] == active_frame_id:
            active_frame = user_frame
            break
    else:
        print(f"Active user frame id '{active_frame_id}' not found.")

    state = active_frame["info"]
    request_slots, binary_questions = set(), set()
    for item in active_frame["requests"]:
        if item["author"] != "user":
            continue
        request_slots.add(item["key"])
    for item in active_frame["binary_questions"]:
        if item["author"] != "user":
            continue
        binary_questions.add(item["key"])

    bspn, bsdx = {}, {}
    for slot, values in state.items():
        # TODO: only informable slots
        if slot not in informable_slots:
            # if slot in ['ref', 'intent']:
            continue
        assert isinstance(values, list) and values, f"Illegal Values: {state}"
        value_list = []
        for item in values:
            if not isinstance(item['val'], (str, bool)):
                print(f"Special Value Type: {slot} - {item['val']}")
                continue
            value_list.append(str(item['val']))
        if len(value_list) == 0:
            continue
        value = value_list[-1]
        if len(set(value_list)) > 1:
            print(f"Multiple Values: {slot} - {value_list}")
        if value == "-1":
            value = "dont care"

        try:
            assert type(slot) == str
        except:
            continue
        slot = map_slot_name(slot)
        slot = format_slot_name(slot)
        bspn[slot] = value
        bsdx[slot] = 1
        if slot in res_bs_name_list:
            pass
        else:
            res_bs_name_list.append(slot)
        res_bs_dict[slot] = value  # update user belief state
    return res_bs_dict, res_bs_name_list, bspn, bsdx, request_slots, binary_questions


def extract_wizard_act(wizard_dict, domain):
    # e.g. wizard_dict = data[0]['turns'][0]
    assert wizard_dict['author'] == 'wizard'
    acts = wizard_dict['labels']['acts']
    action_list = []
    action_type_dict = {}
    action_type_list = []
    for a in acts:
        action_type = '[' + a['name'] + ']'
        if action_type not in action_type_dict:
            action_type_list.append(action_type)
            action_type_dict[action_type] = []
        else:
            pass

        action_value = a['args']
        if len(action_value) == 0:
            pass
        else:
            if action_value[0]['key'] == 'ref':
                pass
            else:
                for item in action_value:
                    slot = map_slot_name(item['key'])
                    if slot in action_type_dict[action_type]:
                        pass
                    else:
                        action_type_dict[action_type].append(slot)
    # print (action_type_dict)
    action_text = f"[{domain}] "
    for a_type in action_type_list:
        one_text = a_type + ' '
        for a in action_type_dict[a_type]:
            one_text += a + ' '
        one_text = one_text.strip().strip(',').strip()
        action_text += one_text + ' '
    action_text = action_text.strip()
    action_text = ' '.join(action_text.split()).strip()
    return action_text
