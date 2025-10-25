from src.data.data_processors.gen_processor import GenProcessor

import os
import re
import copy
import json
import glob
import random
import pandas as pd
from tqdm import tqdm
from itertools import chain
from collections import defaultdict, Counter, OrderedDict

from src.utils import utils
from src.utils import frames_ontology, bitod_ontology, star_ontology


class FramesProcessor(GenProcessor):
    def __init__(self, hparams, save_temp=True, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'frames'
        self.dataset_path = os.path.join(self.data_root, self.data_source)

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = True
        self.has_concrete_resp = True
        self.resp4eval = 'concrete_resp'

        self._load_data(save_temp=save_temp, need_processing=need_processing)

        return

    def _load_schema(self, requested_slots):
        requested_slots = frames_ontology.get_requested_slots(requested_slots)
        self.schema = {
            self.all_domains[0]: {
                "informable slots": list(map(lambda x: frames_ontology.format_slot_name(frames_ontology.map_slot_name(x)), frames_ontology.informable_slots)),
                # "requestable slots": list(map(lambda x: frames_ontology.format_slot_name(frames_ontology.map_slot_name(x)), requested_slots)),
            }
        }
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")
        with open(os.path.join(self.dataset_path, 'schema.json'), 'w', encoding='utf-8') as fout:
            json.dump(self.schema, fout, ensure_ascii=False, indent=2)

    def _load_data(self, save_temp=True, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        self.all_domains = ['vacation_package']
        data = json.load(open(os.path.join(self.dataset_path, 'frames.json'), 'r', encoding='utf-8'))
        res_list = []
        requested_slots = set()
        for idx, dial in enumerate(data):
            res_dict, req_slots = self._process_dialogue(str(idx), dial)
            if res_dict:
                res_list.append(res_dict)
                requested_slots.update(req_slots)
        self._load_schema(requested_slots)
        random.shuffle(res_list)
        dev_list = res_list[:100]
        train_list = res_list[100:]

        self.train, self.dev, self.test = train_list, dev_list, []

        print(json.dumps(self.dev[0], indent=2))
        train_turns = sum([len(dial) for dial in self.train])
        dev_turns = sum([len(dial) for dial in self.dev])
        print(f"train: {len(self.train)} dials, {train_turns} turns, {train_turns / len(self.train):.2f} average turns per dialog.\n"
              f"dev: {len(self.dev)} dials, {dev_turns} turns, {dev_turns / len(self.dev):.2f} average turns per dialog.")

        if save_temp and self.do_process:
            with open(os.path.join(self.dataset_path, 'train_detailed.json'), 'w', encoding='utf-8') as fout:
                json.dump(self.train, fout, ensure_ascii=False, indent=2)
            with open(os.path.join(self.dataset_path, 'dev_detailed.json'), 'w', encoding='utf-8') as fout:
                json.dump(self.dev, fout, ensure_ascii=False, indent=2)
            with open(os.path.join(self.dataset_path, 'test_detailed.json'), 'w', encoding='utf-8') as fout:
                json.dump(self.test, fout, ensure_ascii=False, indent=2)

    def _process_dialogue(self, dial_id, dial):
        assert len(self.all_domains) == 1

        processed_dial = []
        requested_slots = set()

        # build_session_list
        raw_session_list = dial['turns']
        sess_list = []
        one_turn_list = []
        target_speaker = 'user'
        target_map = {'user': 'wizard',
                      'wizard': 'user'}
        for sess in raw_session_list:
            if sess['author'] == target_speaker:
                target_speaker = target_map[sess['author']]
                one_turn_list.append(sess)
                if len(one_turn_list) == 2:
                    sess_list.append(one_turn_list)
                    one_turn_list = []
            else:
                continue
        if len(sess_list) == 0:
            return None, set()

        # process session
        bs_dict, bs_name_list = {}, []
        for turn_idx, (usr_dict, system_dict) in enumerate(sess_list):
            usr_uttr, usr_bs, usr_bsdx, bs_dict, bs_name_list, system_uttr, action_text, turn_info, request_slots, binary_questions = \
                self.zip_turn(usr_dict, system_dict, bs_dict, bs_name_list, self.all_domains[0])

            turn_data = {
                'dial_id': dial_id,
                'turn_num': turn_idx,
                'turn_domain': self.all_domains,
                'user': usr_uttr,
                'bspn': usr_bs.copy(),
            }
            if not self.disable_sys_act:
                turn_data['aspn'] = action_text
            if self.gen_concrete_resp:
                turn_data['concrete_resp'] = system_uttr
            processed_dial.append(turn_data)

            requested_slots.update(request_slots)
            requested_slots.update(binary_questions)

        return processed_dial, requested_slots

    def zip_turn(self, usr_dict, system_dict, prev_bs_dict, prev_bs_name_list, domain):
        usr_uttr = usr_dict['text']
        system_uttr = system_dict['text']
        turn_info = {}
        res_bs_dict, res_bs_name_list, usr_bs, usr_bsdx, request_slots, binary_questions = frames_ontology.update_user_belief_state(
            prev_bs_dict, prev_bs_name_list, usr_dict, turn_info)
        # print (res_bs_name_list)
        if usr_bs:
            usr_bs = {domain: usr_bs}
        if usr_bsdx:
            usr_bsdx = {domain: usr_bsdx}
        action_text = frames_ontology.extract_wizard_act(system_dict, domain)
        return usr_uttr, usr_bs, usr_bsdx, res_bs_dict, res_bs_name_list, system_uttr, action_text, turn_info, request_slots, binary_questions


class BiToDProcessor(GenProcessor):
    def __init__(self, hparams, save_temp=True, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'bitod'
        self.dataset_path = os.path.join(self.data_root, self.data_source)
        self.processed_path = os.path.join(self.dataset_path, "processed")

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = True
        self.has_concrete_resp = True
        self.resp4eval = 'delex_resp'

        os.makedirs(self.processed_path, exist_ok=True)
        self._load_schema()
        self._load_data(save_temp=save_temp, need_processing=need_processing)

        return

    def _load_schema(self):
        schema, domain2slots = bitod_ontology.read_ontology(self.dataset_path)
        with open(os.path.join(self.processed_path, "schema.json"), 'w', encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        slot_value_format = schema.pop("format_for_slot_value")
        self.all_domains = list(domain2slots.keys())
        self.domain2slots = domain2slots
        self.schema = schema
        self.slot_value_format = slot_value_format

        print(json.dumps(self.schema, indent=2))
        print(f"slot format: {json.dumps(slot_value_format, indent=2)}")

    def _load_data(self, save_temp=True, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        dataset_info = {
            "action": defaultdict(lambda: 0),
            "active_intent": defaultdict(lambda: 0),
            "multi_task": defaultdict(lambda: 0),
            "task2num": defaultdict(lambda: 0),
            "relation": defaultdict(lambda: 0),
            "item_num": defaultdict(set),
            "system_action": defaultdict(lambda: 0),
            "value_multi_occurrence": defaultdict(lambda: 0),
            "value_absent": defaultdict(lambda: 0),
            "value_half_occurrence": defaultdict(lambda: 0),
        }

        self.train, self.dev, self.test = [], [], []

        split2new_split = {"train": "train", "valid": "dev", "test": "test"}
        for split in ["train", "valid", "test"]:
            new_split = split2new_split[split]
            with open(os.path.join(self.dataset_path, f"data/en_{split}.json"), 'r', encoding="utf-8") as f:
                for dial_id, dial in json.load(f).items():
                    if split == "train":
                        self.train.append(self._process_dialogue(dial, new_split, dataset_info))
                    elif split == "valid":
                        self.dev.append(self._process_dialogue(dial, new_split, dataset_info))
                    elif split == "test":
                        self.test.append(self._process_dialogue(dial, new_split, dataset_info))
                    else:
                        raise ValueError(f"Unknown split: {split}")

        dataset_info = utils.messup_dataset_info(dataset_info)
        print(f"Dataset Info: {json.dumps(dataset_info, indent=2)}")

        print(json.dumps(self.test[0], indent=2))
        train_turns = sum([len(dial) for dial in self.train])
        print(f"train: {len(self.train)} dialogs, {train_turns} turns, {train_turns / len(self.train):.2f} average turns per dialog.")
        dev_turns = sum([len(dial) for dial in self.dev])
        print(f"dev: {len(self.dev)} dialogs, {sum([len(dial) for dial in self.dev])} turns, {dev_turns / len(self.dev):.2f} average turns per dialog.")
        test_turns = sum([len(dial) for dial in self.test])
        print(f"test: {len(self.test)} dialogs, {sum([len(dial) for dial in self.test])} turns, {test_turns / len(self.test):.2f} average turns per dialog.")

        if save_temp and self.do_process:
            with open(os.path.join(self.processed_path, "train_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.train, f, indent=2, ensure_ascii=False)
            with open(os.path.join(self.processed_path, "dev_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.dev, f, indent=2, ensure_ascii=False)
            with open(os.path.join(self.processed_path, "test_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.test, f, indent=2, ensure_ascii=False)

    def _process_dialogue(self, dial, split, dataset_info):
        dial_id = f"{dial['Dialogue_id']}"
        events = dial["Events"]
        scenario = dial.pop("Scenario")
        dataset_info["multi_task"][len(scenario["WizardCapabilities"])] += 1
        for task in scenario["WizardCapabilities"]:
            dataset_info["task2num"][task["Task"]] += 1
        user_goal = scenario["User_Goal"]
        last_state = [event for event in events if event["Agent"] == "User"][-1]["state"]
        goal_diff, relations = bitod_ontology.dict_diff(user_goal, last_state)
        for relation in relations:
            dataset_info["relation"][relation] += 1
        # assert not goal_diff, f"[{split}-{dial_id}] {goal_diff}"
        # if goal_diff:
        #     print(f"[{split}-{dial_id}] {goal_diff}, relations: {relations}")

        # start processing
        turn_list = []
        turn_id = -1
        db_results = {}
        for event in events:
            agent = event["Agent"]
            assert agent in ["User", "Wizard", "KnowledgeBase"], f"Unknown agent: {agent} in {split}-{dial_id}"
            actions = event.get("Actions", None)
            action_type = actions if not isinstance(actions, list) else "list"
            dataset_info["action"][f"{agent}-{action_type}"] += 1

            if agent == "KnowledgeBase":
                assert bitod_ontology.API_MAP[event[
                    "Topic"]] == active_intent, f"[{split}-{dial_id}] {bitod_ontology.API_MAP[event['Topic']]} != {active_intent}"
                item = event["Item"]
                item.pop("description", None)
                total_items = int(event["TotalItems"])
                dataset_info["item_num"][active_intent].add(f"{'>3' if total_items > 3 else total_items}-{len(item)}")
                if active_intent in bitod_ontology.transactional_apis:
                    db_results[active_intent] = ["booking succeeded" if total_items > 0 else "booking failed", [item]]
                else:
                    db_results[active_intent] = [total_items, [item] if item else []]
            elif agent == "User":
                turn_id += 1
                active_intent = event["active_intent"]
                assert active_intent in bitod_ontology.API_MAP, f"Unknown intent: {active_intent} in {split}-{dial_id}"
                active_intent = bitod_ontology.API_MAP[active_intent]
                dataset_info["active_intent"][active_intent] += 1
                # variable 'turn' must not be defined before this line
                assert "turn" not in locals(), f"Variable 'turn' is defined before this line"
                # process belief state
                current_state = bitod_ontology.clean_belief_state(event["state"])
                turn = {
                    "dial_id": dial_id,
                    "turn_num": turn_id,
                    "turn_domain": [active_intent],
                    "user": event["Text"],
                    "bspn": current_state,
                    "api_call": {},
                }
            else:
                if actions == "query":
                    assert bitod_ontology.API_MAP[event[
                        "API"]] == active_intent, f"{bitod_ontology.API_MAP[event['API']]} != {active_intent} in {split}-{dial_id}"
                    api_call = {
                        "method": active_intent,
                        "parameters": event["Constraints"],
                    }
                    turn["api_call"][active_intent] = json.dumps(api_call)
                else:
                    turn["api_call"] = list(turn["api_call"].values())
                    # TODO: manage db results
                    turn["dbres"] = self.get_dbres(db_results, turn)
                    # TODO: parse system actions
                    for sys_act in actions:
                        act_str = f"{sys_act['act']}-{min(len(sys_act['slot']), 3)}-{min(len(sys_act['relation']), 3)}-{len(sys_act['value'])}"
                        dataset_info["system_action"][act_str] += 1
                        assert sys_act["slot"] in ["", "available_options", "start_date"] or sys_act["slot"] in self.domain2slots[active_intent], f"[{split}-{dial_id}:{turn_id}] {sys_act} not in {active_intent}: {self.domain2slots[active_intent]}"
                    sys_act_text = bitod_ontology.textualize_system_action(active_intent, actions)
                    if not self.disable_sys_act:
                        turn["aspn"] = bitod_ontology.rephrase_system_action(sys_act_text)
                    # TODO: delexicalize system utterance
                    delex_resp, concrete_resp = bitod_ontology.delexicalize_utterance(event["Text"], actions, dataset_info)
                    turn["delex_resp"] = delex_resp
                    if self.gen_concrete_resp:
                        turn["concrete_resp"] = concrete_resp
                    turn_list.append(copy.deepcopy(turn))
                    del turn

        return turn_list

    def get_dbres(self, db_results, turn):
        dbres = {}
        for domain in turn["turn_domain"]:
            if domain in db_results:
                item_strs = [f"\n        {json.dumps(item)}" for idx, item in enumerate(db_results[domain][1])]
                res_str = "".join(item_strs)
                dbres[domain] = {
                    "content": [db_results[domain][0], res_str],
                    "hash_value": [utils.get_md5_hash(res_str), 0]
                }
        return dbres


class STARProcessor(GenProcessor):
    def __init__(self, hparams, save_temp=True, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'star'
        self.dataset_path = os.path.join(self.data_root, self.data_source)
        self.processed_path = os.path.join(self.dataset_path, "processed")

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = True
        self.has_sys_act = True
        self.has_concrete_resp = True
        self.resp4eval = 'concrete_resp'

        os.makedirs(self.processed_path, exist_ok=True)
        self._load_schema()
        if self.disable_intent:
            delattr(self, "intent_schema")
        self._load_data(save_temp=save_temp, need_processing=need_processing)

        return

    def _load_schema(self):
        schema, intent_schema = star_ontology.read_ontology(self.dataset_path)
        with open(os.path.join(self.processed_path, "schema.json"), 'w', encoding="utf-8") as f:
            json.dump([schema, intent_schema], f, indent=2, ensure_ascii=False)

        self.schema = schema
        self.intent_schema = intent_schema
        self.all_domains = star_ontology.all_domains
        self.all_intents = star_ontology.all_intents

        print(json.dumps(self.schema, indent=2))
        print(json.dumps(self.intent_schema, indent=2))

    def _load_data(self, save_temp=True, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        dataset_info = {
            "CompletionLevel2IDs": defaultdict(list),
            "multi_domain": defaultdict(lambda: 0),
            "domain2num": defaultdict(lambda: 0),
            "multi_task": defaultdict(lambda: 0),
            "task2num": defaultdict(lambda: 0),
        }

        dialogs_dir = os.path.join(self.dataset_path, "dialogues")
        all_dialog_files = glob.glob(os.path.join(dialogs_dir, "*.json"))

        all_dials, errors = [], []
        for file_path in tqdm(all_dialog_files):
            turns, error_type = self._process_dialogue(file_path, dataset_info)
            if turns is None:
                errors.append(error_type)
                continue
            all_dials.append(turns)
        self.train, self.dev, self.test = all_dials, [], []

        error_counter = dict(Counter(errors).most_common())
        print(f"Error Counter: {json.dumps(error_counter, indent=2)}")
        CompletionLevel2IDs = dataset_info.pop("CompletionLevel2IDs")
        for level, ids in copy.deepcopy(CompletionLevel2IDs).items():
            CompletionLevel2IDs.pop(level)
            CompletionLevel2IDs[f"{level} [{len(ids)}]"] = sorted(ids)[:50]
        print(CompletionLevel2IDs)
        dataset_info = utils.messup_dataset_info(dataset_info)
        print(f"Dataset Info: {json.dumps(dataset_info, indent=2)}")

        print(json.dumps(self.train[0], indent=2))
        all_turns = sum([len(dial) for dial in all_dials])
        print(
            f"total: {len(all_dials)} dialogs, {all_turns} turns, {all_turns / len(all_dials):.2f} average turns per dialog.")
        print(f"train: {len(self.train)}\n"
              f"dev: {len(self.dev)}\n"
              f"test: {len(self.test)}")

        if save_temp and self.do_process:
            with open(os.path.join(self.processed_path, "train_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.train, f, indent=2, ensure_ascii=False)
            with open(os.path.join(self.processed_path, "dev_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.dev, f, indent=2, ensure_ascii=False)
            with open(os.path.join(self.processed_path, "test_dials.json"), 'w', encoding="utf-8") as f:
                json.dump(self.test, f, indent=2, ensure_ascii=False)

    def _process_dialogue(self, file_path, dataset_info):
        dial = json.load(open(file_path, 'r', encoding="utf-8"))
        dial_id = f"star-{dial['DialogueID']}"
        # information
        events = dial["Events"]
        scenario = dial.pop("Scenario")
        dataset_info["CompletionLevel2IDs"][dial["CompletionLevel"]].append(f"{dial['DialogueID']} ({len(events)})")
        dataset_info["multi_domain"][len(scenario["Domains"])] += 1
        assert int(scenario["MultiTask"]) + int(len(scenario[
                                                        "Domains"]) > 1) != 1, f"[{dial_id}] MultiTask: {scenario['MultiTask']}, Domains: {scenario['Domains']}"
        domains = []
        for domain in scenario["Domains"]:
            if domain is None:
                # print(f"[Null Domain] '{dial_id}':{json.dumps(scenario['WizardCapabilities'])}.")
                assert len(scenario["Domains"]) == 1 and len(scenario["WizardCapabilities"]) == 1
                task = scenario["WizardCapabilities"][0]["Task"]
                assert task == "trivia"
                scenario["WizardCapabilities"][0]["Domain"] = "trivia"
                domain = "trivia"
            domains.append(domain)
            dataset_info["domain2num"][domain] += 1
        dataset_info["multi_task"][len(scenario["WizardCapabilities"])] += 1
        domain2active_intents = defaultdict(set)
        for task in scenario["WizardCapabilities"]:
            dataset_info["task2num"][task["Task"]] += 1
            domain2active_intents[task["Domain"]].add(task["Task"])

        # error detection
        if dial["CompletionLevel"] in ["EarlyDisconnectDuringDialogue"]:
            # print(f"[Empty Dial] Skip dial '{dial_id}'.")
            return None, dial["CompletionLevel"]

        # start processing
        turn_list = []
        turn_id = -1
        db_results = {}
        prev_turn = None
        for event in events:
            event_action = event["Action"]
            if event["Agent"] == "User":
                if event_action == "complete":
                    break
                assert event_action == "utter", f"[{dial_id}] User Action: {event_action}"
                turn_id += 1
                cur_turn = {
                    "dial_id": dial_id,
                    "turn_num": turn_id,
                    "turn_domain": [],
                    "turn_intent": {},
                    "user": event["Text"],
                    "api_call": {},
                }

            if event["Agent"] == "Wizard":
                assert event_action in ["query", "select_task", "select_primary", "select_secondary", "utter",
                                        "request_suggestions",
                                        "pick_suggestion"], f"[{dial_id}] Wizard Action: {event_action}"
                if event_action in ["utter", "pick_suggestion"]:
                    # assert event_action == "utter" and "ActionLabel" not in event or event_action == "pick_suggestion" and "ActionLabel" in event, f"[{dial_id}] Wizard Action: {event_action}, ActionLabel: {event.get('ActionLabel', '')}"
                    item_keys = [key for key in event.keys() if key.endswith("Item")]
                    assert set(item_keys).issubset(
                        {"PrimaryItem", "SecondaryItem"}), f"[{dial_id}] Item Keys: {item_keys}"
                    temp_res = defaultdict(list)
                    for item_key in item_keys:
                        item = event[item_key]
                        temp_res[item["APIName"]].append(item)
                    for method, items in temp_res.items():
                        domain = method.split("_")[0]
                        db_results[domain][1][method] = items
                    cur_turn["api_call"] = list(cur_turn["api_call"].values())
                    cur_turn["aspn"] = event.get("ActionLabel", "")
                    star_ontology.update_domain_and_intent(
                        cur_turn["aspn"], prev_turn, cur_turn, self.all_domains, self.all_intents, domain2active_intents)
                    cur_turn["bspn"] = None
                    cur_turn["dbres"] = self.get_dbres(db_results, cur_turn)
                    cur_turn["concrete_resp"] = event["Text"]
                    assert "turn_domain" in cur_turn and "turn_intent" in cur_turn, f"[{dial_id}] No domain or intent in {json.dumps(cur_turn, indent=2)}."
                    turn_list.append(copy.deepcopy(cur_turn))
                    prev_turn = copy.deepcopy(cur_turn)
                elif event_action == "query":
                    api_call = {
                        "method": event["APIName"],
                        "parameters": event["Constraints"],
                    }
                    cur_turn["api_call"][event["APIName"]] = json.dumps(api_call)
                    star_ontology.update_domain_and_intent(
                        event["APIName"], prev_turn, cur_turn, self.all_domains, self.all_intents, domain2active_intents)
                elif event_action == "select_task":
                    star_ontology.update_domain_and_intent(
                        event["Task"], prev_turn, cur_turn, self.all_domains, self.all_intents, domain2active_intents)

            if event["Agent"] == "KnowledgeBase":
                assert event_action == "return_item", f"[{dial_id}] KB Action: {event_action}"
                method = event["APIName"]
                domain = method.split("_")[0]
                item = event.pop("Item", None)
                # db_results[domain] = [event["TotalItems"], []]
                if domain not in db_results:
                    db_results[domain] = [event["TotalItems"], {}]
                else:
                    db_results[domain][0] = event["TotalItems"]
                db_results[domain][1][method] = [item] if item is not None else []

        if len(turn_list) < 3:
            # print(f"[Empty Dial] Skip dial '{dial_id}'.")
            return None, f"Less than 3 turns: only {len(turn_list)} turns."
        task_sequence = []
        for turn in turn_list:
            tasks = list(chain(*turn["turn_intent"].values()))
            task_sequence.append(tasks[0] if tasks else None)
        active_task = [(idx, task) for idx, task in enumerate(task_sequence) if task is not None]
        if len(active_task) == 0:
            return None, f"No active domain."
        if active_task and task_sequence[0] is None:
            first_idx, first_task = active_task[0]
            if turn_list[0]["aspn"] == "hello":
                turn_list[0]["turn_domain"] = []
                turn_list[0]["turn_intent"] = {}
            allow_change = True
            for turn in turn_list[:first_idx]:
                cur_domain = turn["turn_domain"]
                if cur_domain and not first_task.startswith(cur_domain[0]):
                    allow_change = False
            if allow_change:
                for idx, turn in enumerate(turn_list[:first_idx]):
                    if idx == 0 and turn["aspn"] == "hello":
                        continue
                    star_ontology.update_domain_and_intent(
                        first_task, None, turn, self.all_domains, self.all_intents, domain2active_intents)

        for turn_data in turn_list:
            if self.disable_intent:
                turn_data.pop("turn_intent", None)
            if self.disable_sys_act:
                turn_data.pop("aspn", None)
            if not self.gen_concrete_resp:
                turn_data.pop("concrete_resp", None)

        return turn_list, dial["CompletionLevel"]

    def get_dbres(self, db_results, cur_turn):
        dbres = {}
        for domain in cur_turn["turn_domain"]:
            if domain in db_results:
                num_items = db_results[domain][0]
                items = list(chain(*db_results[domain][1].values()))
                item_strs = [f"\n        {json.dumps(item)}" for idx, item in enumerate(items)]
                # if num_items > 0:
                #     item_strs = [f"\n        [{idx + 1}] {json.dumps(item)}" for idx, item in enumerate(db_results[domain][1])]
                # elif num_items == -1:
                #     assert len(db_results[domain][1]) == 1, f"[{cur_turn['dial_id']}] {domain} has {num_items} items: {json.dumps(db_results[domain], indent=2)}."
                #     item_strs = [f"\n        {json.dumps(db_results[domain][1][0])}"]
                # else:
                #     assert num_items == 0 and len(db_results[domain][1]) == 0, f"[{cur_turn['dial_id']}] {domain} has {num_items} items: {json.dumps(db_results[domain], indent=2)}."
                #     item_strs = []
                res_str = "".join(item_strs)
                dbres[domain] = {
                    "content": [num_items if num_items >= 0 else "", res_str],
                    "hash_value": [utils.get_md5_hash(res_str), 0]
                }
        return dbres
