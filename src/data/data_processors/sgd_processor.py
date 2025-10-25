from src.data.data_processors.gen_processor import GenProcessor

import os
import re
import copy
import json
import glob
from copy import deepcopy
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from itertools import chain, zip_longest


from src.utils import utils
from src.utils import sgd_ontology


class SGDProcessor(GenProcessor):

    def __init__(self, hparams, save_temp=True, need_processing=True):
        super().__init__(hparams)

        assert self.data_version in [f"v{i}" for i in range(6)] + [f"mix_{i}" for i in range(1, 7)], \
            f"Unknown data version for 'SGD': '{self.data_version}'"
        self.data_source = f'sgd/{self.data_version}'
        self.dataset_path = os.path.join(self.data_root, self.data_source)
        self.suffix = "-debug" if self.debug else ""

        self.has_intent = True
        self.has_sys_act = True
        self.has_concrete_resp = True
        # available options: 'delex_resp', 'concrete_resp', None
        self.resp4eval = 'delex_resp'

        self._load_schema()
        # f"{split}-{dial_id}": dial_meta
        self.did2meta = {}
        self.requestable_slots = {"train": defaultdict(list), "dev": defaultdict(list), "test": defaultdict(list)}
        num_train_raw = self._load_data("train", save_temp=save_temp, need_processing=need_processing)
        num_dev_raw = self._load_data("dev", save_temp=save_temp, need_processing=need_processing)
        num_test_raw = self._load_data("test", save_temp=save_temp, need_processing=need_processing)
        print(f"train: {num_train_raw} -> {len(self.train)}\n"
              f"dev: {num_dev_raw} -> {len(self.dev)}\n"
              f"test: {num_test_raw} -> {len(self.test)}")

        merged_requestable_slots = defaultdict(list)
        for split, service2reqs in self.requestable_slots.items():
            for serv, reqs in service2reqs.items():
                merged_requestable_slots[serv].extend(reqs)
        self.requestable_slots['all'] = merged_requestable_slots

        if self.disable_intent:
            delattr(self, "intent_schema")

        if self.inform_request_schema:
            requested_slots = {serv: set(reqs) for serv, reqs in merged_requestable_slots.items()}
            self.schema = sgd_ontology.get_schema_with_informable_and_requestable_slots(
                self.original_schema, copy.deepcopy(requested_slots), self.disable_intent)
            schema_list = [
                f"{serv} ({len(self.service2intents[serv])}, {self._tokenize_text(json.dumps(self.schema[serv]))})"
                for serv in sorted(self.schema.keys())]
            print(f"Merged Inform and Request Schema [{len(self.schema)}]:", schema_list, end='\n\n')
            with open(os.path.join(self.dataset_path, f'inform_request_schema{self.suffix}.json'), 'w', encoding='utf-8') as fout:
                json.dump({serv: {**content, "requested slots": list(requested_slots.get(serv, set()))}
                           for serv, content in self.schema.items()}, fout, ensure_ascii=False, indent=2)

        self.requestable_slots = {
            split: {serv: OrderedDict(Counter(reqs).most_common()) for serv, reqs in service2reqs.items()}
            for split, service2reqs in self.requestable_slots.items()}
        with open(os.path.join(self.dataset_path, f'requested_slots{self.suffix}.json'), 'w', encoding='utf-8') as fout:
            json.dump(self.requestable_slots, fout, ensure_ascii=False, indent=2)
        return

    def _load_schema(self):
        train_schema = json.load(open(os.path.join(self.dataset_path, 'train/schema.json'), 'r', encoding='utf-8'))
        train_domains = sorted(list(set([s["service_name"].split('_')[0] for s in train_schema])))
        dev_schema = json.load(open(os.path.join(self.dataset_path, 'dev/schema.json'), 'r', encoding='utf-8'))
        dev_domains = sorted(list(set([s["service_name"].split('_')[0] for s in dev_schema])))
        test_schema = json.load(open(os.path.join(self.dataset_path, 'test/schema.json'), 'r', encoding='utf-8'))
        test_domains = sorted(list(set([s["service_name"].split('_')[0] for s in test_schema])))
        print(f"Train  Schema [{len(train_schema)}]:", sorted([schema["service_name"] for schema in train_schema]))
        print(f"Valid  Schema [{len(dev_schema)}]:", sorted([schema["service_name"] for schema in dev_schema]))
        print(f"Test   Schema [{len(test_schema)}]:", sorted([schema["service_name"] for schema in test_schema]))

        self.original_schema = {}
        self.all_services = set()
        self.all_domains = set()
        self.service2intents = {}
        for s in train_schema + dev_schema + test_schema:
            service_name = s["service_name"]
            if service_name in self.original_schema:
                assert self.original_schema[service_name] == s, f"{service_name} schema is not consistent"
                continue
            domain_name = service_name.split('_')[0]
            self.original_schema[service_name] = s
            self.all_services.add(service_name)
            self.all_domains.add(domain_name)
            self.service2intents[service_name] = [intent["name"] for intent in s["intents"]]
        with open(os.path.join(self.dataset_path, 'merged_schema.json'), 'w', encoding='utf-8') as fout:
            json.dump(self.original_schema, fout, ensure_ascii=False, indent=2)
        config_path = os.path.join(self.project_root, 'dataset_config', f"sgd_schema_{self.data_version}.json")
        if not os.path.exists(config_path):
            with open(config_path, 'w', encoding='utf-8') as fout:
                json.dump(self.original_schema, fout, ensure_ascii=False, indent=2)
        self.all_domains = sorted(list(self.all_domains))
        domain_str = ', '.join(self.all_domains)
        print(f"Domains: [{self._tokenize_text(domain_str)}] {domain_str}")
        self.all_services = sorted(list(self.all_services))
        service_str = ', '.join([f"\n\t{serv}: {self.original_schema[serv]['description']}" for serv in self.all_services])
        print(f"Services: [{self._tokenize_text(service_str)}] {service_str}")
        schema_list = [
            f"{serv} ({len(self.service2intents[serv])}, {self._tokenize_text(json.dumps(self.original_schema[serv]))})"
            for serv in sorted(self.original_schema.keys())]
        print(f"Merged Schema [{len(self.original_schema)}]:", schema_list, end='\n\n')
        # TODO: simply schema in order to reduce the number of tokens
        self.schema, self.intent_schema = sgd_ontology.simplify_schema(self.original_schema, self.disable_intent)
        with open(os.path.join(self.dataset_path, 'simplified_schema.json'), 'w', encoding='utf-8') as fout:
            json.dump([self.schema, self.intent_schema], fout, ensure_ascii=False, indent=2)
        schema_list = [
            (f"{serv} ({len(self.service2intents[serv])}, {self._tokenize_text(json.dumps(self.schema[serv]))}, "
             f"{[self._tokenize_text(json.dumps(itc)) for _, itc in self.intent_schema[serv].items()]})")
            for serv in sorted(self.schema.keys())]
        print(f"Merged Simplified Schema [{len(self.schema)}]:", schema_list, end='\n\n')

        print(f"Train  Domains [{len(train_domains)}]:", train_domains)
        print(f"Valid  Domains [{len(dev_domains)}]:", dev_domains)
        print(f"Test   Domains [{len(test_domains)}]:", test_domains)
        print(f"Merged Domains [{len(self.all_domains)}]:", self.all_domains, end='\n\n')

    def get_eval_data(self, split_name='test'):
        eval_data = getattr(self, split_name)
        assert len(eval_data) > 0, "Please load data first."
        if isinstance(self.num_infer_samples, int) and self.num_infer_samples > 0:
            servs2ids = defaultdict(list)
            for idx, dial in enumerate(eval_data):
                servs = '-'.join(sorted(self.did2meta[dial[0]["dial_id"]]["services"]))
                servs2ids[servs].append(idx)
            # print(f"Services Stats for {split_name} [{len(servs2ids)}]: "
            #       f"{json.dumps({k: len(v) for k, v in servs2ids.items()}, indent=2)}")
            num_per_servs = self.num_infer_samples // len(servs2ids)
            new_eval_data = [
                copy.deepcopy(eval_data[did]) for group in list(zip_longest(*servs2ids.values()))[:num_per_servs]
                for did in group if did is not None]
            multi_counter = Counter([len(self.did2meta[dial[0]['dial_id']]['services']) for dial in new_eval_data]).most_common()
            print(f"Multi-domain dialogues [{len(multi_counter)}]: {json.dumps(multi_counter, indent=2)}")
        else:
            new_eval_data = eval_data
        # print(f"[{len(new_eval_data)}]: {['-'.join(self.did2meta[dial[0]['dial_id']]['services']) for dial in new_eval_data]}")
        # exit(0)
        stats = {
            "num_dials": len(new_eval_data),
            "num_turns": sum([len(dial) for dial in new_eval_data]),
        }
        return new_eval_data, stats

    def _read_data(self, data_path):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
        return data

    def _load_data(self, split, save_temp=True, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source}:{split} ...")
            return

        # if not self.do_process and self.do_infer and split == 'train':
        #     print(f"Skip processing data for {self.data_source}:{split} ...")
        #     return

        all_files = glob.glob(os.path.join(self.dataset_path, f'{split}/dialogues_*.json'))
        raw_data = []
        for f in all_files:
            raw_data += self._read_data(f)
        print(f"[{utils.highlight(split, 'yellow')}] "
              f"{utils.highlight(len(all_files), 'yellow')} dialogue files, "
              f"containing {utils.highlight(len(raw_data), 'yellow')} dialogues.")

        processed_data = []
        stats = defaultdict(list)
        actions = defaultdict(list)
        errors, error2ids = [], defaultdict(list)
        for idx, dial in tqdm(enumerate(raw_data), desc=f'Processing {split} data'):
            if self.debug and idx > 1000:
                break
            res = self._process_dialogue(split, dial, stats, actions)
            if isinstance(res, list):
                assert res[0]['dial_id'] not in self.did2meta, f"{res[0]['dial_id']} already in {self.did2meta}"
                self.did2meta[res[0]['dial_id']] = res[0]
                processed_data.append(res[1:])
            else:
                errors.append(res[0])
                error2ids[res[0]].append(res[1])
        print(f"{Counter(stats['num_domains'])}\n{Counter(stats['num_turns'])}")
        print(f"user service: {Counter(stats.pop('user_frames'))}, system service: {Counter(stats.pop('asst_frames'))}")
        print(f"num  intents: {Counter(stats.pop('num_intents'))}")
        print(f"user requests: {Counter(stats.pop('user_reqs'))}")
        stats = pd.DataFrame(stats).describe().to_dict()
        print(json.dumps(stats), end='\n\n')
        usract_counter, sysact_counter = (
            OrderedDict(Counter(actions['user']).most_common()), OrderedDict(Counter(actions['system']).most_common()))
        print(f"User actions    [{len(usract_counter)}]: {json.dumps(usract_counter, indent=2)}")
        print(f"System actions  [{len(sysact_counter)}]: {json.dumps(sysact_counter, indent=2)}")
        # usrint_counter = OrderedDict(Counter(actions['intents']).most_common())
        # print(f"User intents    [{len(usrint_counter)}]: {json.dumps(usrint_counter, indent=2)}")
        crossint_counter = OrderedDict(Counter(actions['cross_intents']).most_common())
        print(f"Cross intents   [{len(crossint_counter)}]: {json.dumps(crossint_counter, indent=2)}")
        # multi_intents = OrderedDict(Counter(actions['multi_intents']).most_common())
        # print(f"Multi intents   [{len(multi_intents)}]: {json.dumps(multi_intents, indent=2)}")
        crossdom_counter = OrderedDict(Counter(actions['cross_domains']).most_common())
        print(f"Cross domains   [{len(crossdom_counter)}]: {json.dumps(crossdom_counter, indent=2)}")
        error_counter = OrderedDict((k, f"{v} - {error2ids[k][:5]}") for k, v in Counter(errors).most_common())
        print(f"Errors (dials)  [{len(error_counter)}]: {json.dumps(error_counter, indent=2)}")
        print()
        if save_temp and self.do_process:
            processed_file = os.path.join(self.dataset_path, f'{split}_detailed{self.suffix}.json')
            with open(processed_file, 'w', encoding='utf-8') as fout:
                json.dump(processed_data, fout, ensure_ascii=False, indent=2)
            print(f"Processed '{split}' data saved to {processed_file}")

        setattr(self, split, processed_data)
        return len(raw_data)

    def _process_dialogue(self, split, dialogue, stats, actions):
        dial_id = f"{split}-{dialogue['dialogue_id']}"
        stats["num_domains"].append(len(dialogue['services']))
        stats["num_turns"].append(len(dialogue['turns']) // 2)
        assert len(dialogue['turns']) % 2 == 0, f"[{dial_id}]: {len(dialogue['turns'])} % 2 != 0"

        dom_counter = Counter([serv.split('_')[0] for serv in dialogue['services']])
        actions["cross_domains"] += [dom for dom, cnt in dom_counter.items() if cnt > 1]

        dial_turns = dialogue.pop('turns')
        dial_meta = {
            "dial_id": dial_id,
            "turn_num": len(dial_turns) // 2,
            "services": dialogue['services'],
            "value_mapping": {},
            "api_calls": {},
            "user_reqs": {}
        }
        processed_dial = [dial_meta]
        accumulated_belief = {}
        service_results, db_results = {}, {}
        for i in range(len(dial_turns) // 2):
            dt_str = f"[{dial_id}] [{i}]"
            user_turn, asst_turn = dial_turns[i * 2], dial_turns[i * 2 + 1]
            assert user_turn['speaker'] == 'USER' and asst_turn['speaker'] == 'SYSTEM', \
                f"{dt_str}: {user_turn['speaker']}, {asst_turn['speaker']}"

            user_frames, asst_frames = user_turn['frames'], asst_turn['frames']
            turn_services = [fr["service"] for fr in user_frames]
            assert len(turn_services) == len(set(turn_services)), f"{dt_str}: {turn_services}"

            stats["user_frames"].append(len(user_frames))
            stats["asst_frames"].append(len(asst_frames))
            service2intent, cross_intents = {}, set()
            usr_reqs = {}
            for frame in user_frames:
                assert frame['service'] in self.schema, f"{dt_str}: {frame['service']} not in schema"
                if frame['state']['requested_slots']:
                    usr_reqs[frame['service']] = frame['state']['requested_slots']
                    self.requestable_slots[split][frame['service']].extend(frame['state']['requested_slots'])
                actions["user"] += list(set([a['act'] for a in frame['actions']]))
                intent = frame['state']['active_intent']
                if intent == 'NONE':
                    continue
                service2intent[frame['service']] = [intent]
                if sum([intent in self.service2intents[serv] for serv in turn_services]) > 1:
                    cross_intents.add(intent)
            stats["user_reqs"].append(len(usr_reqs))

            system_service, system_domain, sys_act_text, slot_spans = "", "", "", []
            db_api_calls = []
            for frame in asst_frames:
                system_service, system_domain = frame['service'], frame['service'].split('_')[0]
                assert system_service in self.schema, f"{dt_str}: {system_service} not in schema"
                actions["system"] += list(set([a['act'] for a in frame['actions']]))
                sys_act_text += ' ' + sgd_ontology.textualize_system_action(system_domain, frame['actions'])
                slot_spans += frame["slots"]
                if "service_call" in frame:
                    db_api_calls.append(f"[{len(frame['service_results'])}] {json.dumps(frame['service_call'])}")
                    dial_meta["api_calls"][i] = (system_service, frame['service_call'], len(frame['service_results']))
            assert system_service in turn_services, \
                f"{dt_str}: system '{system_service}' not in user '{turn_services}'"

            stats["num_intents"].append(len(list(chain(*service2intent.values()))))
            if stats["num_intents"][-1] > 1:
                actions["multi_intents"].append(json.dumps(service2intent))
            actions["intents"] += list(chain(*service2intent.values()))
            actions["cross_intents"] += list(cross_intents)

            turn_data = {}
            turn_data['dial_id'] = dial_id
            turn_data['turn_num'] = i
            # Note: In SGD dataset, system only process one domain-intent pair at each turn
            # turn_intent = domain2intent.get(system_domain, [])
            user_domains = [serv.split('_')[0] for serv in service2intent]
            turn_data['turn_domain'] = user_domains if service2intent else []
            turn_data['domain_mapping'] = {system_domain: system_service}
            if not self.disable_intent:
                # turn_data['turn_intent'] = {system_service: turn_intent} if turn_intent else {}
                turn_data['turn_intent'] = service2intent
            turn_data['user'] = user_turn['utterance']
            turn_data['user_reqs'] = usr_reqs
            dial_meta["user_reqs"][i] = usr_reqs
            # Note: clean belief state
            current_belief = sgd_ontology.accumulate_belief_state(
                user_turn, asst_turn, dial_meta["value_mapping"], accumulated_belief, dt_str)
            turn_data['bspn'] = copy.deepcopy(accumulated_belief)
            # TODO: get db result
            dbres, error_info = self.update_db_result(
                sgd_ontology.normalize_constraint(current_belief, dial_meta["value_mapping"]),
                user_frames, asst_frames[-1], service_results, db_results,
                sys_act_text, dt_str)
            if error_info:
                return (error_info, dt_str)
            turn_data['dbres'] = dbres
            turn_data['api_call'] = db_api_calls
            # Note: get system action
            if not self.disable_sys_act:
                turn_data['aspn'] = sgd_ontology.rephrase_system_action(sys_act_text.strip())
            # Note: get delexicalized response
            turn_data['delex_resp'] = sgd_ontology.delexicalize_utterance(asst_turn['utterance'], slot_spans)
            if self.gen_concrete_resp:
                turn_data['concrete_resp'] = asst_turn['utterance']

            processed_dial.append(turn_data)

        return processed_dial

    def constraint_to_DBpointer(self, constraint_dict, turn_domains, turn_data):
        # TODO: return true db result if constraint is correct else return empty?
        #       or always return true db result during inference?
        gold_dbres = turn_data['dbres']
        if not gold_dbres:
            return {}
        current_service = list(gold_dbres.keys())[0].lower()
        domain = current_service.split('_')[0]
        value_mapping = json.loads(json.dumps(self.did2meta[turn_data['dial_id']]['value_mapping']).lower())
        constraint_dict = json.loads(json.dumps(constraint_dict).lower())

        pred_constraint = constraint_dict.get(current_service, {})
        if not pred_constraint:
            pred_constraint = constraint_dict.get(domain, {})
        pred_constraint = sgd_ontology.normalize_constraint(
            {current_service: pred_constraint}, value_mapping)[current_service]

        gold_constraint = json.loads(json.dumps(turn_data['bspn']).lower())[current_service]
        gold_constraint = sgd_ontology.normalize_constraint(
            {current_service: gold_constraint}, value_mapping)[current_service]

        for slot, value in gold_constraint.items():
            if value != pred_constraint.get(slot, ''):
                break
        else:
            return gold_dbres
        return {}

    def update_db_result(
            self, constraint, user_frames, system_frame, prev_service_results, prev_db_results, sys_act_text, idx):
        # prefixes4query = ["find", "search", "get", "lookup"]
        escape_non_transactional_intents = {'Balance', 'BalanceCheck', 'CheckAccountBalance', 'CheckBalance', 'CheckBankBalance', 'FindAccountBalance', 'GetAccountBalance', 'ProvideBankAccountBalance', 'ViewFundsInAccount'}

        def _format_matched_items(matched_items, total_num):
            if len(matched_items) > 3:
                print(f"{idx}: {len(matched_items)} items matched")
            items_str = '\n        '.join([f"[{idx + 1}] {json.dumps(item)}" for idx, item in
                                           enumerate(matched_items)])
            return f"\n        {items_str}"

        for user_frame in user_frames:
            user_service = user_frame['service']
            user_constraint = constraint.get(user_service, {})
            usracts = [a['act'].lower() for a in user_frame['actions']]
            assert Counter(usracts)['select'] <= 1, f"{idx}: {usracts}"
            select_idx = usracts.index("select") if "select" in usracts else -1
            if select_idx == -1 and user_service not in prev_service_results:
                continue
            assert user_service in prev_service_results, f"{idx}: {user_service} not in {prev_service_results}"
            current_dbres, offer_items, api_status = prev_service_results[user_service]
            if select_idx >= 0:
                assert api_status[1] == '', f"{idx}: {user_service} api_status {api_status}"
                assert 0 < len(offer_items) <= len(current_dbres), \
                    f"{idx}: {user_service} offer_items {len(offer_items)} should be in (0, {len(current_dbres)}]"
                action = user_frame['actions'][select_idx]
                if action['slot']:
                    value = action['canonical_values'][0]
                    offer_items = [item for item in offer_items if item[action['slot']] == value]
                assert len(offer_items) == 1, \
                    f"{idx}: after select, {user_service} offer_items {len(offer_items)} should be 1"
                prev_service_results[user_service] = (offer_items, [], api_status)
                new_res = _format_matched_items(offer_items, len(offer_items))
                prev_db_results[user_service] = {
                    "content": [len(offer_items), new_res],
                    "hash_value": [utils.get_md5_hash(new_res), 0]
                }

            if api_status[1] and (api_status[0] != user_constraint or "negate" in usracts):
                assert select_idx == -1, f"{idx}: {user_service} api_status {api_status}"
                assert "request" not in usracts, f"{idx}: {user_service} api_status {api_status}"
                prev_db_results[user_service] = {}
                prev_service_results[user_service] = ([], [], (user_constraint, ""))

        system_service = system_frame['service']
        focus_constraint = constraint.get(system_service, {})
        current_dbres, offer_items, api_status = prev_service_results.get(system_service, ([], [], ("", "")))
        prev_content = prev_db_results.get(system_service, {})
        content = []
        if "service_call" in system_frame:
            assert "service_results" in system_frame, f"{idx}: {system_frame}"
            method = system_frame['service_call']['method']
            method_transactional = [
                intent["is_transactional"] for intent in self.original_schema[system_service]["intents"]
                if intent["name"] == method][0]
            service_results = system_frame['service_results']
            if "[notify_failure]" in sys_act_text:
                api_status = (focus_constraint, "FAIL")
                # if any([method.lower().startswith(prefix) for prefix in prefixes4query]):
                if not method_transactional and method not in escape_non_transactional_intents:
                    assert len(service_results) == 0, f"{idx}: notify_failure for {method} -> {service_results}"
                    content = [0, '']
                else:
                    assert method in escape_non_transactional_intents or len(offer_items) == 0, \
                        f"{idx}: when notify_failure for {method}, offer_items {len(offer_items)} should be empty"
                    assert len(service_results) <= 1, f"{idx}: notify_failure for {method} -> {service_results}"
                    content = [f"{method} failed", f"\n        {json.dumps(service_results[0])}" if len(service_results) else '']
            elif "[notify_success]" in sys_act_text:
                api_status = (focus_constraint, "SUCCESS")
                assert len(offer_items) == 0, \
                    f"{idx}: when notify_success for {method}, offer_items {len(offer_items)} should be empty"
                assert len(service_results) == 1, f"{idx}: notify_success for {method} -> {service_results}"
                content = [f"{method} succeeded", f"\n        {json.dumps(service_results[0])}"]
            else:
                api_status = (focus_constraint, "")
                assert len(service_results) >= 1, f"{idx}: {method} -> {service_results}"
                assert "[offer]" in sys_act_text, f"{idx}: no offer for {method} -> {sys_act_text}"
            current_dbres = service_results
        else:
            assert "service_results" not in system_frame, f"{idx}: {system_frame}"
            assert "[notify_failure]" not in sys_act_text and "[notify_success]" not in sys_act_text, \
                f"{idx}: {sys_act_text}"

        if "[offer]" in sys_act_text and "[notify_failure]" not in sys_act_text:
            # assert not dbres_changed, \
            #     f"{idx}: current db results {current_dbres} are inconsistent with constraint '{focus_constraint}'"
            assert len(current_dbres) >= 1, f"{idx}: offer non-existent candidates according to {current_dbres}"
            # find relevant results
            related_results = []
            slot2values = dict((act['slot'], act['canonical_values']) for act in system_frame['actions']
                               if act['act'].lower() == 'offer')
            slot_value_num = [len(values) for slot, values in slot2values.items()]
            assert len(set(slot_value_num)) == 1, \
                f"{idx}: inconsistent slot value number in {slot2values}"
            for i in range(slot_value_num[0]):
                cur_offer = {slot: values[i] for slot, values in slot2values.items()}
                for candidate in current_dbres:
                    if sgd_ontology.is_constraint_consistent_with_db_item(candidate, cur_offer):
                        related_results.append(candidate)
                        break
                else:
                    return {}, "offer inconsistent candidates with db results"
                    # print(f"{idx}: no candidate found for {cur_offer} in {current_dbres}")
            content = [len(current_dbres), _format_matched_items(related_results, len(current_dbres))]
            offer_items = related_results

        prev_service_results[system_service] = (current_dbres, offer_items, api_status)
        if content:
            prev_db_results[system_service] = {
                "content": content,
                "hash_value": [utils.get_md5_hash(content[1]), 0]
            }
        db_result = copy.deepcopy(prev_db_results.get(system_service, {}))
        return {system_service: db_result} if db_result else {}, None

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  'ispn', 'real_ispn_gen', 'ispn_gen',
                  'bspn', 'real_bspn_gen', 'bspn_gen', 'aspn', 'real_aspn_gen', 'aspn_gen',
                  'delex_resp', 'delex_resp_gen', 'concrete_resp', 'concrete_resp_gen', 'resp', 'resp_gen',
                  'dbres', 'db_return']
        for turn_results in dialogue_results:
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    entry[key] = value

                results.append(entry)

        return results, fields
