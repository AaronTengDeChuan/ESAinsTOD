from src.data.data_processors.gen_processor import GenProcessor

import os
import re
import copy
import json
import glob
import random
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter, OrderedDict
from itertools import chain

import spacy
import Levenshtein as Lev
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize as nltk_word_tokenize

from src.utils import utils
from src.args import str2bool
from src.utils.db_ops import MultiWozDB
from src.utils import ontology, frames_ontology
from src.utils.ontologies import CamRest676Ontology, KvretOntology

from raw_data.restore_text_pattern import normalize_and_restore


class MultiWOZProcessor(GenProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.nlp = spacy.load('en_core_web_sm')
        self.db_path = os.path.join(self.data_root, 'multiwoz_db')
        db_suffix = "_processed"
        self.db = MultiWozDB({
            'attraction': os.path.join(self.db_path, f'attraction_db{db_suffix}.json'),
            'hospital': os.path.join(self.db_path, f'hospital_db{db_suffix}.json'),
            'hotel': os.path.join(self.db_path, f'hotel_db{db_suffix}.json'),
            'police': os.path.join(self.db_path, f'police_db{db_suffix}.json'),
            'restaurant': os.path.join(self.db_path, f'restaurant_db{db_suffix}.json'),
            'taxi': os.path.join(self.db_path, f'taxi_db{db_suffix}.json'),
            'train': os.path.join(self.db_path, f'train_db{db_suffix}.json'),
        })

        assert self.data_version in ["2.0", "2.1"], f"Unknown data version for 'MultiWOZ': '{self.data_version}'"
        self.data_source = f'multiwoz{self.data_version}'

        test_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'multiwoz{self.data_version}/testListFile.json'), 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(
            os.path.join(self.data_root, f'multiwoz{self.data_version}/valListFile.json'), 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1
        if self.few_shot_training != '':
            train_few_file = os.path.join(self.data_root, f'multiwoz{self.data_version}/trainListFile_{self.few_shot_training}.json')
            train_list = [l.strip().lower() for l in open(train_few_file, 'r').readlines()]
            self.train_files = {fn.replace('.json', ''): 1 for fn in train_list}

        constraints = {"name": "dontcare", "area": "centre", "food": "dontcare", "pricerange": "cheap"}
        res = self.db.queryJsons("restaurant", constraints, return_name=True)
        print(f"restaurant ({constraints}) -> num_result ({len(res)}), num_unique ({len(set(res))})")
        print(f"dev_files: {len(self.dev_files)}")
        print(f"test_files: {len(self.test_files)}")

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = True
        self.has_concrete_resp = True
        self.has_raw_utterance = True
        # available options: 'delex_resp', 'concrete_resp', None
        self.resp4eval = 'delex_resp'

        self.slot2values = {"train": defaultdict(lambda: defaultdict(list)), "dev": defaultdict(lambda: defaultdict(list)), "test": defaultdict(lambda: defaultdict(list))}
        self._load_data(save_temp=True, need_processing=need_processing)
        ontology.save_schema_file(self.slot2values, os.path.join(self.data_root, f'{self.data_source}'))

        if self.zero_shot_enhancement:
            self.extra_rules4sys_act = "(1) When users request contact information, the assistant is required to supply both the [value_postcode] and [value_phone] ; (2) Within the domains of hotel, attraction, train, and restaurant, the assistant, upon successfully aiding users in making reservations, is required to provide a [value_reference] , alongside any information explicitly requested by the user ; (3) Furthermore, following a successful taxi booking, the assistant must provide the corresponding [value_car] and [value_phone] ."

        return

    def _load_data(self, save_temp=True, need_processing=True):
        def _get_domains_in_dial(dial):
            domains_in_dial = set()
            for domain in ontology.all_domains:
                if dial['goal'].get(domain):
                    domains_in_dial.add(domain)
            return domains_in_dial

        self._load_schema()
        exp_domains = getattr(self, 'exp_domains', None)
        if exp_domains and "all" not in exp_domains:
            if "except" in exp_domains:
                self.all_domains = [dom for dom in self.all_domains if dom not in exp_domains]
        else:
            exp_domains = None

        # Note: make sure the content of data_for_space.json is lower cased
        # self.data = json.loads(open(os.path.join(
        #     self.data_root, f'multiwoz{self.data_version}/data_for_space.json'),
        #     'r', encoding='utf-8').read().lower())
        self.data = json.loads(open(os.path.join(
            self.data_root, f'multiwoz{self.data_version}/data_for_space.json'),
            'r', encoding='utf-8').read())

        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        self.train, self.dev, self.test = [], [], []
        for idx, (fn, dial) in enumerate(tqdm(self.data.items())):
            if exp_domains:
                domains_in_dial = _get_domains_in_dial(dial)
                overlap = any(dom in exp_domains for dom in domains_in_dial)
                if "except" in exp_domains:
                    if overlap:
                        continue
                elif not overlap:
                    continue

            if self.debug and idx > 500:
                break
            if '.json' in fn:
                fn = fn.replace('.json', '')
            if self.dev_files.get(fn):
                self.dev.append(self._get_detailed_data('dev', fn, dial))
            elif self.test_files.get(fn):
                self.test.append(self._get_detailed_data('test', fn, dial))
            else:
                if self.few_shot_training != '' and not self.train_files.get(fn):
                    continue
                self.train.append(self._get_detailed_data('train', fn, dial))

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.train)}\ndev: {len(self.dev)}\ntest: {len(self.test)}")

    def _load_schema(self):
        self.schema = json.load(open("./dataset_config/multiwoz_schema.json", 'r', encoding='utf-8'))

        intent_schema = json.load(open("./dataset_config/multiwoz_intent_schema.json", 'r', encoding='utf-8'))
        if self.pseudo_intent:
            self.intent_schema = {}
            for domain in intent_schema:
                self.schema[domain]['intents'] = list(intent_schema[domain].keys())
                self.intent_schema[domain] = {}
                for intent_name, intent_dict in intent_schema[domain].items():
                    self.intent_schema[domain][intent_name] = {
                        "required_slots": list(self.schema[domain]["informable slots"].keys()),
                        "optional_slots": []
                    }
                    self.intent_schema[domain][intent_name].update(intent_dict)
            del self.intent_schema

        self.all_domains = ontology.all_domains
        assert self.schema.keys() == set(self.all_domains), f"{self.schema.keys()}, {set(self.all_domains)}"
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")

        self.fuzzy_domain_mapping = {
            "attraction": ["travel", "pool", "theatre", "church", "university", "music", "museum", "park", "movie",
                           "event", "club", "sport", "cinema", "gallery", "art", "entertainment", "school",
                           "architect", "place", "college"],
            "taxi": ["ride"],
            "train": ["bus"]
        }
        for legal_domain in self.all_domains:
            if legal_domain not in self.fuzzy_domain_mapping:
                self.fuzzy_domain_mapping[legal_domain] = []
            self.fuzzy_domain_mapping[legal_domain].append(legal_domain)

    def _get_detailed_data(self, split, fn, dial):
        detailed_data = []
        for idx, t in enumerate(dial['log']):
            turn_data = {}
            turn_data['dial_id'] = fn
            turn_data['turn_num'] = t['turn_num']
            turn_data['turn_domain'] = []
            for dom in t['turn_domain'].split():
                if dom.startswith('['):
                    dom = dom[1:-1]
                turn_data['turn_domain'].append(dom)
            if self.pseudo_intent:
                turn_data['turn_intent'] = {
                    domain: self.schema[domain]['intents'][:1]
                        for domain in turn_data['turn_domain'] if domain in self.schema
                }
            turn_data['pointer'] = [int(i) for i in t['pointer'].split(',')]

            turn_data['user'] = t['original_user'] if self.use_raw_utterance else t['user']
            # TODO: convert "[value_{slot}]" to "[entity{idx}.{slot}]"
            # TODO: complement slot value for abstract attribute, e.g. "[value_{slot}]={value}" or "[value_{slot}](={value})"
            turn_data['delex_resp'] = t['restored_resp'] if self.use_raw_utterance else t['resp']
            nodelx_resp = t['nodelx_resp'] if 'nodelx_resp' in t else t['sys']
            if self.gen_concrete_resp:
                turn_data['concrete_resp'] = t['original_nodelx_resp'] if self.use_raw_utterance else nodelx_resp
            turn_data["resp4db_summary"] = nodelx_resp
            turn_data["raw_resp"] = t['original_nodelx_resp']

            turn_data['bspn'] = self.bspan_to_constraint_dict(t['constraint'])
            turn_data['bsdx'] = self.bspan_to_constraint_dict(t['cons_delex'], bspn_mode='bsdx')
            turn_data['sys_act'] = t['sys_act']
            db_result = self.constraint_to_DBpointer(turn_data['bspn'], turn_data['turn_domain'], turn_data)

            turn_data['dbres'] = db_result
            if not self.disable_sys_act:
                turn_data['aspn'] = t['sys_act'].strip()
            detailed_data.append(turn_data)
        [self.slot2values[split][dom][slot].append(value) for dom, cons in detailed_data[-1]['bspn'].items()
         for slot, value in cons.items()]
        return detailed_data

    def constraint_to_DBpointer(self, constraint_dict, turn_domains, turn_data):
        turn_pointer = turn_data['pointer']
        # matnums = self.db.get_match_num(constraint_dict)
        # res = {dom: [matnums[dom]] for dom in turn_domains}
        constraint_dict = {k: v for k, v in constraint_dict.items() if isinstance(v, dict)}
        constraint_dict = json.loads(json.dumps(constraint_dict).lower())
        dbres = self.db.get_match_num(constraint_dict, return_entry=True, return_raw=self.enable_capital)
        # dbres = self.db.get_match_num(constraint_dict, return_entry=True, return_raw=False)
        res = {}

        if self.zero_shot_enhancement:
            extracted_slot_values, not_founds = utils.extract_values_in_concrete_resp(turn_data['resp4db_summary'], turn_data['delex_resp'])
            # if not_founds:
            #     print(f"not_founds: {not_founds}\n\textracted: {extracted_slot_values}\n\t{turn_data}")

        extra_info = {}
        if turn_pointer[-2:] == [0, 1]:
            book_pointer = 'Booking Result: success'
            if self.zero_shot_enhancement:
                # 8 chars: a-z 0-9
                reference_number = extracted_slot_values.get("reference", "").strip()
                if len(reference_number) != 8:
                    reference_number = "".join([random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(8)])
                extra_info.update({
                    "reference": reference_number
                })
        elif turn_pointer[-2:] == [1, 0]:
            book_pointer = 'Booking Result: fail'
        else:
            assert turn_pointer[-2:] == [0, 0]
            # book_pointer = 'Booking Result: no result'
            book_pointer = ''

        taxi_booked = False
        if (("[taxi] [inform]" in turn_data['sys_act'] or "[taxi] [offerbooked]" in turn_data['sys_act'])
                and "taxi" in turn_domains):
            taxi_booked = True
            if self.zero_shot_enhancement:
                random_car = f"{random.choice(self.db.dbs['taxi'][0]['taxi_colors'])} {random.choice(self.db.dbs['taxi'][0]['taxi_types'])}"
                random_phone = "".join([str(random.randint(0, 9)) for _ in range(10)])
                dbres["taxi"] = [{
                    **{k: v for k, v in turn_data['bspn'].get("taxi", {}).items() if k in ontology.informable_slots['taxi']},
                    "car": extracted_slot_values.get("car", random_car),
                    "phone": extracted_slot_values.get("phone", random_phone)
                }]

        for dom, match_results in dbres.items():
            res[dom] = {
                "content": [len(match_results)],
                "hash_value": [len(match_results), 0]
            }
            summary, md5_hash = self.summarize_db_result(
                match_results, set(self.schema[dom]['informable slots']) - {'name'},
                delex_resp=turn_data.get('delex_resp'),
                concrete_resp=turn_data.get('resp4db_summary'), max_display=2, extra_info=extra_info)
            res[dom]["content"].append(summary)
            res[dom]["hash_value"] = md5_hash
        for dom in turn_domains:
            if dom not in res:
                res[dom] = {
                    "content": ['', ''],
                    "hash_value": ['', 0]
                }

        if turn_domains[0] in res:
            if book_pointer:
                res[turn_domains[0]]["content"][0] = book_pointer
        if taxi_booked:
            res["taxi"]["content"][0] = "Booking Result: success"
        return res

    def bspan_to_constraint_dict(self, bspan, bspn_mode='bspn'):
        """
        ['[hotel]', 'pricerange', 'cheap', 'type', 'hotel'] -> {'hotel': {'pricerange': 'cheap', 'type': 'hotel'}}
        """
        assert isinstance(bspan, str)
        bspan = bspan.split()
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx + 1]
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx + 1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  *(['ispn', 'real_ispn_gen', 'ispn_gen'] if self.pseudo_intent else []),
                  'bspn', 'real_bspn_gen', 'bspn_gen', 'bsdx', 'aspn', 'real_aspn_gen', 'aspn_gen',
                  'delex_resp', 'delex_resp_gen', 'concrete_resp', 'concrete_resp_gen', 'resp', 'resp_gen',
                  'raw_resp',
                  'dbres', 'db_return', 'pointer']
        for turn_results in dialogue_results:
            dial_id = turn_results[0]['dial_id']
            turn_len = len(turn_results)
            entry = {'dial_id': dial_id, 'turn_num': turn_len}
            # for f in fields[2:]:
            #     entry[f] = ''
            results.append(entry)
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn['turn_domain'][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    entry[key] = value

                results.append(entry)

        return results, fields


class KvretProcessor(GenProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.word_tokenize = nltk_word_tokenize
        self.wn = WordNetLemmatizer()

        self.data_source = f'kvret'
        self.dataset_path = os.path.join(self.data_root, 'kvret')
        self.raw_data_path = {
            'train': os.path.join(self.dataset_path, 'kvret_train_public.json'),
            'dev': os.path.join(self.dataset_path, 'kvret_dev_public.json'),
            'test': os.path.join(self.dataset_path, 'kvret_test_public.json')
        }

        self.ontology_path = os.path.join(self.dataset_path, 'kvret_entities.json')
        self.otlg = KvretOntology(self.ontology_path)
        self.requestable_slots = self.otlg.requestable_slots
        self.informable_slots = self.otlg.informable_slots
        self.all_domains = self.otlg.all_domains

        self.data_path = {
            'train': os.path.join(self.dataset_path, 'train_preprocessed.json'),
            'dev': os.path.join(self.dataset_path, 'dev_preprocessed.json'),
            'test': os.path.join(self.dataset_path, 'test_preprocessed.json')
        }

        self.entities = json.loads(open(self.ontology_path).read().lower())
        self.get_value_to_slot_mapping(self.entities)

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = False
        self.has_concrete_resp = True
        self.has_raw_utterance = True
        # available options: 'delex_resp', 'concrete_resp', None
        self.resp4eval = 'delex_resp'

        self._load_data(save_temp=True, need_processing=need_processing)

        # for eval
        self.sos_r_token = "<sos_r>"
        self.eos_r_token = "<eos_r>"

        return

    def _tokenize(self, sent):
        return ' '.join(self.word_tokenize(sent))

    def _lemmatize(self, sent):
        return ' '.join([self.wn.lemmatize(_) for _ in sent.split()])

    def get_value_to_slot_mapping(self, entity_data):
        self.raw2lemma = {}
        self.entity_dict = {}
        self.abbr_dict = {}
        for k in entity_data:
            if type(entity_data[k][0]) is str:
                for entity in entity_data[k]:
                    old_entity = entity
                    entity = self._lemmatize(self._tokenize(entity))
                    self.raw2lemma[old_entity] = [entity, old_entity == entity]
                    self.entity_dict[entity] = k
                    if k in ['event','poi_type']:
                        self.entity_dict[entity.split()[0]] = k
                        self.abbr_dict[entity.split()[0]] = entity
            elif type(entity_data[k][0]) is dict:
                for entity_entry in entity_data[k]:
                    for entity_type, entity in entity_entry.items():
                        old_entity = entity
                        entity_type = 'poi_type' if entity_type == 'type' else entity_type
                        entity = self._lemmatize(self._tokenize(entity))
                        self.raw2lemma[old_entity] = [entity, old_entity == entity]
                        self.entity_dict[entity] = entity_type
                        if entity_type in ['event', 'poi_type']:
                            self.entity_dict[entity.split()[0]] = entity_type
                            self.abbr_dict[entity.split()[0]] = entity

    def _load_data(self, save_temp=True, need_processing=True):
        self._load_schema()
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        self.raw_data = {}
        for d in ['train', 'dev', 'test']:
            raw_content = open(self.raw_data_path[d], 'r', encoding='utf-8').read()
            if not self.use_raw_utterance:
                raw_content = raw_content.lower()
            self.raw_data[d] = json.loads(raw_content)

        self.medium_data = {}
        for d in ['train', 'dev', 'test']:
            self.medium_data[d] = json.loads(open(self.data_path[d], 'r', encoding='utf-8').read().lower())
            self.medium_data[d] = [v for k, v in sorted(self.medium_data[d].items(), key=lambda x: int(x[0]))]
            assert len(self.medium_data[d]) == len(self.raw_data[d]), f"{len(self.medium_data[d])}, {len(self.raw_data[d])}"

        # directly read raw and medium data and process
        self.train, self.dev, self.test = [], [], []
        for raw_dial, medium_dial in tqdm(zip(self.raw_data['train'], self.medium_data['train'])):
            temp = self._get_detailed_data(raw_dial, medium_dial)
            if temp: self.train.append(temp)
        for raw_dial, medium_dial in tqdm(zip(self.raw_data['dev'], self.medium_data['dev'])):
            temp = self._get_detailed_data(raw_dial, medium_dial)
            if temp: self.dev.append(temp)
        for raw_dial, medium_dial in tqdm(zip(self.raw_data['test'], self.medium_data['test'])):
            temp = self._get_detailed_data(raw_dial, medium_dial)
            if temp: self.test.append(temp)

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.raw_data['train'])} -> {len(self.train)}\n"
              f"dev: {len(self.raw_data['dev'])} -> {len(self.dev)}\n"
              f"test: {len(self.raw_data['test'])} -> {len(self.test)}")

        # with open(os.path.join(self.dataset_path, 'train_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.train, fout, ensure_ascii=False, indent=2)
        # with open(os.path.join(self.dataset_path, 'dev_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.dev, fout, ensure_ascii=False, indent=2)
        # with open(os.path.join(self.dataset_path, 'test_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.test, fout, ensure_ascii=False, indent=2)
        # with open(os.path.join(self.dataset_path, 'entity_lemmatization.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.raw2lemma, fout, ensure_ascii=False, indent=2)
        # exit(0)

    def _load_schema(self):
        self.schema = {}
        for domain in self.all_domains:
            self.schema[domain] = {
                "informable slots": self.otlg.informable_slots_dict[domain],
                "requestable slots": self.otlg.requestable_slots_dict[domain],
            }
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")

    def _get_detailed_data(self, raw_dial, medium_dial):
        assert isinstance(raw_dial, dict) and isinstance(medium_dial, list), f"{type(raw_dial)}, {type(medium_dial)}"

        if len(medium_dial) == 0:
            return None

        assert set(raw_dial.keys()) == {'dialogue', 'scenario'}, f"{raw_dial.keys()} != {'dialogue', 'scenario'}"
        unfiltered_raw_dialog, scenario = raw_dial['dialogue'], raw_dial['scenario']
        raw_dialog = []
        prev_role = None
        for i, utt in enumerate(unfiltered_raw_dialog):
            assert utt['turn'] in ['driver', 'assistant'], f"{utt['turn']} not in ['driver', 'assistant']"
            if prev_role != utt['turn']:
                assert utt['turn'] in ['driver', 'assistant'], f"{utt['turn']} not in ['driver', 'assistant']"
                prev_role = utt['turn']
                raw_dialog.append(utt)

        # if len(medium_dial) > 0 and medium_dial[0]['dial_id'] in [214, 415, 687, 824, 1404, 2179, 142, 236]:
        #     print(f"Skip {medium_dial[0]['dial_id']}")
        #     return []

        if len(raw_dialog) <= 1:
            print(f"Skip {scenario['uuid']} with {len(raw_dialog)} utterance.")
            return None

        # assert len(raw_dialog) % 2 == 0, f"{len(raw_dialog)} % 2 != 0\n{medium_dial}"
        assert len(raw_dialog) // 2 == len(medium_dial), f"{len(raw_dialog)} // 2 != {len(medium_dial)}\n{json.dumps(raw_dialog, indent=2)}\n{json.dumps(medium_dial, indent=2)}"
        assert set(scenario.keys()) == {'kb', 'task', "uuid"}, f"{scenario.keys()} != {'kb', 'task', 'uuid'}"
        assert len(medium_dial) > 0, f"{len(medium_dial)} == 0\n{raw_dialog}"

        domain = scenario["task"]["intent"]
        inform_slots = self.schema[domain]["informable slots"]
        raw_constraint = {domain: {}}

        kb_items = scenario['kb']['items']
        column_names = scenario['kb']['column_names']
        kb_stats = defaultdict(list)
        if kb_items:
            for item in kb_items:
                for k, v in item.items():
                    assert k in column_names, f"{k} not in {column_names}"
                    kb_stats[k].append(v)
        kb_stats = {k: json.dumps(Counter(v)) for k, v in kb_stats.items()}

        detailed_data = []
        for idx in range(len(medium_dial)):
            user_turn, asst_turn = raw_dialog[idx * 2:idx * 2 + 2]
            medium_turn = medium_dial[idx]
            assert user_turn['turn'] == 'driver' and asst_turn['turn'] == 'assistant', f"{user_turn['turn']}, {asst_turn['turn']}"

            turn_data = {}
            turn_data['dial_id'] = str(medium_turn['dial_id'])
            turn_data['turn_num'] = medium_turn['turn_num']
            turn_data['turn_domain'] = [domain]

            # turn_data['user'] = medium_turn['user']
            turn_data['user'] = user_turn['data']['utterance'].strip()
            # TODO: convert "[value_{slot}]" to "[entity{idx}.{slot}]"
            # TODO: complement slot value for abstract attribute, e.g. "[value_{slot}]={value}" or "[value_{slot}](={value})"
            concrete_resp = asst_turn['data']['utterance'].strip()
            if self.use_raw_utterance:
                turn_data['delex_resp'] = normalize_and_restore(
                    concrete_resp, medium_turn['response'].strip())
            else:
                turn_data['delex_resp'] = medium_turn['response'].strip()
            if self.gen_concrete_resp:
                turn_data['concrete_resp'] = concrete_resp
            turn_data["resp4db_summary"] = concrete_resp
            turn_data["raw_resp"] = concrete_resp

            raw_bs = {k: v.strip() for k, v in asst_turn['data']['slots'].items() if
                      k in inform_slots and v.strip() != ''}
            raw_constraint[domain].update(raw_bs)
            belief_state = json.loads(medium_turn['constraint'])
            medium_constraint = {domain: {}}
            for k, v in belief_state.items():
                if v.strip() == '':
                    continue
                assert k in self.otlg.informable_slots, f"{k} not in {self.otlg.informable_slots}"
                dom, slot = k.split('-')
                assert domain == dom, f"{domain} != {dom}"
                assert slot not in medium_constraint[domain], f"{slot} occurs twice in {belief_state}"
                medium_constraint[domain][slot] = v.strip()
            # turn_data['raw_bspn'] = json.dumps(copy.deepcopy(raw_constraint))
            # turn_data['medium_bspn'] = json.dumps(medium_constraint)
            final_bspn = copy.deepcopy(raw_constraint)
            final_bspn[domain].update(medium_constraint[domain])
            self.fix_value(domain, final_bspn[domain], kb_items)
            turn_data['bspn'] = final_bspn if final_bspn[domain] else {}
            # turn_data['bspn'] = medium_constraint
            turn_data['kb_items'] = kb_items
            db_result = self.constraint_to_DBpointer(final_bspn, turn_data['turn_domain'], turn_data)
            turn_data['dbres'] = db_result
            # turn_data['kb_items'] = kb_stats
            # turn_data['columns'] = ", ".join(column_names) + f"; informable slots: {', '.join(inform_slots)}"

            # Note: fail to construct system action based on the existing annotations
            if not self.disable_sys_act:
                # force to generate sys act during evaluation
                turn_data['aspn'] = ''

            # db_result = self.constraint_to_DBpointer(turn_data['bspn'], turn_data['turn_domain'], turn_data['pointer'])
            # turn_data['dbres'] = db_result

            detailed_data.append(turn_data)

        return detailed_data

    def fix_value(self, domain, constraint, kb_items):
        match domain:
            case "weather":
                assert kb_items, f"No kb items in weather domain"
                loc = constraint.get("location", None)
                if isinstance(loc, str):
                    loc = loc.lower()
                raw_loc_candidates = [item["location"] for item in kb_items]
                loc_candidates = [e.lower() for e in raw_loc_candidates]
                if loc and loc not in loc_candidates:
                    new_loc = loc.replace("city", "").strip()
                    flag = False
                    if new_loc:
                        for i, candidate in enumerate(loc_candidates):
                            if candidate.startswith(new_loc) or candidate.endswith(new_loc):
                                flag = True
                                print(f"Fix location from '{loc}' to '{candidate}'")
                            elif Lev.distance(candidate, new_loc) <= 2:
                                flag = True
                                print(f"Fix location from '{loc}' to '{candidate}', Levenshtein distance: {Lev.distance(candidate, new_loc)}")
                            if flag:
                                constraint["location"] = raw_loc_candidates[i]
                                break
                    if not flag:
                        print(utils.highlight(f"Fail to fix location '{loc}' from '{loc_candidates}'", 'red'))
                        constraint.pop("location")

    def constraint_to_DBpointer(self, constraint_dict, turn_domains, turn_data):
        # constraint_dict = copy.deepcopy(constraint_dict)
        constraint_dict = json.loads(json.dumps(constraint_dict).lower())
        turn_domain = turn_domains[0]
        kb_items = turn_data['kb_items']
        constraint = constraint_dict.get(turn_domain, {})
        if not kb_items or turn_domain not in turn_data['turn_domain'] or not constraint or not isinstance(constraint, dict):
            return {}
        matnum, matstat = '', ''
        if turn_domain == "weather":
            self.fix_value(turn_domain, constraint, kb_items)
            loc = constraint.get("location", None)
            if loc:
                loc = loc.lower()
                flags = [loc in item["location"].lower() for item in kb_items]
                if not any(flags):
                    matstat = f"Location '{loc}' not found"
                else:
                    matnum = sum(flags)
                    matstat = [kb_items[i] for i, flag in enumerate(flags) if flag]
        elif turn_domain == "navigate":
            poi_type = constraint.get("poi_type", None)
            if poi_type:
                type_flags = [self.otlg.similar_poi_type(poi_type, item["poi_type"].lower()) for item in kb_items]
                name_flags = [poi_type in item["poi"].lower() for item in kb_items]
                if any(type_flags):
                    matnum = sum(type_flags)
                    matstat = [kb_items[i] for i, flag in enumerate(type_flags) if flag]
                elif any(name_flags):
                    types = set([kb_items[i]["poi_type"] for i, flag in enumerate(name_flags) if flag])
                    matstat = [item for item in kb_items if item["poi_type"] in types]
                    matnum = len(matstat)
                    # matstat.insert(0, f"Entity '{poi_type}' not found in type, but found in name")
                    # if len(types) > 1:
                    #     matstat.insert(0, f"Multiple types found for '{poi_type}': {types}")
                else:
                    matstat = f"Poi Type '{poi_type}' not found"
        elif turn_domain == "schedule":
            event = constraint.get("event", None)
            if event:
                flags = [self.otlg.similar_event(event, item['event'].lower()) for item in kb_items]
                if not any(flags):
                    kb_str = '\n        '.join([json.dumps(item) for item in kb_items])
                    matstat = f"Event '{event}' not found"
                    if self.detailed_db_result:
                        matstat += f". All events in the schedule will be listed below.\n        {kb_str}"
                else:
                    matnum = sum(flags)
                    matstat = [kb_items[i] for i, flag in enumerate(flags) if flag]
            # else:
            #     matstat = "Event not specified"

        if isinstance(matstat, list):
            matstat = '\n        '.join([f"[{idx + 1}] {json.dumps(item)}" for idx, item in enumerate(matstat)])
            matstat = f"\n        {matstat}"

        if matnum or matstat:
            if matnum == '' and not self.detailed_db_result:
                matnum = matstat
            return {turn_domain: {"content": [matnum, matstat], "hash_value": [utils.get_md5_hash(matstat), 0]}}
        else:
            return {}

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  'bspn', 'real_bspn_gen', 'bspn_gen',
                  'delex_resp', 'delex_resp_gen', 'concrete_resp', 'concrete_resp_gen', 'resp', 'resp_gen',
                  'raw_resp',
                  'dbres', 'db_return']
        for turn_results in dialogue_results:
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    if key in ['bspn', 'bspn_gen']:
                        # flatten bspn
                        constraint = {}
                        for dom, cons in value.items():
                            if not isinstance(cons, dict):
                                continue
                            for slot, val in cons.items():
                                constraint[f"{dom}-{slot}"] = val
                        value = constraint
                    entry[key] = value

                results.append(entry)

        return results, fields


class CamRestProcessor(GenProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)
        self.word_tokenize = nltk_word_tokenize
        self.wn = WordNetLemmatizer()

        # extra part
        self.data_source = f'camrest'
        self.dataset_path = os.path.join(self.data_root, 'camrest')
        self.raw_data_path = os.path.join(self.dataset_path, 'CamRest676.json')
        self.data_path = os.path.join(self.dataset_path, 'CamRest676_preprocessed_add_request_47.json')

        self.ontology_path = os.path.join(self.dataset_path, 'CamRestOTGY.json')
        self.otlg = CamRest676Ontology(self.ontology_path)
        self.requestable_slots = self.otlg.requestable_slots
        self.informable_slots = self.otlg.informable_slots
        self.all_domains = self.otlg.all_domains

        db_json_path = os.path.join(self.dataset_path, 'CamRestDB.json')
        db_content = open(db_json_path).read()
        if not self.use_raw_utterance:
            db_content = db_content.lower()
        self.db_json = json.loads(db_content)

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = True
        self.has_concrete_resp = True
        self.has_raw_utterance = True
        # available options: 'delex_resp', 'concrete_resp', None
        self.resp4eval = 'delex_resp'

        self.split = (3, 1, 1)
        self._load_data(save_temp=True, need_processing=need_processing)

        # for eval
        self.sos_r_token = "<sos_r>"
        self.eos_r_token = "<eos_r>"

        return

    def _split_data(self, encoded_data, split):
        """
        split data into train/dev/test
        :param encoded_data: list
        :param split: tuple / list
        :return:
        """
        total = sum(split)
        dev_thr = len(encoded_data) * split[0] // total
        test_thr = len(encoded_data) * (split[0] + split[1]) // total
        train, dev, test = {}, {}, {}
        for i, (k, v) in enumerate(encoded_data.items()):
            if i < dev_thr:
                train[k] = v
            elif dev_thr <= i < test_thr:
                dev[k] = v
            else:
                test[k] = v
        # train, dev, test = encoded_data[:dev_thr], encoded_data[dev_thr:test_thr], encoded_data[test_thr:]
        return train, dev, test

    def _load_schema(self):
        self.schema = {}
        for domain in self.all_domains:
            self.schema[domain] = {
                "informable slots": self.otlg.informable_slots_dict[domain],
                "requestable slots": self.otlg.requestable_slots_dict[domain],
            }
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")

    def _load_data(self, save_temp=True, need_processing=True):
        self._load_schema()
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        # Note: make sure the content of CamRest676_preprocessed_add_request_47.json is lower cased
        # data = json.loads(open(self.data_path, 'r', encoding='utf-8').read().lower())
        data = json.loads(open(self.data_path, 'r', encoding='utf-8').read())
        train, dev, test = self._split_data(data, self.split)
        self.data = {'train': train, 'dev': dev, 'test': test}

        self.train, self.dev, self.test = [], [], []
        for fn, dial in tqdm(self.data['train'].items()):
            self.train.append(self._get_encoded_data(fn, dial))
        for fn, dial in tqdm(self.data['dev'].items()):
            self.dev.append(self._get_encoded_data(fn, dial))
        for fn, dial in tqdm(self.data['test'].items()):
            self.test.append(self._get_encoded_data(fn, dial))

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.data['train'])} -> {len(self.train)}\n"
              f"dev: {len(self.data['dev'])} -> {len(self.dev)}\n"
              f"test: {len(self.data['test'])} -> {len(self.test)}")

        # with open(os.path.join(self.dataset_path, 'train_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.train, fout, ensure_ascii=False, indent=2)
        # with open(os.path.join(self.dataset_path, 'dev_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.dev, fout, ensure_ascii=False, indent=2)
        # with open(os.path.join(self.dataset_path, 'test_detailed.json'), 'w', encoding='utf-8') as fout:
        #     json.dump(self.test, fout, ensure_ascii=False, indent=2)
        # exit(0)

    def _get_encoded_data(self, fn, dial):
        assert "log" in dial, f"no log in {dial}"
        assert "goal" in dial, f"no goal in {dial}"
        assert len(dial["log"]) > 0, f"no log in {dial}"
        assert len(self.all_domains) == 1

        encoded_dial = []
        prev_dbres = {}
        for t in dial['log']:
            enc = {}
            enc['dial_id'] = fn
            enc['turn_num'] = t['turn']
            enc['turn_domain'] = self.all_domains
            enc['user'] = t['original_user'] if self.use_raw_utterance else t['user']
            enc['delex_resp'] = t['restored_response'] if self.use_raw_utterance else t['response']
            # TODO: get concrete response
            concrete_resp = t['original_nodelx_resp']
            if self.gen_concrete_resp:
                enc['concrete_resp'] = concrete_resp if self.use_raw_utterance else concrete_resp.lower()
            enc["resp4db_summary"] = concrete_resp
            enc["raw_resp"] = concrete_resp

            belief_state = {}
            raw_constraint = json.loads(t['constraint'])
            for k in self.otlg.informable_slots:
                assert k in raw_constraint
                value = raw_constraint[k]
                if len(value) >= 1:
                    domain, slot = k.split('-')
                    if domain not in belief_state:
                        belief_state[domain] = {}
                    belief_state[domain][slot] = ' '.join(value)
                    if len(value) > 1:
                        print(f"[{domain}] [{slot}] {value} > 1")
            enc['bspn'] = belief_state

            db_result = self.constraint_to_DBpointer(belief_state, enc['turn_domain'], enc)
            db_match = db_result[enc['turn_domain'][0]]['content'][0] if len(db_result) > 0 else None
            # TODO: Unified management when constructing instruct data or inference
            # TODO: only display the number of matched items when the db result is same as previous turns ?
            # cur_db_result = {}
            # for dom, res in db_result.items():
            #     prev_res = prev_dbres.get(dom, None)
            #     if prev_res != res:
            #         prev_dbres[dom] = res
            #         cur_db_result[dom] = res
            enc['dbres'] = db_result
            enc['db_match'] = f"{t['db_match']} == {db_match} = {t['db_match'] == db_match}"

            sys_act = ""
            user_request = t['user_request'].strip()
            if user_request:
                num_raw = len(user_request.split())
                slot2count = Counter(re.findall(r"\[value_(\w+)\]", user_request))
                assert num_raw == sum(slot2count.values()), f"{user_request}: {num_raw} != {sum(slot2count.values())}"
                request_slots = list(slot2count.keys())
                if not set(request_slots).issubset(set(self.requestable_slots)):
                    request_slots = [s for s in request_slots if s in self.requestable_slots]
                    print(f"[{fn}] user request '{user_request}' is not subset of {self.requestable_slots} -> {request_slots}")
                if request_slots:
                    sys_act += f"[inform] {' '.join(request_slots)} "
            sys_request = t['sys_request'].strip()
            if sys_request:
                assert len(re.findall(r"\[value_(\w+)\]", sys_request)) == 0, \
                    f"Incorrect system request format: {sys_request}"
                cand_slots = sys_request.split()
                if not set(cand_slots).issubset({'food', 'pricerange', 'area', 'price', 'range'}):
                    legal_req_slots = [s for s in cand_slots if s in {'food', 'pricerange', 'area', 'price', 'range'}]
                    print(f"[{fn}] Incorrect system request: '{sys_request}' -> {legal_req_slots}")
                    sys_request = ' '.join(legal_req_slots)
                if sys_request:
                    sys_act += f"[request] {sys_request} "
            if sys_act:
                sys_act = f"[{enc['turn_domain'][0]}] " + sys_act
            if not self.disable_sys_act:
                enc['aspn'] = sys_act.strip()

            encoded_dial.append(enc)
        return encoded_dial

    def constraint_to_DBpointer(self, constraint_dict, turn_domains, turn_data):
        legal_domains = [dom for dom in turn_domains if dom in self.all_domains]
        if len(legal_domains) == 0 or not constraint_dict:
            return {}

        constraint_dict = json.loads(json.dumps(constraint_dict).lower())

        res = {}
        for dom in legal_domains:
            constraint = constraint_dict.get(dom, {})
            legal_constraint = {}
            for slot, val in constraint.items():
                if slot in self.schema[dom]['informable slots']:
                    legal_constraint[slot] = val
            if not legal_constraint:
                continue
            match_results = self.db_json_search(legal_constraint)
            res[dom] = {
                "content": [len(match_results)],
                "hash_value": [len(match_results), 0]
            }
            summary, md5_hash = self.summarize_db_result(
                match_results, self.schema[dom]['informable slots'],
                delex_resp=turn_data.get('delex_resp'),
                concrete_resp=turn_data.get('resp4db_summary'))
            res[dom]["content"].append(summary)
            res[dom]["hash_value"] = md5_hash
        return res

    def db_json_search(self, constraints):
        match_results = []
        for entry in self.db_json:
            if 'food' not in entry:
                entry_values = entry['area'] + ' ' + entry['pricerange']
            else:
                entry_values = entry['area'] + ' ' + entry['food'] + ' ' + entry['pricerange']
            entry_values = entry_values.lower()
            match = True
            for s, v in constraints.items():
                if v in self.otlg.skip_mapping['dontcare']:
                    continue
                if v not in entry_values:  # v is str here
                    match = False
                    break
            if match:
                match_results.append(entry)
        # print(len(match_results))
        return match_results

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  'bspn', 'real_bspn_gen', 'bspn_gen', 'aspn', 'real_aspn_gen', 'aspn_gen',
                  'delex_resp', 'delex_resp_gen', 'concrete_resp', 'concrete_resp_gen', 'resp', 'resp_gen',
                  'raw_resp',
                  'dbres', 'db_return']
        for turn_results in dialogue_results:
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    if key in ['bspn', 'bspn_gen']:
                        # flatten bspn
                        constraint = {}
                        for dom, cons in value.items():
                            if not isinstance(cons, dict):
                                continue
                            for slot, val in cons.items():
                                constraint[f"{dom}-{slot}"] = val
                        value = constraint
                    entry[key] = value

                results.append(entry)

        return results, fields

