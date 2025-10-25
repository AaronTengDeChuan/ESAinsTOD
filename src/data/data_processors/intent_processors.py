import math

from src.data import GenProcessor

import os
import re
import copy
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from itertools import zip_longest

from src.utils import utils
from src.args import str2bool


class IntentProcessor(GenProcessor):

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("IntentProcessor")
        group.add_argument("--few_shot_evaluation", type=str2bool, default=False,
                           help="Whether to run few-shot evaluation.")
        group.add_argument("--num_shot", type=int, default=30, help="The number of few-shot examples.")
        return group

    def __init__(self, hparams):
        super().__init__(hparams)

        self.intent_dropout_ratio = hparams.intent_dropout_ratio
        if self.intent_dropout_ratio > 0 and self.do_infer:
            print(utils.highlight(f"Set 'intent_dropout_ratio' to '0.0' for inference.", "red"))
            self.intent_dropout_ratio = 0.0
        self.num_duplication = hparams.num_duplication
        if self.num_duplication > 1 and self.do_infer:
            print(utils.highlight(f"Set 'num_duplication' to '1' for inference.", "red"))
            self.num_duplication = 1
        if self.num_duplication > 1:
            assert self.training_level == "turn_level", f"When 'num_duplication' > 1, 'training_level' must be 'turn_level'."
            # assert self.intent_dropout_ratio > 0.0, f"When 'num_duplication' > 1, 'intent_dropout_ratio' must > 0.0."

        self.few_shot_evaluation = hparams.few_shot_evaluation
        self.num_shot = hparams.num_shot

        self.has_intent = True
        self.has_sys_act = False
        self.has_concrete_resp = False
        self.resp4eval = None

        return

    def read_single_turn(self, data_file, split):
        print(f"Reading examples from '{data_file}' ...")
        examples = []

        with open(data_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        turn_counter = []

        for dialog_id in tqdm(input_data):
            one_dial = []
            turns = input_data[dialog_id]['turns']
            turn_counter.append(len(turns))
            for turn in turns:
                label = turn['label']
                role = turn['role']
                text = turn['text']

                if role == "user":
                    assert "DEFAULT_DOMAIN" in label
                    one_dial.append({
                        "dial_id": f"{split}-{dialog_id}",
                        "turn_num": turn["turn_id"],
                        "label": label["DEFAULT_DOMAIN"],
                        "user": text,
                        "extra_info": turn.get("extra_info", {})
                    })
            examples.append(one_dial)
        print(f"Statistics for Turn Num: {json.dumps(Counter(turn_counter))}")
        return examples

    @staticmethod
    def get_intent2ids(examples):
        intent2ids = defaultdict(list)
        for idx, dial in enumerate(examples):
            dom = dial[0]['turn_domain'][0]
            intent = dial[0]['turn_intent'][dom][0]
            intent2ids[f"{dom}_{intent}"].append(idx)
        print(f"Statistics for Intent Num: {json.dumps(Counter([len(v) for v in intent2ids.values()]))}")
        return intent2ids

    @staticmethod
    def select_few_shot_examples(intent2ids, num_shot=30):
        intents = list(intent2ids.keys())
        cand_intents = random.choices(intents, k=num_shot)
        cand_ids = []
        for intent in cand_intents:
            cand_ids.append(random.choice(intent2ids[intent]))
        return cand_ids

    def create_few_shot_data(self, src_data, target_data, num_shot=30):
        final_data = []
        intent2ids = self.get_intent2ids(src_data)
        for dial in target_data:
            assert len(dial) == 1, f"Invalid dial: {dial}"
            cand_ids = self.select_few_shot_examples(intent2ids, num_shot=num_shot)
            new_dial = []
            for turn_idx, sample_idx in enumerate(cand_ids):
                turn = copy.deepcopy(src_data[sample_idx][0])
                turn['turn_num'] = turn_idx
                new_dial.append(turn)
            target_turn = copy.deepcopy(dial[0])
            target_turn['turn_num'] = len(new_dial)
            new_dial.append(target_turn)
            final_data.append(new_dial)
        return final_data

    def get_eval_data(self, split_name='test'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        eval_data = name_to_set[split_name]
        assert len(eval_data) > 0, "Please load data first."

        if self.few_shot_evaluation:
            few_shot_data = self.create_few_shot_data(self.train, eval_data, num_shot=self.num_shot)
            # only generate for the last turn
            eval_data = []
            for dial in few_shot_data:
                dial_id, sys_msg, dial_body, prev_meta = self._generate_instruct_dial(dial[:-1])
                last_turn = copy.deepcopy(dial[-1])
                last_turn['turn_num'] = 0
                separate_texts = [t for turn in dial_body for t in turn[0]]
                last_turn['extra_prompt'] = ''.join(separate_texts)
                last_turn['prev_domain_schema'] = prev_meta.pop('prev_domain_schema')
                last_turn['prev_intent_schema'] = prev_meta.pop('prev_intent_schema')
                last_turn['prev_dbres'] = prev_meta.pop('prev_dbres')
                eval_data.append([last_turn])

        num_dials = len(eval_data)
        num_turns = sum([len(dial) for dial in eval_data])
        dropouts, _ = self.count_dropouts(eval_data)
        stats = {
            "num_dials": num_dials,
            "num_turns": num_turns,
            "intent_dropouts": dropouts,
            "real_intent_dropout_ratio": dropouts / num_turns,
        }
        one_sample = copy.deepcopy(eval_data[0])
        extra_prompt = one_sample[0].pop('extra_prompt', '')
        print(f"Extra Prompt for {split_name}:\n{extra_prompt}")
        print(f"Other Fields for {split_name}:\n{json.dumps(one_sample, indent=2)}")

        return eval_data, stats

    def detailed_postprocess(self):
        # duplicate data
        if self.num_duplication > 1:
            print(f"Duplicate training data {self.num_duplication} times ...")
            self.train = [copy.deepcopy(dial) for _ in range(self.num_duplication) for dial in self.train]

        if self.training_level == "session_level":
            self.train = self._convert_to_session_level(self.train)
            self.dev = self._convert_to_session_level(self.dev)
            # self.test = self._convert_to_session_level(self.test)

    def count_dropouts(self, data):
        dropouts, total = 0, 0
        for dial in data:
            for turn in dial:
                total += 1
                if set(turn['turn_domain']) != set(turn['turn_intent']):
                    dropouts += 1
        return dropouts, total

    def dropout_intent(self, domain_schema, turn_data):
        if not turn_data['dial_id'].startswith('train'):
            return domain_schema
        new_schema = copy.deepcopy(domain_schema)
        for domain in domain_schema:
            if random.random() < self.intent_dropout_ratio:
                candidate_intents = domain_schema[domain]['intents']
                gold_intents = turn_data['turn_intent'].pop(domain)
                new_schema[domain]['intents'] = [intent for intent in candidate_intents if intent not in gold_intents]
        return new_schema

    def _organize_instruct_data(self, data):
        dial_id = data[0][-1]['dial_id']
        dev_or_test = dial_id.startswith('dev') or dial_id.startswith('test')
        if self.num_shot == 1:
            dev_or_test = False

        unordered_ids = list(range(len(data)))
        random.shuffle(unordered_ids)
        raw_instruct_data = []
        new_id = []
        num_curr_samples = 0
        curr_data = []
        curr_tokens = 0
        curr_lengths = []
        for did in tqdm(unordered_ids, desc=f"Organizing instruct data for {self.data_source}"):
            dial = data[did]
            dial_id, sys_msg, dial_body, _ = self._generate_instruct_dial(dial)
            dial_body[-1][0][-1] = dial_body[-1][0][-1].rstrip() + f"{self.eod_token}\n"

            new_tokens = (self._tokenize_text(sys_msg) +
                          sum([self._tokenize_text(text) for turn in dial_body for text in turn[0]]))
            if curr_tokens + new_tokens > self.model_max_length or (
                    not dev_or_test and 0 < self.num_shot <= num_curr_samples):
                raw_instruct_data.append((new_id, num_curr_samples, curr_data, curr_tokens, curr_lengths))
                new_id = []
                num_curr_samples = 0
                curr_data = []
                curr_tokens = 0
                curr_lengths = []
            new_id += [dial_id]
            num_curr_samples += 1
            curr_data += [([sys_msg], [False])] + dial_body
            curr_tokens += new_tokens
            curr_lengths.append(new_tokens)
        raw_instruct_data.append((new_id, num_curr_samples, curr_data, curr_tokens, curr_lengths))

        instruct_data = []
        dial_tokens = []
        dial_lengths = []
        for new_id, num_curr_samples, curr_data, curr_tokens, curr_lengths in raw_instruct_data:
            instruct_data.append({
                "id": "-".join(new_id),
                "source": self.data_source,
                "conversations": [{"turn_texts": t, "turn_labels": l} for t, l in curr_data],
                "text_lengths": json.dumps(curr_lengths),
                "total_length": curr_tokens,
            })
            dial_tokens.append(curr_tokens)
            dial_lengths.append(num_curr_samples)

        long_dial_path = os.path.join(self.save_dir, self.data_source, "long_dial.txt")
        with open(long_dial_path, "a", encoding="utf-8") as f:
            max_idx = np.argmax(dial_tokens)
            save_sample = instruct_data[max_idx]
            separate_texts = [t for turn in save_sample['conversations'] for t in turn["turn_texts"]]
            concat_text = ''.join(separate_texts)
            tokens = self._tokenize_text(concat_text, return_tokens=True)
            f.write(f"{save_sample['id']}: "
                    f"{json.dumps(save_sample['text_lengths'])} -> {save_sample['total_length']} >|< {len(tokens)}\n")
            f.write(f"{concat_text}\n\n")

        stats = pd.DataFrame({"tokens": np.array(dial_tokens), 'turns': np.array(dial_lengths)})
        stats['tokens_per_turn'] = stats['tokens'] / stats['turns']
        stats_dict = stats.describe().to_dict()
        stats_dict['real_dials'] = len(data)
        stats_dict['total_tokens'] = stats_dict['tokens']['count'] * stats_dict['tokens']['mean']
        stats_dict['total_tokens (B)'] = stats_dict['total_tokens'] / 1e9
        stats_dict['train_tokens (B)'] = len(instruct_data) * self.model_max_length / 1e9
        dropouts, total = self.count_dropouts(data)
        stats_dict['intent_dropouts'] = dropouts
        stats_dict['num_turns'] = total
        stats_dict['real_intent_dropout_ratio'] = dropouts / total
        return instruct_data, stats_dict

    def _convert_to_session_level(self, data):
        intent2ids = self.get_intent2ids(data)
        all_ids = []
        for group in zip_longest(*intent2ids.values()):
            group = [idx for idx in group if idx is not None]
            all_ids.extend(group)
        session_data = []
        group_num = math.ceil(len(all_ids) / self.num_shot)
        for group_id in range(group_num):
            group = all_ids[group_id * self.num_shot: (group_id + 1) * self.num_shot]
            random.shuffle(group)
            session = []
            for idx in group:
                turn = data[idx][0]
                turn['turn_num'] = len(session)
                session.append(turn)
            session_data.append(session)
        return session_data

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  'ispn', 'real_ispn_gen', 'ispn_gen']
        for turn_results in dialogue_results:
            dial_turns = []
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    entry[key] = value
                dial_turns.append(entry)
            results.append(dial_turns)

        return results, fields


class HWUProcessor(IntentProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'single_turn/hwu'
        self.dataset_path = os.path.join(self.data_root, self.data_source)

        self.load_schema(os.path.join(self.dataset_path, "NLU-Data-Home-Domain-Annotated-All.csv"))
        self._load_data(need_processing=need_processing)
        return

    def _load_data(self, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        train_file = os.path.join(
            self.dataset_path,
            f"train_{self.few_shot_training}.json" if self.few_shot_training else "train.json")
        self.train = self.read_single_turn(train_file, 'train')
        self.dev = self.read_single_turn(os.path.join(self.dataset_path, "valid.json"), 'dev')
        self.test = self.read_single_turn(os.path.join(self.dataset_path, "test.json"), 'test')
        self._get_detailed_data(self.train)
        self._get_detailed_data(self.dev)
        self._get_detailed_data(self.test)
        self.detailed_postprocess()

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.train)}\tdev: {len(self.dev)}\ttest: {len(self.test)}")

    def load_schema(self, schema_file):
        csv_data = pd.read_csv(schema_file, sep=';')
        schema = csv_data.groupby('scenario')['intent'].unique().to_dict()
        self.all_domains = list(schema.keys())
        self.schema = {k: {"intents": vs.tolist()} for k, vs in schema.items()}
        with open(os.path.join(self.dataset_path, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(self.schema, f, ensure_ascii=False, indent=2)
        print(f"Loaded schema from '{schema_file}'.")
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")

    def _get_detailed_data(self, data):
        domain_intent_counter = defaultdict(Counter)
        split_counter = defaultdict(set)
        intent2labels = defaultdict(set)
        for dial in data:
            for example in dial:
                label = example.pop('label')
                assert isinstance(label, dict) and len(label.keys()) == 1, f"Invalid label: {label}"
                label_text = list(label.keys())[0]

                extra_info = example.pop('extra_info')
                assert "intent_label" in extra_info, f"Invalid extra_info: {extra_info}"
                intent_label = extra_info['intent_label']
                intent2labels[label_text].add(intent_label)
                assert len(intent2labels[label_text]) <= 1, \
                    f"Duplicate intent label: {label_text}: {intent2labels[label_text]}"

                fields = label_text.split('_')
                split_counter[len(fields)].add(label_text)
                domain, intent = fields[0], '_'.join(fields[1:])
                assert domain in self.schema, f"Invalid domain: {domain}"
                assert intent in self.schema[domain]['intents'], f"Invalid intent: {intent}"
                domain_intent_counter[domain][intent] += 1

                example['turn_domain'] = [domain]
                example['turn_intent'] = {domain: [intent]}
        print(f"Statistics for Split Num: {[f'{k}: {len(v)}' for k, v in split_counter.items()]}")
        domain2intent = {
            f"{domain} [{len(intent_counter)}: {sum(intent_counter.values())}]": str(list(intent_counter.keys())) for
            domain, intent_counter in domain_intent_counter.items()}
        print(len(domain2intent), json.dumps(domain2intent, indent=2))


class ClincProcessor(IntentProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'single_turn/clinc'
        self.dataset_path = os.path.join(self.data_root, self.data_source)

        self.load_schema(os.path.join(self.dataset_path, "domains.json"))
        self._load_data(need_processing=need_processing)
        return

    def load_schema(self, schema_file):
        with open(schema_file, "r", encoding="utf-8") as f:
            schema = json.load(f)
        self.all_domains = list(schema.keys())
        self.schema = {k: {"intents": vs} for k, vs in schema.items()}
        self.intent2domain = self.get_intent2domain(self.schema)
        print(f"Loaded schema from '{schema_file}'.")
        print(f"schema:\n{json.dumps(self.schema, indent=2)}")

    @staticmethod
    def get_intent2domain(schema):
        intent2domain = {}
        for domain, dom_schema in schema.items():
            intents = dom_schema['intents']
            assert len(set(intents)) == len(intents), f"Duplicate intent in {domain}: {intents}"
            for intent in intents:
                intent2domain[intent] = domain
        return intent2domain

    def _load_data(self, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        train_file = os.path.join(
            self.dataset_path,
            f"train_{self.few_shot_training}.json" if self.few_shot_training else "train.json")
        self.train = self.read_single_turn(train_file, 'train')
        self.dev = self.read_single_turn(os.path.join(self.dataset_path, "valid.json"), 'dev')
        self.test = self.read_single_turn(os.path.join(self.dataset_path, "test.json"), 'test')
        self._get_detailed_data(self.train)
        self._get_detailed_data(self.dev)
        self._get_detailed_data(self.test)
        self.detailed_postprocess()

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.train)}\tdev: {len(self.dev)}\ttest: {len(self.test)}")

    def _get_detailed_data(self, data):
        domain_intent_counter = defaultdict(Counter)
        intent2labels = defaultdict(set)
        for dial in data:
            for example in dial:
                label = example.pop('label')
                assert isinstance(label, dict) and len(label.keys()) == 1, f"Invalid label: {label}"
                label_text = list(label.keys())[0]

                extra_info = example.pop('extra_info')
                assert "intent_label" in extra_info, f"Invalid extra_info: {extra_info}"
                intent_label = extra_info['intent_label']
                intent2labels[label_text].add(intent_label)
                assert len(intent2labels[label_text]) <= 1, \
                    f"Duplicate intent label: {label_text}: {intent2labels[label_text]}"

                domain, intent = self.intent2domain[label_text], label_text
                domain_intent_counter[domain][intent] += 1
                example['turn_domain'] = [domain]
                example['turn_intent'] = {domain: [intent]}
        domain2intent = {
            f"{domain} [{len(intent_counter)}: {sum(intent_counter.values())}]": str(list(intent_counter.keys())) for
            domain, intent_counter in domain_intent_counter.items()}
        print(len(domain2intent), json.dumps(domain2intent, indent=2))


class BankingProcessor(IntentProcessor):

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'single_turn/banking'
        self.dataset_path = os.path.join(self.data_root, self.data_source)

        self.all_domains = ["banking"]
        self._load_data(need_processing=need_processing)
        return

    def _load_data(self, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        train_file = os.path.join(
            self.dataset_path,
            f"train_{self.few_shot_training}.json" if self.few_shot_training else "train.json")
        self.train = self.read_single_turn(train_file, 'train')
        self.dev = self.read_single_turn(os.path.join(self.dataset_path, "valid.json"), 'dev')
        self.test = self.read_single_turn(os.path.join(self.dataset_path, "test.json"), 'test')

        train_intents = self._get_detailed_data(self.train)
        dev_intents = self._get_detailed_data(self.dev)
        test_intents = self._get_detailed_data(self.test)
        all_intents = list(set(train_intents + dev_intents + test_intents))
        # set schema
        self.schema = {"banking": {"intents": all_intents}}
        with open(os.path.join(self.dataset_path, "schema.json"), "w", encoding="utf-8") as f:
            json.dump(self.schema, f, ensure_ascii=False, indent=2)
        self.detailed_postprocess()

        print(json.dumps(self.test[0], indent=2))
        print(f"train: {len(self.train)}\tdev: {len(self.dev)}\ttest: {len(self.test)}")

    def _get_detailed_data(self, data):
        intent_counter = Counter()
        intent2labels = defaultdict(set)
        for dial in data:
            for example in dial:
                label = example.pop('label')
                assert isinstance(label, dict) and len(label.keys()) == 1, f"Invalid label: {label}"
                label_text = list(label.keys())[0]

                extra_info = example.pop('extra_info')
                assert "intent_label" in extra_info, f"Invalid extra_info: {extra_info}"
                intent_label = extra_info['intent_label']
                intent2labels[label_text].add(intent_label)
                assert len(intent2labels[label_text]) <= 1, \
                    f"Duplicate intent label: {label_text}: {intent2labels[label_text]}"

                intent_counter[label_text] += 1
                example['turn_domain'] = ["banking"]
                example['turn_intent'] = {"banking": [label_text]}
        print(len(intent_counter), json.dumps(intent_counter, indent=2))
        return list(intent_counter.keys())
