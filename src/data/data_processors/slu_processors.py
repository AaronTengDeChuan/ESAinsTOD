from src.data.data_processors.gen_processor import GenProcessor

import os
import json
import copy
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter

from src.utils import utils
from src.args import str2bool


class SnipsProcessor(GenProcessor):

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("SLUProcessor")
        group.add_argument("--few_shot_evaluation", type=str2bool, default=False,
                           help="Whether to run few-shot evaluation.")
        group.add_argument("--num_shot", type=int, default=30, help="The number of few-shot examples.")
        return group

    def __init__(self, hparams, need_processing=True):
        super().__init__(hparams)

        self.data_source = f'single_turn/snips'
        self.dataset_path = os.path.join(self.data_root, self.data_source)

        self.num_duplication = hparams.num_duplication
        if self.num_duplication > 1 and self.do_infer:
            print(utils.highlight(f"Set 'num_duplication' to '1' for inference.", "red"))
            self.num_duplication = 1
        if self.num_duplication > 1:
            assert self.training_level == "turn_level", f"When 'num_duplication' > 1, 'training_level' must be 'turn_level'."

        self.few_shot_evaluation = hparams.few_shot_evaluation
        self.num_shot = hparams.num_shot

        # These variables indicate whether the dataset has corresponding annotations.
        self.has_intent = False
        self.has_sys_act = False
        self.has_concrete_resp = False
        self.resp4eval = None

        self._load_data(need_processing=need_processing)
        return

    def detailed_postprocess(self):
        # duplicate data
        if self.num_duplication > 1:
            print(f"Duplicate training data {self.num_duplication} times ...")
            self.train = [copy.deepcopy(dial) for _ in range(self.num_duplication) for dial in self.train]

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
        return instruct_data, stats_dict

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
                        "dial_id": dialog_id if dialog_id.startswith(split) else f"{split}-{dialog_id}",
                        "turn_num": turn["turn_id"],
                        "label": label["DEFAULT_DOMAIN"],
                        "user": text,
                        "extra_info": turn.get("extra_info", {})
                    })
            examples.append(one_dial)
        print(f"Statistics for Turn Num: {json.dumps(Counter(turn_counter))}")
        return examples

    def _load_data(self, need_processing=True):
        if not need_processing:
            print(f"Skip processing data for {self.data_source} ...")
            return

        self.train = self.read_single_turn(os.path.join(self.dataset_path, "train_stack.json"), 'train')
        self.dev = self.read_single_turn(os.path.join(self.dataset_path, "dev_stack.json"), 'dev')
        self.test = self.read_single_turn(os.path.join(self.dataset_path, "test_stack.json"), 'test')

        intent2slots = defaultdict(set)
        self._get_detailed_data(self.train, intent2slots)
        self._get_detailed_data(self.dev, intent2slots)
        self._get_detailed_data(self.test, intent2slots)
        self.detailed_postprocess()

        self.all_intents = list(intent2slots.keys())
        # set schema based on domain-intent-slot structure
        # self.all_domains = ["snips"]
        # self.schema = {"snips": {"intents": list(intent2slots.keys())}}
        # TODO: 'informable slots' or 'required_slots'
        # self.intent_schema = {"snips": {intent: {"required_slots": list(set(slots))}
        #                                 for intent, slots in intent2slots.items()}}
        # set schema where recast intents as domains
        self.all_domains = list(intent2slots.keys())
        self.schema = {intent: {"informable slots": list(set(slots))} for intent, slots in intent2slots.items()}
        with open(os.path.join(self.dataset_path, "domain_slot_schema.json"), "w", encoding="utf-8") as f:
            json.dump(self.schema, f, ensure_ascii=False, indent=2)
        print(f"schema: {json.dumps(self.schema, indent=2)}")

        print(json.dumps(self.train[0], indent=2))
        print(f"train: {len(self.train)}\ndev: {len(self.dev)}\ntest: {len(self.test)}")

    def _get_detailed_data(self, data, intent2slots):
        for dial in data:
            for example in dial:
                label = example.pop('label')
                extra_info = example.pop('extra_info')
                assert isinstance(label, dict) and len(label.keys()) == 1, f"Invalid label: {label}"
                intent = list(label.keys())[0]
                slots = list(label[intent].keys())
                intent2slots[intent].update(slots)

                state = {}
                for slot in slots:
                    slot_annotations = label[intent][slot]
                    # assert len(slot_annotations) == 1, f"Invalid slot annotations: {slot_annotations}"
                    st, ed = slot_annotations[-1]["span"]
                    state[slot] = example["user"][st:ed].strip()

                example['turn_domain'] = [intent]
                example['bspn'] = {intent: state} if state else {}
                # example['turn_domain'] = ["snips"]
                # example['turn_intent'] = {"snips": [intent]}
                # example['bspn'] = {"snips": state} if state else {}
        return

    def wrap_result_lm(self, dialogue_results):
        results = []
        fields = ['dial_id', 'turn_num', 'user', 'dspn', 'real_dspn_gen', 'dspn_gen',
                  # 'ispn', 'real_ispn_gen', 'ispn_gen',
                  'bspn', 'real_bspn_gen', 'bspn_gen']
        for turn_results in dialogue_results:
            dial_turns = []
            for turn_idx, turn in enumerate(turn_results):
                entry = {}
                for key in fields:
                    value = turn.get(key, '')
                    if key in ['dspn', 'dspn_gen']:
                        entry[key.replace('dspn', 'ispn')] = value
                        # pass
                    elif key in ['bspn', 'bspn_gen']:
                        value = value.get(turn[key.replace('bspn', 'dspn')][0], {})
                        value = value if isinstance(value, dict) else {}
                    entry[key] = value
                dial_turns.append(entry)
            results.append(dial_turns)

        return results, fields
