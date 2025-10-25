# coding: utf-8
import copy
import os
import re
import time
import math
import roman
import random
from typing import List, Dict, Tuple, Union, Optional
from collections import OrderedDict, defaultdict, Counter
from itertools import chain
from functools import partial
import json
import requests

import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import bisect
import sqlite3 as sql
from multiprocessing import Pool

import torch
from transformers import AutoConfig
from transformers import AutoTokenizer, LlamaTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM


# if vllm is installed, import it
try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass

from src.args import str2bool

from src.utils import utils


class GenProcessor(object):
    tod_roles = {
        "system": "system",
        "user": "user",
        "assistant": "AI assistant",
    }

    # 请作为一名人工智能助手，在任务型对话场景中与用户交互以完成其需求。针对用户的每一条消息，跟随以下的指令，生成中间结果直到助手回复。
    tod_system_message = f"Please act as an AI assistant to interact with the user in a task-oriented dialogue scenario to meet his/her needs.\nFor each message from the user, follow the instructions below to generate intermediate results until the assistant replies:"
    simple_tod_system_message = f"Please act as an AI assistant to interact with the user in a task-oriented dialogue scenario to meet his/her needs.\nAlso, please adhere to the following rules:"
    tod_intermediate_prompts = {
        # 从下列领域中，识别出用户消息涉及到的领域
        "domain": f"From the following domains, identify the domains involved in the user message: ",
        #
        "intent": f"Select the correct intent that is expressed in the user message among the list of intents provided.",
        # 遵循以下槽值对格式来维护从对话开始至今的用户需求:
        "constraint": f"Please maintain the user's needs from the beginning of the dialogue to the present in the following format of slot-value pairs: ",
        # 在生成助手回复前，先概括助手应该采取的行为
        "sys_act": "Before generating the assistant's reply, summarize the actions the assistant should take.",
        # 生成去词汇化的助手回复
        "delex_resp": "Generate delexicalized assistant reply.",
        # 生成具体的助手回复
        "concrete_resp": "Generate concrete assistant reply.",
    }

    pptod_prompts = {
        "domain": "translate dialogue to domain:",
        "intent": "translate dialogue to user intent:",
        "constraint": "translate dialogue to belief state:",
        "sys_act": "translate dialogue to dialogue action:",
        "delex_resp": "translate dialogue to system response:"
    }

    @classmethod
    def add_cmdline_argument(cls, parser):
        group = parser.add_argument_group("Processor")
        group.add_argument("--processor_version", type=str, default="v1.0",
                           help="The version of the dataset.")

        group.add_argument("--special_tokens_file", type=str, default="special_tokens.json",
                           help="The special tokens file path.")
        group.add_argument("--project_root", type=str, default='',
                           help="The root directory of the project.")
        group.add_argument("--raw_data_root", type=str, default='raw_data',
                            help="The root directory of the raw data.")
        group.add_argument("--data_name", type=str, required=True,
                           choices=["camrest", "kvret", "multiwoz", "sgd", "frames", "bitod", "star", "hwu", "clinc", "banking", "snips"],
                           help="The name of dataset.")
        group.add_argument("--data_version", type=str, default="", help="The version of multiwoz or sgd datasets.")
        group.add_argument("--exp_domains", type=str, default=None)
        group.add_argument("--data_processed", type=str, default='data_for_space_processed.json')
        group.add_argument("--model_max_length", type=int, default=4096,
                           help="The max length of the input sequence.")
        group.add_argument("--tokenizer_path", type=str, default='',
                           help="The path of the tokenizer.")
        group.add_argument("--llama", action="store_true",
                           help="Whether to use the llama model.")
        group.add_argument("--model_name_or_path", type=str, default='',
                           help="The path of the model.")
        group.add_argument("--infer_backend", type=str, default='guidance', choices=['vllm'],
                           help="The backend of inference.")
        group.add_argument("--num_processes", type=int, default=1, help="Number of processes to infer.")
        group.add_argument("--host", type=str, default="localhost")
        group.add_argument("--port", nargs='+', type=int, default=[8000])
        # model loading parameters
        group.add_argument("--tie_word_embeddings", type=str2bool, default=None, help="")
        # sft data parameters
        group.add_argument("--pptod_strategy", type=str2bool, default=False, help="")
        group.add_argument("--use_raw_utterance", type=str2bool, default=False, help="")
        group.add_argument("--no_task_instruction", type=str2bool, default=False, help="")
        group.add_argument("--simple_instruction", type=str2bool, default=False, help="")
        group.add_argument("--no_schema_info", type=str2bool, default=False, help="")
        group.add_argument("--few_shot_training", type=str, default='', help="")
        group.add_argument("--num_duplication", type=int, default=1, help="Copy the data for multiple times.")
        group.add_argument("--training_level", type=str, default='session_level',
                           choices=['turn_level', 'session_level'],
                           help="The training level of the model.")
        group.add_argument("--history_ratio", type=float, default=0.6, help="")
        group.add_argument("--turn_schema", type=str2bool, default=False, help="")
        group.add_argument("--turn_dbres", type=str2bool, default=False, help="")
        group.add_argument("--inform_request_schema", type=str2bool, default=True, help="")
        group.add_argument("--disable_intent", type=str2bool, default=False, help="")
        # TODO: for intent datasets, remove gold intent from provided intent candidates and set gold intent as empty with certain probability
        group.add_argument("--intent_dropout_ratio", type=float, default=0.0, help="")
        group.add_argument("--detailed_db_result", type=str2bool, default=False, help="")
        group.add_argument("--disable_sys_act", type=str2bool, default=False, help="")
        # intermediate results
        group.add_argument("--gen_concrete_resp", type=str2bool, default=True, help="")
        group.add_argument("--concrete_resp_first", type=str2bool, default=False, help="")
        # control losses of intermediate results
        group.add_argument("--disable_concrete_resp_loss", type=str2bool, default=False, help="")
        # generation parameters
        group.add_argument("--start_with_bos", type=str2bool, default=True,
                            help="Generation starts with bos token or not.")
        group.add_argument("--do_sample", type=str2bool, default=True,
                           help="Whether to use sampling.")
        group.add_argument("--temperature", type=float, default=0.6,
                           help="The value used to module the next token probabilities.")
        group.add_argument("--top_p", type=float, default=0.9,
                           help="The cumulative probability of parameter highest probability vocabulary tokens.")
        group.add_argument("--num_return_sequences", type=int, default=1,
                           help="Number of output sequences to return for the given prompt.")
        group.add_argument("--num_beams", type=int, default=1,
                           help="The number of beams for beam search.")
        group.add_argument("--max_tokens", type=int, default=80, help="The max tokens of the generated sequence.")
        group.add_argument("--num_history_turns", type=int, default=None, help="The number of history turns.")
        # evaluation parameters
        group.add_argument("--accumulate_belief_state", type=str2bool, default=False, help="")
        group.add_argument("--enable_capital", type=str2bool, default=False, help="")
        group.add_argument("--pseudo_intent", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_domain", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_intent", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_bspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_aspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_prev_resp", type=str2bool, default=False, help="")
        # Note: separate controlling parameters for delex resp and concrete resp
        group.add_argument("--use_true_prev_concrete_resp", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_domain", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_intent", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_bspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_curr_aspn", type=str2bool, default=False, help="")
        group.add_argument("--use_true_bspn_for_ctr_eval", type=str2bool, default=True, help="")
        group.add_argument("--use_true_domain_for_ctr_eval", type=str2bool, default=True, help="")
        group.add_argument("--use_true_resp_for_ctr_eval", type=str2bool, default=False, help="")
        group.add_argument("--zero_shot_enhancement", type=str2bool, default=False,
                           help="Whether to enable zero-shot enhancement")
        group.add_argument("--comments", type=str, default=None, help="Extra comments for evaluation.")

        return group

    def __init__(self, hparams):
        self.processor_version = hparams.processor_version

        self.debug = hparams.debug
        self.do_process = hparams.do_process
        self.do_infer = hparams.do_infer
        self.num_infer_samples = hparams.num_infer_samples

        self.project_root = hparams.project_root
        self.data_root = os.path.join(self.project_root, hparams.raw_data_root)
        self.data_name = hparams.data_name
        self.data_version = hparams.data_version
        self.save_dir = hparams.save_dir
        self.exp_domains = hparams.exp_domains

        self.train, self.dev, self.test = [], [], []
        self.instruct_train, self.instruct_dev, self.instruct_test = [], [], []

        self.backbone_model = None
        self.model_name_or_path = hparams.model_name_or_path
        self.tie_word_embeddings = hparams.tie_word_embeddings
        model_config = AutoConfig.from_pretrained(self.model_name_or_path)
        model_type = model_config.model_type
        if "llama" in model_type:
            self.llama = True
        else:
            if hparams.llama:
                print(f"Warning: model_type={model_type} does not contain 'llama', but hparams.llama is set to True.")
                print("Setting llama to False.")
            self.llama = False

            old_assistant = self.tod_roles["assistant"]
            self.tod_roles["assistant"] = "assistant"
            print(f"For '{model_type}', change assistant role from '{old_assistant}' to '{self.tod_roles['assistant']}'")

        # whether the model is trained on raw user and sys utterance
        self.model_trained_on_raw = "raw_uttr" in self.model_name_or_path
        if self.model_trained_on_raw:
            print(f"The model '{self.model_name_or_path}' is trained on raw user and system utterance.")

        self.model_max_length = hparams.model_max_length

        self.special_tokens_dict, _ = json.load(
            open(os.path.join(self.data_root, hparams.special_tokens_file)))
        self.tokenizer = None
        self.tokenizer_path = hparams.tokenizer_path or self.model_name_or_path
        self.set_tokenizer()

        self.no_task_instruction = hparams.no_task_instruction
        self.simple_instruction = hparams.simple_instruction
        assert not (self.no_task_instruction and self.simple_instruction), \
            "Cannot set both no_task_instruction and simple_instruction."
        self.no_schema_info = hparams.no_schema_info

        self.num_duplication = 1
        self.training_level = hparams.training_level
        self.few_shot_training = hparams.few_shot_training
        self.history_ratio = hparams.history_ratio
        self.turn_schema = hparams.turn_schema
        self.turn_dbres = hparams.turn_dbres
        self.inform_request_schema = hparams.inform_request_schema
        self.disable_intent = hparams.disable_intent
        self.intent_dropout_ratio = 0.0
        self.detailed_db_result = hparams.detailed_db_result
        self.disable_sys_act = hparams.disable_sys_act

        self.pptod_strategy = hparams.pptod_strategy
        self.enable_capital = hparams.enable_capital
        self.pseudo_intent = hparams.pseudo_intent
        self.use_raw_utterance = hparams.use_raw_utterance
        self.gen_concrete_resp = hparams.gen_concrete_resp
        self.concrete_resp_first = hparams.concrete_resp_first
        assert self.gen_concrete_resp or not self.concrete_resp_first, \
            f"gen_concrete_resp: {self.gen_concrete_resp}, concrete_resp_first: {self.concrete_resp_first}"

        self.disable_concrete_resp_loss = hparams.disable_concrete_resp_loss

        self.infer_backend = hparams.infer_backend
        self.num_processes = hparams.num_processes
        self.host = hparams.host
        self.port = hparams.port
        print(f"host: {self.host}, port: {self.port}")

        self.start_with_bos = hparams.start_with_bos
        self.temperature = hparams.temperature
        self.top_p = hparams.top_p
        self.do_sample = hparams.do_sample
        self.num_return_sequences = hparams.num_return_sequences
        self.num_beams = hparams.num_beams
        self.max_tokens = hparams.max_tokens
        self.num_history_turns = hparams.num_history_turns

        self.accumulate_belief_state = hparams.accumulate_belief_state
        self.use_true_prev_domain = hparams.use_true_prev_domain
        self.use_true_prev_intent = hparams.use_true_prev_intent
        self.use_true_prev_bspn = hparams.use_true_prev_bspn
        self.use_true_prev_aspn = hparams.use_true_prev_aspn
        self.use_true_prev_resp = hparams.use_true_prev_resp
        self.use_true_prev_concrete_resp = hparams.use_true_prev_concrete_resp
        self.use_true_curr_domain = hparams.use_true_curr_domain
        self.use_true_curr_intent = hparams.use_true_curr_intent
        self.use_true_curr_bspn = hparams.use_true_curr_bspn
        self.use_true_curr_aspn = hparams.use_true_curr_aspn
        self.use_true_domain_for_ctr_eval = hparams.use_true_domain_for_ctr_eval
        self.use_true_bspn_for_ctr_eval = hparams.use_true_bspn_for_ctr_eval
        self.use_true_resp_for_ctr_eval = hparams.use_true_resp_for_ctr_eval

        self.zero_shot_enhancement = hparams.zero_shot_enhancement
        self.comments = hparams.comments

        self.first_sample = True

        self.slot_value_format = {
            "format": {
                "{slot_name}": "{slot_value}"
            },
            "examples": {
                "slot_1": "value_1",
                "slot_2": "value_2"
            }
        }

        self.extra_rules4sys_act = ""

    def set_tokenizer(self):
        if self.llama:
            # self.tokenizer = LlamaTokenizer.from_pretrained(self.tokenizer_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        # self.tokenizer.add_special_tokens({"additional_special_tokens": list(self.special_tokens_dict.values())})
        for idx, (key, value) in enumerate(self.special_tokens_dict.items()):
            setattr(self, key, value)
            print(f"[{idx}] Add special token {key}={getattr(self, key)}")
        print("tokenizer.eos_token_id = {}".format(self.tokenizer.eos_token_id))
        print("tokenizer.pad_token_id = {}".format(self.tokenizer.pad_token_id))
        print("tokenizer.bos_token_id = {}".format(self.tokenizer.bos_token_id))
        print("tokenizer.additional_special_tokens : {}".format(" ; ".join([f"{tok}={tok_id}" for tok, tok_id in zip(
            self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids)])))
        self.start_token_map = {
            "domain": self.bodom_token,
            "intent": self.bointent_token,
            "bspn": self.bodst_token,
            "aspn": self.bosys_act_token,
            "delex_resp": self.bodelex_resp_token,
            "concrete_resp": self.boconcrete_resp_token,
        }
        self.end_token_map = {
            "domain": [self.eodom_token, self.eou_token],
            "intent": [self.eointent_token, self.eou_token],
            "bspn": [self.eodst_token, self.eou_token],
            "aspn": [self.eosys_act_token, self.eou_token],
            "delex_resp": [self.eodelex_resp_token, self.eoconcrete_resp_token, self.eou_token],
            "concrete_resp": [self.eoconcrete_resp_token, self.eodelex_resp_token, self.eou_token],
        }
        print(f"start_token_map:\n{json.dumps(self.start_token_map, indent=2)}")
        print(f"\nend_token_map:\n{json.dumps(self.end_token_map, indent=2)}")

    def _tokenize_text(self, text, return_tokens=False):
        assert getattr(self, 'tokenizer', None) is not None, "Please set tokenizer first."
        tokens = self.tokenizer.tokenize(text)
        if return_tokens:
            return tokens
        else:
            return len(tokens)

    def build_instruct_data(self):
        assert getattr(self, 'data_source', None) is not None, "Please set data_source first."
        assert len(self.train) + len(self.dev) + len(self.test) > 0, "Please load data first."
        assert not self.use_raw_utterance or getattr(self, "has_raw_utterance", False), \
            f"Not found raw user utterance in {self.data_source} dataset."
        assert not self.pseudo_intent or not getattr(self, "has_intent", False), \
            f"Cannot set pseudo_intent to True for {self.data_source} dataset with intent annotation."

        if (self.disable_intent or self.intent_dropout_ratio > 0) and not self.has_intent:
            print(utils.highlight("Please Note this dataset does not contain intent annotations.", "yellow"))
            # return False
        if self.disable_sys_act and not self.has_sys_act:
            print(utils.highlight("Please Note this dataset does not contain system action annotations.", "yellow"))
            # return False
        if not self.gen_concrete_resp and not self.has_concrete_resp:
            print(utils.highlight("Please Note this dataset does not contain golden concrete responses.", "yellow"))
        if self.disable_concrete_resp_loss and self.has_concrete_resp:
            print(utils.highlight("Please Note that the concrete response loss is disabled.", "yellow"))

        if self.pptod_strategy:
            self.data_source += "-pptod"
        if self.use_raw_utterance:
            self.data_source += "-raw_uttr"
        if self.no_task_instruction:
            self.data_source += "-no_ia"
        if self.simple_instruction:
            self.data_source += "-sim_i"
        if self.no_schema_info:
            self.data_source += "-no_sa"
        if self.training_level == "turn_level":
            self.data_source += "-turn_level"
        if not self.inform_request_schema:
            self.data_source += "-non_irs"
        if self.disable_intent and self.has_intent:
            self.data_source += "-no_intent"
        if self.intent_dropout_ratio > 0 and self.has_intent:
            self.data_source += f"-idr_{self.intent_dropout_ratio:.2f}"
        if self.detailed_db_result:
            self.data_source += "-ddb"
        if not self.gen_concrete_resp and self.has_concrete_resp:
            self.data_source += "-no_con_resp"
        if self.concrete_resp_first:
            self.data_source += "-con_resp_first"

        if self.turn_schema:
            self.data_source += "-turn_schema"

        if self.disable_sys_act and self.has_sys_act:
            self.data_source += "-no_sys_act"
        if self.num_duplication > 1:
            self.data_source += f"-dup_{self.num_duplication}"
        if self.few_shot_training != '':
            self.data_source += f"-few_shot_{self.few_shot_training}"
        if self.exp_domains and "all" not in self.exp_domains:
            self.data_source += f"-{self.exp_domains.strip().replace(' ', '_')}"

        instruct_file_dir = os.path.join(self.save_dir, self.data_source)
        os.makedirs(instruct_file_dir, exist_ok=True)

        long_dial_path = os.path.join(self.save_dir, self.data_source, "long_dial.txt")
        if os.path.exists(long_dial_path):
            os.remove(long_dial_path)

        organize_func = self._organize_pptod_data if self.pptod_strategy else self._organize_instruct_data
        self.instruct_train, train_stats = organize_func(self.train)
        self.instruct_dev, dev_stats = organize_func(self.dev)
        self.instruct_test, test_stats = organize_func(self.test)

        json.dump({"train": train_stats, "dev": dev_stats, "test": test_stats},
                  open(os.path.join(instruct_file_dir, 'stats.json'), 'w', encoding="utf-8"), indent=2)

        json.dump(self.instruct_train, open(os.path.join(instruct_file_dir, 'train.json'), 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.instruct_dev, open(os.path.join(instruct_file_dir, 'dev.json'), 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(self.instruct_test, open(os.path.join(instruct_file_dir, 'test.json'), 'w', encoding="utf-8"), ensure_ascii=False, indent=2)
        return True

    def _organize_pptod_data(self, data):
        from src.data.data_processors.plug_and_play import organize_instruct_data
        return organize_instruct_data(self, data)

    def _organize_instruct_data(self, data):
        # TODO: split too long dialogues (total_tokens > 4096),
        #  preceeding turns contains at least (4096 - len(sys_msg)) / 2 tokens
        # TODO: whether to concat too short dialogues (total_tokens < 1000) ?
        instruct_data = []
        dial_tokens = []
        dial_lengths = []
        num_dials = len(data)
        output_long_dial = True
        long_dial_path = os.path.join(self.save_dir, self.data_source, "long_dial.txt")
        short_dial_dataset = any([self.data_source.startswith(short_dn) for short_dn in ['kvret', 'camrest']])

        if len(data) >= 10000:
            num_processes = 10
            with Pool(num_processes) as pool:
                generated_data = list(tqdm(
                    pool.imap(self._generate_instruct_dial, data, chunksize=len(data) // num_processes),
                    total=len(data), desc="Generating instruct data"))
        else:
            generated_data = [self._generate_instruct_dial(dial) for dial in tqdm(
                data, desc=f"Generating instruct data for {self.data_source}")]
        assert len(generated_data) == len(data), f"{len(generated_data)}, {len(data)}"
        before_dial_ids = [dial[-1]['dial_id'] for dial in data]
        after_dial_ids = [res[0] for res in generated_data]
        equals = [a == b for a, b in zip(before_dial_ids, after_dial_ids)]
        assert before_dial_ids == after_dial_ids, f"Inconsistent happened at 'index={equals.index(False)}'"

        # for didx, dial in enumerate(tqdm(data, desc=f"Organizing instruct data for {self.data_source}")):
        #     dial_id, sys_msg, dial_body, _ = self._generate_instruct_dial(dial)
        for didx, (dial_id, sys_msg, dial_body, _) in enumerate(tqdm(
                generated_data, desc=f"Organizing instruct data for {self.data_source}")):
            # tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(concat_text, add_special_tokens=False))
            # assert len(self.tokenizer.tokenize(concat_text)) == len(tokens), f"{len(self.tokenizer.tokenize(concat_text))}, {len(tokens)}"
            turn_tokenized = [self._tokenize_text(sys_msg, return_tokens=True), [
                [self._tokenize_text(text, return_tokens=True) for text in turn[0]] for turn in dial_body]]
            turn_lengths = [len(turn_tokenized[0]), list(map(lambda tn: sum(map(len, tn)), turn_tokenized[1]))]

            short_spans, _ = utils.split_long_dial(turn_lengths, self.model_max_length, history_ratio=self.history_ratio)
            for st, sp, ed in short_spans:
                assert sp < ed, f"{sp}, {ed}"
                temp_conversations = [([sys_msg], [False])]
                for turn in dial_body[st:sp]:
                    temp_conversations.append((turn[0], [False] * len(turn[1])))
                for turn in dial_body[sp:ed]:
                    temp_conversations.append(deepcopy(turn))
                # temp_conversations[-1][0][-1] = temp_conversations[-1][0][-1].rstrip() + f"{self.eod_token}\n"

                num_tokens = turn_lengths[0] + sum(turn_lengths[1][st:ed])
                text_lengths = [turn_lengths[0], turn_lengths[1][st:sp], turn_lengths[1][sp:ed]]
                if (len(short_spans) >= 2 or short_dial_dataset) and output_long_dial:
                    separate_texts = [t for turn in temp_conversations for t in turn[0]]
                    concat_text = ''.join(separate_texts)
                    tokens = self._tokenize_text(concat_text, return_tokens=True)
                    with open(long_dial_path, 'a', encoding="utf-8") as f:
                        f.write(f"{dial_id}: {json.dumps(text_lengths)} -> {num_tokens} >|< {len(tokens)}\n")
                        f.write(f"{concat_text}\n")
                        for text in separate_texts:
                            f.write(f"{self._tokenize_text(text, return_tokens=True)}\n")
                            f.write(f"{self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(text, add_special_tokens=False))}\n\n")
                        f.write(f"{tokens}\n\n")

                instruct_data.append({
                    "id": dial_id,
                    "source": self.data_source,
                    "conversations": [{"turn_texts": t, "turn_labels": l} for t, l in temp_conversations],
                    "text_lengths": json.dumps(text_lengths),
                    "total_length": num_tokens,
                })
                dial_tokens.append(num_tokens)
                dial_lengths.append(ed - st)
            if len(short_spans) >= 2 or short_dial_dataset:
                output_long_dial = False

        stats = pd.DataFrame({"tokens": np.array(dial_tokens), 'turns': np.array(dial_lengths)})
        stats['tokens_per_turn'] = stats['tokens'] / stats['turns']
        stats_dict = stats.describe().to_dict()
        stats_dict['real_dials'] = num_dials
        stats_dict['total_tokens'] = stats_dict['tokens']['count'] * stats_dict['tokens']['mean']
        stats_dict['total_tokens (B)'] = stats_dict['total_tokens'] / 1e9
        stats_dict['train_tokens (B)'] = len(instruct_data) * self.model_max_length / 1e9
        return instruct_data, stats_dict

    def _generate_instruct_dial(self, dial):
        """
        turn_data: list of turn_data (dict)
        """
        # TODO: use json or plain text to represent the intermediate results and schema ?
        assert {'user', 'dial_id'}.issubset(dial[0].keys()), f"turn_data keys: {list(dial[0].keys())}"
        dial_id = dial[-1]['dial_id']

        # determine the order of tasks and fields
        if self.concrete_resp_first:
            task_orders = ["domain", "intent", "constraint", "sys_act", "concrete_resp", "delex_resp"]
            field_orders = ["turn_domain", "turn_intent", "bspn", "aspn", "concrete_resp", "delex_resp"]
        else:
            task_orders = ["domain", "intent", "constraint", "sys_act", "delex_resp", "concrete_resp"]
            field_orders = ["turn_domain", "turn_intent", "bspn", "aspn", "delex_resp", "concrete_resp"]
        task_orders, field_orders = zip(
            *[(task_name, field_name) for task_name, field_name in zip(task_orders, field_orders) if
              field_name in dial[0]])

        # create system message
        task_idx = 0
        sys_msg = f"{self.bou_token}{self.tod_roles['system']}\n{self.tod_system_message}\n"
        # TODO: for each session, shuffle the order of domains?
        for task_name, field_name in zip(task_orders, field_orders):
            if dial[0][field_name] is None:
                continue
            all_doms = ', '.join(self.all_domains) if task_name == "domain" else ''
            task_idx += 1
            if task_name == "domain":
                extra_content = f"{all_doms}."
            elif task_name == "constraint":
                extra_content = f"{json.dumps(self.slot_value_format)}."
            else:
                extra_content = ""
            sys_msg += f"{task_idx}. {self.tod_intermediate_prompts[task_name]}{extra_content}\n"

        if self.no_task_instruction:
            sys_msg = ""
        elif self.simple_instruction:
            rule_idx = 0
            sys_msg = f"{self.bou_token}{self.tod_roles['system']}\n{self.simple_tod_system_message}\n"
            if "domain" in task_orders:
                rule_idx += 1
                all_doms = ', '.join(self.all_domains)
                sys_msg += f"\t{roman.toRoman(rule_idx)}. {self.tod_intermediate_prompts['domain']}{all_doms}.\n"

            if rule_idx == 0:
                sys_msg = ""
            else:
                sys_msg = sys_msg.strip() + f"{self.eou_token}\n"
        else:
            sys_msg = sys_msg.strip() + f"{self.eou_token}\n"

        # manage schema and db result
        total_tokens = self._tokenize_text(sys_msg)
        max_meta_tokens = self.model_max_length - total_tokens - 500
        prev_domain_schema, prev_intent_schema = {}, {}
        prev_dbres = {}

        field2type = defaultdict(set)
        # create turn message
        dial_body = []
        for turn_idx, turn_data in enumerate(dial):
            turn_msgs = []
            turn_loss = []
            turn_msgs.append(f"{self.bou_token}{self.tod_roles['user']}\n{turn_data['user']}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
            turn_loss.append(False)
            another_task_idx = 0
            for field_name in field_orders:
                field2type[field_name].add(type(turn_data[field_name]).__name__)
                match field_name:
                    case "turn_domain":
                        # TODO: fix issue when turn_domain is not in all_domains
                        domains = [dom for dom in turn_data['turn_domain'] if dom in self.all_domains]
                        another_task_idx += 1
                        turn_msgs.append(
                            f"{another_task_idx}. {self.bodom_token}{json.dumps(domains)}{self.eodom_token}\n")
                        turn_loss.append(True)
                        # Note: Minimize the repetition of domain schema
                        domain_schema = self.get_domain_schema(domains, turn_data)
                        domain_schema = self.manage_schema(
                            domain_schema, prev_domain_schema, max_meta_tokens, inplace=True)
                        if domain_schema:
                            if self.intent_dropout_ratio > 0:
                                domain_schema = self.dropout_intent(domain_schema, turn_data)
                            if not self.no_schema_info:
                                turn_msgs.append(f"{self.textualize_schema(domain_schema)}\n")
                                turn_loss.append(False)
                    case "turn_intent":
                        another_task_idx += 1
                        turn_msgs.append(
                            f"{another_task_idx}. {self.bointent_token}{json.dumps(turn_data['turn_intent'])}{self.eointent_token}\n")
                        turn_loss.append(True)
                        # Note: Minimize the repetition of intent schema
                        intent_schema = self.get_intent_schema(turn_data['turn_intent'], turn_data)
                        intent_schema = self.manage_schema(
                            intent_schema, prev_intent_schema, max_meta_tokens, inplace=True)
                        if intent_schema and not self.no_schema_info:
                            turn_msgs.append(f"{self.textualize_schema(intent_schema, schema_type='intent')}\n")
                            turn_loss.append(False)
                    case "bspn":
                        if turn_data["bspn"] is not None:
                            another_task_idx += 1
                            turn_msgs.append(f"{another_task_idx}. {self.bodst_token}{json.dumps(turn_data['bspn'])}{self.eodst_token}\n")
                            turn_loss.append(True)
                        if "dbres" in turn_data:
                            field2type["dbres"].add(type(turn_data["dbres"]).__name__)
                            # Note: Minimize the repetition of database results
                            curr_dbres = self.manage_db_result(
                                turn_data['dbres'], prev_dbres, turn_data['turn_domain'],
                                turn_data.get("domain_mapping", None), max_meta_tokens, inplace=True)
                            db_result = self.textualize_db_result(curr_dbres)
                            if db_result:
                                turn_msgs.append(db_result)
                                turn_loss.append(False)
                    case "aspn":
                        another_task_idx += 1
                        turn_msgs.append(f"{another_task_idx}. {self.bosys_act_token}{turn_data['aspn']}{self.eosys_act_token}\n")
                        # TODO: if sys_act is empty, turn loss should be False ??? should allow annotation missing !!!
                        # turn_loss.append(True if turn_data['aspn'].strip() else False)
                        turn_loss.append(True)
                    # Note: if possible, generate both delx_resp and nodelx_resp
                    case "delex_resp":
                        another_task_idx += 1
                        turn_msgs.append(f"{another_task_idx}. {self.bodelex_resp_token}{turn_data['delex_resp']}{self.eodelex_resp_token}\n")
                        turn_loss.append(True)
                    case "concrete_resp":
                        another_task_idx += 1
                        turn_msgs.append(f"{another_task_idx}. {self.boconcrete_resp_token}{turn_data['concrete_resp']}{self.eoconcrete_resp_token}\n")
                        if self.disable_concrete_resp_loss:
                            turn_loss.append(False)
                        else:
                            turn_loss.append(True)
                    case _:
                        raise ValueError(f"field_name: {field_name}")
            turn_msgs[-1] = turn_msgs[-1].rstrip() + f"{self.eou_token}\n"

            # Accumulate the number of tokens that appear in the history
            curr_turn_tokens = sum([self._tokenize_text(text) for text in turn_msgs])
            for dom in prev_domain_schema:
                prev_domain_schema[dom] += curr_turn_tokens
            for intent in prev_intent_schema:
                prev_intent_schema[intent] += curr_turn_tokens
            for dom in prev_dbres:
                prev_dbres[dom][1] += curr_turn_tokens

            total_tokens += curr_turn_tokens
            if total_tokens > max_meta_tokens:
                max_meta_tokens = math.ceil(self.history_ratio * (self.model_max_length - self._tokenize_text(sys_msg)))

            assert len(turn_msgs) == len(turn_loss), f"turn_msgs: {len(turn_msgs)}, turn_loss: {len(turn_loss)}"
            dial_body.append((turn_msgs, turn_loss))
            assert task_idx == another_task_idx, f"task_idx: {task_idx}, another_task_idx: {another_task_idx}"
        # dial_body[-1][0][-1] = dial_body[-1][0][-1].strip()

        if self.first_sample:
            print(f"[{self.data_source}] Types of fields: {json.dumps(field2type, default=utils.set_to_list, indent=2)}")
            self.first_sample = False

        return dial_id, sys_msg, dial_body, {"prev_domain_schema": prev_domain_schema,
                                             "prev_intent_schema": prev_intent_schema, "prev_dbres": prev_dbres}

    def get_eval_data(self, split_name='test'):
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        eval_data = deepcopy(name_to_set[split_name])
        assert len(eval_data) > 0, "Please load data first."
        num_turns = 0
        num_dials = len(eval_data)
        for dial in eval_data:
            num_turns += len(dial)
        stats = {
            "num_dials": num_dials,
            "num_turns": num_turns,
        }
        return eval_data, stats

    def infer(self, split_name='test', evaluator=None, num_infer_samples=None):
        print("Start inference ...")
        assert getattr(self, 'data_source', None) is not None, "Please set data_source first."
        assert hasattr(self, 'resp4eval'), "Please set resp4eval first."
        assert self.resp4eval in ['delex_resp', 'concrete_resp', None], f"resp4eval: {self.resp4eval}"
        assert not self.use_raw_utterance or getattr(self, "has_raw_utterance", False), \
            f"Not found raw user utterance in {self.data_source} dataset."
        assert not self.pseudo_intent or not getattr(self, "has_intent", False), \
            f"Cannot set pseudo_intent to True for {self.data_source} dataset with intent annotation."

        begin_time = time.time()
        current_time = utils.get_current_time()
        begin_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        suffix = current_time.strftime("%Y-%m-%d_%H-%M-%S.%f")
        infer_samples_file_name = f"{split_name}_samples_{suffix}.json"
        infer_parameters = {
            "model_name_or_path": self.model_name_or_path,
            "max_length": self.model_max_length,
            "start_token_map": self.start_token_map,
            "end_token_map": self.end_token_map,
            "start_with_bos": self.start_with_bos,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "num_return_sequences": self.num_return_sequences,
            "num_beams": self.num_beams,
            "max_tokens": self.max_tokens,
            "tie_word_embeddings": self.tie_word_embeddings,
            "exp_domains": self.exp_domains,
            "pptod_strategy": self.pptod_strategy,
            "use_raw_utterance": self.use_raw_utterance,
            "pseudo_intent": self.pseudo_intent,
            "enable_capital": self.enable_capital,
            "num_history_turns": self.num_history_turns,
            "accumulate_belief_state": self.accumulate_belief_state,
            "use_true_prev_domain": self.use_true_prev_domain,
            "use_true_prev_intent": self.use_true_prev_intent,
            "use_true_prev_bspn": self.use_true_prev_bspn,
            "use_true_prev_aspn": self.use_true_prev_aspn,
            "use_true_prev_resp": self.use_true_prev_resp,
            "use_true_prev_concrete_resp": self.use_true_prev_concrete_resp,
            "use_true_curr_domain": self.use_true_curr_domain,
            "use_true_curr_intent": self.use_true_curr_intent,
            "use_true_curr_bspn": self.use_true_curr_bspn,
            "use_true_curr_aspn": self.use_true_curr_aspn,
            "use_true_domain_for_ctr_eval": self.use_true_domain_for_ctr_eval,
            "use_true_bspn_for_ctr_eval": self.use_true_bspn_for_ctr_eval,
            "use_true_resp_for_ctr_eval": self.use_true_resp_for_ctr_eval,
            "zero_shot_enhancement": self.zero_shot_enhancement,
            "few_shot_evaluation": getattr(self, 'few_shot_evaluation', None),
            "num_shot": getattr(self, 'num_shot', None),
            "no_task_instruction": self.no_task_instruction,
            "simple_instruction": self.simple_instruction,
            "no_schema_info": self.no_schema_info,
            "turn_schema": self.turn_schema,
            "turn_dbres": self.turn_dbres,
            "inform_request_schema": self.inform_request_schema,
            "disable_intent": self.disable_intent,
            "detailed_db_result": self.detailed_db_result,
            "disable_sys_act": self.disable_sys_act,
            "gen_concrete_resp": self.gen_concrete_resp,
            "concrete_resp_first": self.concrete_resp_first,
            "infer_backend": self.infer_backend,
            "resp4eval": self.resp4eval,
            "comments": self.comments,
            "inference_result_file": infer_samples_file_name,
            "begin_time": begin_time_str,
        }
        infer_save_dir = os.path.join(self.project_root, "infer_results", f"{self.processor_version}", self.data_source)
        os.makedirs(infer_save_dir, exist_ok=True)
        infer_metric_file = os.path.join(infer_save_dir, f"{split_name}_metrics.json")
        infer_samples_file = os.path.join(infer_save_dir, infer_samples_file_name)

        eval_data, eval_stats = self.get_eval_data(split_name)
        print(f"'{split_name}' stats:\n{json.dumps(eval_stats, indent=2)}")

        start_index, end_index = 0, len(eval_data) if num_infer_samples is None else min(len(eval_data), num_infer_samples)
        # start_index, end_index = 0, 16
        real_data = eval_data[start_index:end_index]

        raw_results = []
        if self.infer_backend == "vllm":
            # multi-gpu inference when using vllm engine
            if self.pptod_strategy:
                from src.data.data_processors.plug_and_play import inference
                raw_results = inference(self, real_data)
            else:
                raw_results = self.inference_turn_by_turn(real_data)

        # sort results by dial_id
        sorted_results = utils.sort_by_ids(
            [d[-1]['dial_id'] for d in real_data], raw_results, key_func=lambda x: x[0][-1]['dial_id'])
        del raw_results

        device2consume = defaultdict(int)
        for idx, (dialog, turn_results, elapsed_times, device_id) in enumerate(sorted_results):
            elapsed = sum(elapsed_times)
            elapsed_times = list(map(lambda x: round(x, 2), elapsed_times))
            if idx < 10:
                print(f"[{idx}] {dialog[0]['dial_id']} ({len(dialog)} turns vs. {len(turn_results)}):\n"
                      f"{json.dumps(turn_results[len(dialog) // 2], indent=2)}\n{elapsed_times} -> {elapsed:.2f}\n\n")
            if device_id is not None:
                device2consume[device_id] += elapsed

        if len(device2consume):
            for i, chunk in enumerate(chunks):
                print(f"Group {i} has {len(chunk)} dials: consume {device2consume[i]:.2f} secs; "
                      f"{list(map(len, chunk))} -> {sum(map(len, chunk))}; "
                      f"{device2consume[i] / sum(map(len, chunk)):.2f} secs per turn.")

        print(f"Inference elapsed time: {time.time() - begin_time:.3f} seconds")

        num_eval_dial, num_eval_turn = 0, 0
        num_legal_dom, num_legal_int, num_legal_bspn = 0, 0, 0
        dom_corr, dom_total = 0, 0
        # for idx, dialog in enumerate(eval_data[start_index:end_index]):
        inference_results = []
        for idx, (dialog, turn_results, elapsed_times, _) in enumerate(sorted_results):
            inference_results.append(dialog)
            dial_id = dialog[0]['dial_id']

            for turn_idx, (turn, turn_res) in enumerate(zip(dialog, turn_results)):
                if 'turn_domain' in turn:
                    turn['dspn'] = turn['turn_domain']
                    turn['real_dspn_gen'] = turn_res['domain_gen']
                    dspn_gen, legal_dom = self.parse_domain(
                        turn_res['domain_gen'], None if turn_idx == 0 else dialog[turn_idx - 1]['dspn_gen'])
                    turn['dspn_gen'] = turn['dspn'] if self.use_true_curr_domain else dspn_gen
                    num_legal_dom += int(legal_dom)
                    dom_corr += int(set([e.lower() for e in dspn_gen]) == set([e.lower() for e in turn['dspn']]))
                    dom_total += 1
                if 'turn_intent' in turn:
                    turn['ispn'] = turn['turn_intent']
                    turn['real_ispn_gen'] = turn_res['intent_gen']
                    ispn_gen, legal_int = self.parse_intent(turn_res['intent_gen'], None)
                    turn['ispn_gen'] = turn['ispn'] if self.use_true_curr_intent else ispn_gen
                    num_legal_int += int(legal_int)
                if 'bspn' in turn:
                    turn['real_bspn_gen'] = turn_res['bspn_gen']
                    bspn_gen, legal_bspn = self.parse_bspn(
                        turn_res['bspn_gen'], None if turn_idx == 0 else dialog[turn_idx - 1]['bspn_gen'])
                    turn['bspn_gen'] = turn['bspn'] if self.use_true_curr_bspn else bspn_gen
                    num_legal_bspn += int(legal_bspn)
                if 'dbres' in turn:
                    turn['db_return'] = turn_res['db_return']
                if 'aspn' in turn:
                    turn['real_aspn_gen'] = turn_res['aspn_gen']
                    turn['aspn_gen'] = turn['aspn'] if self.use_true_curr_aspn else turn_res['aspn_gen']
                if 'delex_resp' in turn:
                    turn['delex_resp_gen'] = turn_res['delex_resp_gen']
                if 'concrete_resp' in turn:
                    turn['concrete_resp_gen'] = turn_res['concrete_resp_gen']
                if self.resp4eval is not None:
                    turn['resp'] = turn[self.resp4eval]
                    turn['resp_gen'] = turn['resp'] if self.use_true_resp_for_ctr_eval else turn_res[f"{self.resp4eval}_gen"]
            num_eval_dial += 1
            num_eval_turn += len(dialog)
        # post-processing for evaluation
        results, _ = self.wrap_result_lm(inference_results)

        # save inference results before evaluation
        with open(infer_samples_file, "w", encoding="utf-8") as fo:
            json.dump(results, fo, indent=2, ensure_ascii=False)
        print(f"Save inference results to {infer_samples_file}")

        # start evaluation
        legal_nums = ', '.join(map(str, [num_legal_dom, num_legal_int, num_legal_bspn]))
        message = f"[{num_eval_dial}-{num_eval_turn}] [{dom_total} ({legal_nums})] "
        metric_results = {}
        if dom_total > 0:
            metric_results["dom_acc"] = dom_corr / dom_total * 100
            print(f"Raw Domain Accuracy: {metric_results['dom_acc']:.2f}")
        if evaluator is not None:
            metric_results.update(evaluator.validation_metric(results, infer_samples_file=infer_samples_file))
            message += ", ".join(
                [f"{mn}: {mv:.2f}" for mn, mv in metric_results.items() if isinstance(mv, (float, int))]) + "; "

        message += f"TIME-{time.time() - begin_time:.3f} seconds"
        end_time_str = utils.get_current_time().strftime("%Y-%m-%d %H:%M:%S.%f")
        infer_parameters['end_time'] = end_time_str

        metric_results['data_source'] = f"{self.data_source}: {split_name}"
        metric_results['num_dials'] = num_eval_dial
        metric_results['num_turns'] = num_eval_turn
        metric_results['num_legal_dom'] = num_legal_dom
        metric_results['num_legal_int'] = num_legal_int
        metric_results['num_legal_bspn'] = num_legal_bspn
        metric_results['message'] = message
        metric_results['infer_parameters'] = infer_parameters
        # save metrics
        if os.path.exists(infer_metric_file):
            utils.atomic_json_update(infer_metric_file, metric_results)
        else:
            json.dump([metric_results], open(infer_metric_file, "w", encoding="utf-8"), indent=2)
        # assert isinstance(infer_metrics, list), f"{type(infer_metrics)}" + json.dumps(infer_metrics, indent=2)
        # infer_metrics.append(metric_results)
        # json.dump(infer_metrics, open(infer_metric_file, "w", encoding="utf-8"), indent=2)
        print(f"Save inference metrics to {infer_metric_file}")

        ineqs = [i for i, (c1, c2) in enumerate(zip(self.model_name_or_path, os.getcwd())) if c1 != c2]
        print(f"Evaluated Model: {self.model_name_or_path[ineqs[0]:]}\n{message}")
        # save inference samples after evaluation
        with open(infer_samples_file, "w", encoding="utf-8") as fo:
            json.dump(results, fo, indent=2, ensure_ascii=False)
        print(f"Save inference samples to {infer_samples_file}")

        # group inference files by model
        groupby_model_file = os.path.join(infer_save_dir, f"{split_name}_groupby_model.json")
        if os.path.exists(groupby_model_file):
            groupby_model = json.load(open(groupby_model_file, "r", encoding="utf-8"))
        else:
            groupby_model = {}
        assert isinstance(groupby_model, dict), f"{type(groupby_model)}" + json.dumps(groupby_model, indent=2)
        if self.model_name_or_path not in groupby_model:
            groupby_model[self.model_name_or_path] = {}
        if str(num_eval_dial) not in groupby_model[self.model_name_or_path]:
            groupby_model[self.model_name_or_path][str(num_eval_dial)] = []
        groupby_model[self.model_name_or_path][str(num_eval_dial)].append(infer_samples_file)
        json.dump(groupby_model, open(groupby_model_file, "w", encoding="utf-8"), indent=2)

        return infer_samples_file

    @staticmethod
    def parse_domain(domain_text, prev_domain=None, legal_type=list):
        parse_success = False
        try:
            domain = json.loads(domain_text)
            if isinstance(domain, legal_type):
                parse_success = True
            else:
                domain = prev_domain
        except:
            domain = prev_domain
        if not isinstance(domain, legal_type):
            domain = legal_type()
        return domain, parse_success

    @staticmethod
    def parse_intent(intent_text, prev_intent=None, legal_type=dict):
        return GenProcessor.parse_bspn(intent_text, prev_intent, legal_type)

    @staticmethod
    def parse_bspn(bspn_text, prev_bspn=None, legal_type=dict):
        parse_success = False
        try:
            bspn = json.loads(bspn_text)
            if isinstance(bspn, legal_type):
                parse_success = True
            else:
                bspn = prev_bspn
        except:
            bspn = prev_bspn
        if not isinstance(bspn, legal_type):
            bspn = legal_type()
        return bspn, parse_success

    def map_bspn(self, bspn, enable_capital=False):
        fuzzy_mapping = getattr(self, 'fuzzy_domain_mapping', None)
        if fuzzy_mapping is None:
            return bspn

        if isinstance(bspn, str):
            bspn_dict, parse_success = self.parse_bspn(bspn, legal_type=dict)
            if not parse_success:
                return bspn
        else:
            bspn_dict = bspn

        mapped_bspn = {}
        lower_domains = [dom.lower() for dom in self.all_domains]
        for dom, sv in bspn_dict.items():
            mapped_dom = dom.strip()
            if dom.lower().strip() not in lower_domains:
                cand_doms = utils.fuzzy_match_domain(fuzzy_mapping, dom)
                if cand_doms:
                    if len(cand_doms) > 1:
                        print(f"NOTE: Find multiple candidate domains '{cand_doms}' for '{dom}'")
                    mapped_dom = cand_doms[0]
            if enable_capital:
                mapped_dom = mapped_dom.capitalize()
            mapped_bspn[mapped_dom] = sv
        return mapped_bspn

    def is_in_domain(self, domain, domain_mapping=None):
        lower_domains = [dom.lower() for dom in self.all_domains]
        lower_dom = domain.lower().strip()
        if not lower_dom:
            return None
        dom_idx = -1
        for i, cand in enumerate(lower_domains):
            if lower_dom.startswith(cand) or cand.startswith(lower_dom):
                dom_idx = i
                break
        else:
            fuzzy_mapping = getattr(self, 'fuzzy_domain_mapping', None)
            cand_doms = utils.fuzzy_match_domain(fuzzy_mapping, lower_dom)
            if cand_doms:
                dom_idx = lower_domains.index(cand_doms[0].lower())

        # dom_idx = lower_domains.index(lower_dom) if lower_dom in lower_domains else -1
        if domain_mapping:
            lower_dom_map = {k.lower(): v for k, v in domain_mapping.items()}
        if dom_idx >= 0:
            if domain_mapping is None:
                return self.all_domains[dom_idx]
            elif lower_domains[dom_idx] in lower_dom_map:
                return lower_dom_map[lower_domains[dom_idx]]
        elif domain_mapping:
            for dn, sn in domain_mapping.items():
                if lower_dom == sn.lower():
                    return sn
        return None

    def get_domain_schema(self, domains, turn_data):
        domain_schema = {}
        for dom in domains:
            dom = self.is_in_domain(dom, turn_data.get("domain_mapping", None))
            if dom is None or not self.schema[dom]:
                continue
            domain_schema[dom] = self.schema[dom]
        return domain_schema

    def get_intent_schema(self, intent_dict, turn_data):
        if not hasattr(self, 'intent_schema'):
            return {}
        intent_schema = {}
        for domain, intents in intent_dict.items():
            service = self.is_in_domain(domain, turn_data.get("domain_mapping", None))
            if service is None:
                continue
            if isinstance(intents, str):
                intents = [intents]
            for intent in intents:
                lower_schema = {k.lower(): v for k, v in self.intent_schema[service].items()}
                if intent.lower() in lower_schema:
                    intent_schema[f"{service} -> {intent}"] = lower_schema[intent.lower()]
        return intent_schema

    def manage_schema(self, schema, prev_schema, max_meta_tokens, inplace=False):
        domain_schema = {}
        for dom in schema:
            if self.turn_schema or dom not in prev_schema or prev_schema[dom] > max_meta_tokens:
                if inplace:
                    prev_schema[dom] = 0
                domain_schema[dom] = schema[dom]
        return domain_schema

    def manage_db_result(
            self, dbres, prev_dbres, curr_domain, domain_mapping, max_meta_tokens, inplace=False, inference=False):
        # TODO: if db_hidden and inference is True, hidden db results in previous turns (inplace=True) and
        #  present latest results (inplace=False)
        curr_domain = [self.is_in_domain(dom, domain_mapping) for dom in curr_domain
                       if self.is_in_domain(dom, domain_mapping)]
        curr_dbres = {}
        for dom, dom_dbres in dbres.items():
            prev_hash = prev_dbres[dom][0]["hash_value"] if dom in prev_dbres else None
            dom_hash = dom_dbres["hash_value"]
            if self.turn_dbres or dom not in prev_dbres or prev_hash[0] != dom_hash[0] or prev_hash[1] < dom_hash[1]:
                if inplace:
                    prev_dbres[dom] = [dom_dbres, 0]
                if dom in curr_domain:
                    curr_dbres[dom] = dom_dbres["content"]
            elif prev_dbres[dom][1] > max_meta_tokens:
                if inplace:
                    prev_dbres[dom][1] = 0
                if dom in curr_domain:
                    curr_dbres[dom] = (dom_dbres["content"][:1] + [prev_dbres[dom][0]["content"][1]]
                                       + dom_dbres["content"][2:])
            elif dom in curr_domain:
                summary = "all information is same as the database results of the previous turns" \
                    if dom_dbres["content"][1] else ""
                curr_dbres[dom] = dom_dbres["content"][:1] + [summary] + dom_dbres["content"][2:]
        return curr_dbres

    # @staticmethod
    def textualize_schema(self, schema, schema_type="domain"):
        type2prefix = {"domain": "Domain", "intent": "Intent"}
        schema_str = ""
        for dom in schema:
            dom_show = dom.capitalize() if self.enable_capital else dom
            schema_str += f"\n    {dom_show}: {json.dumps(schema[dom])}"
        if schema_str:
            schema_str = f"{type2prefix[schema_type]} Schema:{schema_str}"
        return schema_str

    @staticmethod
    def summarize_db_result(match_results, informable_slots, delex_resp=None, concrete_resp=None,
                            enable_multi_options=False, max_display=2, max_options=3, extra_info=None):
        # computing md5 hash for match_results
        def _get_md5_hash(entity_list):
            return utils.get_md5_hash(json.dumps(sorted([json.dumps(_) for _ in entity_list])))

        max_display = min(max_display, len(match_results))
        if not match_results:
            return "", [_get_md5_hash(match_results), 0]

        summary = ""
        if enable_multi_options:
            stats = defaultdict(list)
            for item in match_results:
                for slot in informable_slots:
                    if slot in item:
                        stats[slot].append(item[slot])
            key2entropy = utils.count_and_entropy(stats)
            texts = []
            # TODO: control the number of slots to display
            for slot, stat in sorted(key2entropy.items(), key=lambda x: x[1][1], reverse=True):
                if len(stat[0]) <= 1:
                    continue
                texts.append(f"{len(stat[0])} distinct [{slot}] " + ' '.join(
                    [f"{v} ({c} items)" for v, c in sorted(stat[0].items(), key=lambda x: x[1], reverse=True)[:max_options]]))
            if texts:
                summary = f"statistics for attributes with multiple options - " + ' ; '.join(texts)
            else:
                summary = ""

        counter = [0] * len(match_results)
        if concrete_resp and isinstance(concrete_resp, str):
            concrete_resp = concrete_resp.lower()
            for idx, item in enumerate(match_results):
                for slot, value in item.items():
                    slot = slot.lower()
                    if slot == "arriveby":
                        slot = "arrive"
                    elif slot == "leaveat":
                        slot = "leave"
                    if isinstance(value, str) and value.strip():
                        value = value.strip().lower()
                    else:
                        continue
                    if ((delex_resp is None or f"[value_{slot}]" in delex_resp.lower()) and
                            f" {value}" in f" {concrete_resp}"):
                        counter[idx] += 1
            match_results, counter = zip(*sorted(zip(match_results, counter), key=lambda x: x[1], reverse=True))
        info_mass = utils.count_and_entropy({'occur': counter})['occur'][-1]

        if extra_info:
            match_results[0].update(extra_info)
        md5_hash = _get_md5_hash(match_results)

        items_str = '\n        '.join([f"[{idx + 1}] {json.dumps(item)}" for idx, item in
                                       enumerate(match_results[:max_display])])
        summary += f"\n        {items_str}"

        return summary, [md5_hash, info_mass]

    def textualize_db_result(self, dbres):
        # TODO: collect detailed db results, e.g., value count for each slot
        extra_result_idx = 1 if self.detailed_db_result else 2
        empty = True
        db_result = f"Results from Database:\n"
        for dom, res in dbres.items():
            dom_show = dom.capitalize() if self.enable_capital else dom
            empty = False
            db_result += f"    {dom_show}: {str(res[0]) + ' matched entities' if isinstance(res[0], int) else res[0]}"
            for r in res[extra_result_idx:]:
                if r:
                    db_result += f", {r}"
            db_result += '\n'
        if empty:
            db_result = ''
        return db_result

    def get_vllm_sampling_params(self):
        # NOTE: for up-to-date vLLM,
        #  if beam search is required, use BeamSearchParams instead of SamplingParams
        # use_beam_search = self.num_beams > 1
        sampling_params = {
            "n": self.num_return_sequences,
            "repetition_penalty": 1,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "skip_special_tokens": False,
        }
        return sampling_params

    def load_vllm_engine(self):
        cur_time = time.time()
        # start vllm engine
        print(utils.highlight(f"'{torch.cuda.device_count()}' Visible Cuda Devices: "
                              f"{os.environ.get('CUDA_VISIBLE_DEVICES', '')}", 'yellow'))
        print(f'VLLM Worker Multiprocessing Method: {os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")}')

        # update tie_word_embeddings in config.json
        if self.tie_word_embeddings is not None:
            config_file = os.path.join(self.model_name_or_path, "config.json")
            model_config = json.load(open(config_file, "r"))
            if "tie_word_embeddings" in model_config:
                model_config["tie_word_embeddings"] = self.tie_word_embeddings
                json.dump(model_config, open(config_file, "w"), indent=2)
                print(f"Update 'tie_word_embeddings' to '{self.tie_word_embeddings}' in {config_file}")

        vllm_engine = LLM(model=self.model_name_or_path, tensor_parallel_size=torch.cuda.device_count())
        # tokenizer = vllm_engine.get_tokenizer()
        if self.llama:
            # tokenizer = LlamaTokenizer.from_pretrained(self.model_name_or_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=False)
        else:
            tokenizer = vllm_engine.get_tokenizer()

        sampling_params = self.get_vllm_sampling_params()
        print(f"SAMPLING PARAMS: {json.dumps(sampling_params, indent=2)}")
        print(f"Tokenizer: {tokenizer}")
        print(f"Finish loading vllm engine, consume {time.time() - cur_time:.2f} seconds.", flush=True)
        return vllm_engine, tokenizer, sampling_params


    def inference_turn_by_turn(self, dials):
        start_time = time.time()
        vllm_engine, tokenizer, sampling_params = self.load_vllm_engine()
        cur_time = time.time()

        # task flags
        domain = None
        if "turn_domain" in dials[0][0]:
            domain = ', '.join(self.all_domains)
            # domain = ', '.join(sorted(self.all_domains))
            # domain = ', '.join(dom.capitalize() for dom in self.all_domains)
            if self.enable_capital:
                domain = ', '.join(dom.capitalize() for dom in sorted(self.all_domains))
        intent = "turn_intent" in dials[0][0]
        constraint = "bspn" in dials[0][0]
        db_res = "dbres" in dials[0][0]
        sys_act = "aspn" in dials[0][0]
        delex_resp = "delex_resp" in dials[0][0]
        concrete_resp = "concrete_resp" in dials[0][0]
        task_name2field_name = {
            "domain": "turn_domain",
            "intent": "turn_intent",
            "constraint": "bspn",
            "sys_act": "aspn",
            "delex_resp": "delex_resp",
            "concrete_resp": "concrete_resp",
        }
        # determine the order of tasks and fields
        if self.concrete_resp_first:
            task_orders = ["domain", "intent", "constraint", "sys_act", "concrete_resp", "delex_resp"]
        else:
            task_orders = ["domain", "intent", "constraint", "sys_act", "delex_resp", "concrete_resp"]
        for task_name in copy.deepcopy(task_orders):
            if not locals()[task_name]:
                task_orders.remove(task_name)
        print(f"Task orders: {task_orders}")

        # system message
        task_id = 0
        system_lm = f"{self.bou_token}{self.tod_roles['system']}\n{self.tod_system_message}"
        for task_name in task_orders:
            field_content = dials[0][0][task_name2field_name[task_name]]
            assert field_content is not None, f"{task_name}: {type(field_content)}"
            task_id += 1
            if task_name == "domain":
                extra_content = f"{domain}."
            elif task_name == "constraint":
                extra_content = f"{json.dumps(self.slot_value_format)}."
            elif task_name == "sys_act" and self.extra_rules4sys_act:
                extra_content = f" {self.extra_rules4sys_act}"
                print(f"Extra rules for sys_act: {extra_content}")
            else:
                extra_content = ""
            system_lm += f"\n{task_id}. {self.tod_intermediate_prompts[task_name]}{extra_content}"
        if self.no_task_instruction:
            system_lm = ""
        elif self.simple_instruction:
            rule_idx = 0
            system_lm = f"{self.bou_token}{self.tod_roles['system']}\n{self.simple_tod_system_message}"
            if "domain" in task_orders:
                rule_idx += 1
                system_lm += f"\n\t{roman.toRoman(rule_idx)}. {self.tod_intermediate_prompts['domain']}{domain}."
            if "sys_act" in task_orders and self.extra_rules4sys_act:
                rule_idx += 1
                system_lm += f"\n\t{roman.toRoman(rule_idx)}. Rules for system action: {self.extra_rules4sys_act}"

            if rule_idx == 0:
                system_lm = ""
            else:
                system_lm += f"{self.eou_token}\n"
        else:
            system_lm += f"{self.eou_token}\n"
        system_lm_token_ids = tokenizer.encode(system_lm, add_special_tokens=False)

        allow_max_token = self.model_max_length - 1000
        # max_token_limit = math.ceil(max_token_limit * 2 / 3)

        dial_meta = {}
        for d in dials:
            new_system_lm = system_lm + d[0].get("extra_prompt", "")
            new_system_lm_token_ids = tokenizer.encode(new_system_lm, add_special_tokens=False)
            dial_meta[d[-1]["dial_id"]] = {
                "turns": d,
                "prompt": new_system_lm,
                "prompt_token_ids": new_system_lm_token_ids,
                "turn_prompt": "",
                "turn_prompt_token_ids": [],
                "turn_text": [],
                "prev_turn_text": [],
                "history_turns": [new_system_lm],
                "turn_lengths": [len(new_system_lm_token_ids)],
                "history_turns_token_ids": [copy.deepcopy(new_system_lm_token_ids)],
                "cur_context_length": len(new_system_lm_token_ids),
                "max_token_limit": allow_max_token,
                "prev_domain_schema": d[0].get("prev_domain_schema", {}),
                "prev_intent_schema": d[0].get("prev_intent_schema", {}),
                "prev_dbres": d[0].get("prev_dbres", {}),
            }

        print(f"Finish preparing dialogue meta, consume {time.time() - cur_time:.2f} seconds.", flush=True)
        cur_time = time.time()

        format_fn = lambda x: json.dumps(x)
        def filter_domain(domain):
            parsed_domains, _ = self.parse_domain(domain)
            filtered_domains = []
            for dom in parsed_domains:
                real_dom = self.is_in_domain(dom, turn_data.get("domain_mapping", None))
                if real_dom is None:
                    continue
                # filtered_domains.append(dom)
                filtered_domains.append(real_dom.capitalize() if self.enable_capital else real_dom)
            return filtered_domains

        def get_domain_schema(domain, turn_data, use_true_curr_domain):
            gold_domain = turn_data['turn_domain']
            # print(f"domain: {domain}, gold_domain: {gold_domain}")
            if use_true_curr_domain:
                domain_schema = self.get_domain_schema(gold_domain, turn_data)
            else:
                parsed_domain, _ = self.parse_domain(domain)
                domain_schema = self.get_domain_schema(parsed_domain, turn_data)
            return domain_schema

        def get_intent_schema(intent, turn_data, use_true_curr_intent, domain_gen, use_true_domain):
            gold_intent = turn_data['turn_intent']
            # print(f"intent: {intent}, gold_intent: {gold_intent}")
            if use_true_curr_intent:
                if not use_true_domain and domain_gen is not None:
                    # Note: filter gold_intent by generated domain even if use_true_curr_intent
                    domains = get_domain_schema(domain_gen, turn_data, use_true_domain).keys()
                    gold_intent = {k: v for k, v in gold_intent.items() if k in domains}
                intent_schema = self.get_intent_schema(gold_intent, turn_data)
            else:
                parsed_intent, _ = self.parse_intent(intent)
                intent_schema = self.get_intent_schema(parsed_intent, turn_data)
            return intent_schema, gold_intent

        def get_dbres(bspn, turn_data, use_true_curr_bspn):
            gold_bspn = turn_data['bspn']
            gold_dbres = turn_data['dbres']
            # print(f"bspn: {bspn}, gold_bspn: {gold_bspn}")
            if use_true_curr_bspn:
                dbres = gold_dbres
            else:
                parsed_bspn, _ = self.parse_bspn(bspn)
                parsed_bspn = self.map_bspn(parsed_bspn)
                dbres = self.constraint_to_DBpointer(parsed_bspn, turn_data['turn_domain'], turn_data)
            return dbres

        def record_info(task, elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens):
            elapsed_dict[f"{task}"] = gen_elapsed
            elapsed_dict[f"{task}_toks"] = sum(num_gen_tokens)
            elapsed_dict['prompt_toks'] += sum(num_prompt_tokens)
            min_max_gen_len[f"{task}_min"].append(min(num_gen_tokens))
            min_max_gen_len[f"{task}_max"].append(max(num_gen_tokens))

        def tokenize_text_list(texts):
            return list(chain(*[tokenizer.encode(text, add_special_tokens=False) for text in texts])), ''.join(texts)

        max_turn_num = max([len(d) for d in dials])
        elapseds = []
        detailed_elapseds = []
        min_max_gen_len = defaultdict(list)
        conv_results = defaultdict(list)
        turn_idx = 0
        while len(dial_meta) > 0:
            # maintain valid dials
            for dial_id in tqdm(list(dial_meta.keys()), desc=f"Turn {turn_idx} Maintain Valid Dials:"):
                if len(dial_meta[dial_id]["turns"]) <= turn_idx:
                    del dial_meta[dial_id]
            if len(dial_meta) == 0:
                break
            # manage context
            for dial_id, v in tqdm(dial_meta.items(), desc=f"Turn {turn_idx} Context Management:"):
                dial_meta[dial_id] = self.manage_context(dial_id, turn_idx, v, dial_meta[dial_id]["max_token_limit"])
            # turn start
            for dial_id, v in tqdm(dial_meta.items(), desc=f"Turn {turn_idx} Turn Start:"):
                v["turn_text"] = [f"{self.bou_token}{self.tod_roles['user']}\n{v['turns'][turn_idx]['user']}"
                                  f"{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n"]
                v["turn_prompt"] = v["turn_text"][0]
                v["turn_prompt_token_ids"] = tokenizer.encode(v["turn_prompt"], add_special_tokens=False)
            # generate turn results
            dial_ids = list(dial_meta.keys())
            elapsed_dict = {'turn_idx': turn_idx, 'count': len(dial_ids), 'prompt_toks': 0}
            task2gens = defaultdict(list)
            task_id = 0

            if db_res and turn_idx > 0 and self.turn_dbres:
                self.swap_prev_turn_text(dial_meta, tokenizer)

            # TODO: when generated domain, intent, belief state are illegal type, what should we do?
            for task_name in task_orders:
                match task_name:
                    case "domain":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['domain']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "domain", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["domain"], **sampling_params), vllm_engine, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} domain Post-Processing:"):
                            task2gens["domain_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            # dom_use = format_fn(turn_data['turn_domain']) if self.use_true_curr_domain else gen_result
                            dom_use = format_fn(
                                turn_data['turn_domain'] if self.use_true_curr_domain else filter_domain(gen_result)
                            )
                            strings = [f"{start_str}{dom_use}{self.end_token_map['domain'][0]}\n"]
                            # Note: Minimize the repetition of domain schema
                            domain_schema = get_domain_schema(gen_result, turn_data, self.use_true_curr_domain)
                            domain_schema = self.manage_schema(
                                domain_schema, dial_meta[dial_id]["prev_domain_schema"],
                                dial_meta[dial_id]["max_token_limit"])
                            if domain_schema and not self.no_schema_info:
                                strings += [f"{self.textualize_schema(domain_schema)}\n"]
                            string_ids, string = tokenize_text_list(strings)
                            dial_meta[dial_id]["turn_prompt"] += string
                            dial_meta[dial_id]["turn_prompt_token_ids"] += string_ids

                            # dom_use = format_fn(turn_data['turn_domain']) if self.use_true_prev_domain else gen_result
                            dom_use = format_fn(
                                turn_data['turn_domain'] if self.use_true_prev_domain else filter_domain(gen_result)
                            )
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{dom_use}{self.end_token_map['domain'][0]}\n"]
                            # Note: Minimize the repetition of domain schema
                            domain_schema = get_domain_schema(gen_result, turn_data, self.use_true_prev_domain)
                            domain_schema = self.manage_schema(
                                domain_schema, dial_meta[dial_id]["prev_domain_schema"],
                                dial_meta[dial_id]["max_token_limit"], inplace=True)
                            if domain_schema and not self.no_schema_info:
                            # if not self.turn_schema and domain_schema and not self.no_schema_info:
                                dial_meta[dial_id]["turn_text"] += [f"{self.textualize_schema(domain_schema)}\n"]
                        assert len(task2gens["domain_gen"]) == len(dial_ids), f"{len(task2gens['domain_gen'])} != {len(dial_ids)}"
                        record_info(
                            "dom", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                    case "intent":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['intent']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "intent", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["intent"], **sampling_params), vllm_engine, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} intent Post-Processing:"):
                            didx = dial_ids.index(dial_id)
                            task2gens["intent_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            intent_schema, gold_intent = get_intent_schema(
                                gen_result, turn_data, self.use_true_curr_intent,
                                task2gens["domain_gen"][didx] if "domain_gen" in task2gens else None,
                                self.use_true_curr_domain
                            )
                            intent_use = format_fn(gold_intent) if self.use_true_curr_intent else gen_result
                            strings = [f"{start_str}{intent_use}{self.end_token_map['intent'][0]}\n"]
                            # Note: Minimize the repetition of intent schema
                            intent_schema = self.manage_schema(
                                intent_schema, dial_meta[dial_id]["prev_intent_schema"],
                                dial_meta[dial_id]["max_token_limit"])
                            if intent_schema and not self.no_schema_info:
                                strings += [f"{self.textualize_schema(intent_schema, schema_type='intent')}\n"]
                            string_ids, string = tokenize_text_list(strings)
                            dial_meta[dial_id]["turn_prompt"] += string
                            dial_meta[dial_id]["turn_prompt_token_ids"] += string_ids

                            intent_schema, gold_intent = get_intent_schema(
                                gen_result, turn_data, self.use_true_prev_intent,
                                task2gens["domain_gen"][didx] if "domain_gen" in task2gens else None,
                                self.use_true_prev_domain
                            )
                            intent_use = format_fn(gold_intent) if self.use_true_prev_intent else gen_result
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{intent_use}{self.end_token_map['intent'][0]}\n"]
                            # Note: Minimize the repetition of intent schema
                            intent_schema = self.manage_schema(
                                intent_schema, dial_meta[dial_id]["prev_intent_schema"],
                                dial_meta[dial_id]["max_token_limit"], inplace=True)
                            if not self.turn_schema and intent_schema and not self.no_schema_info:
                                dial_meta[dial_id]["turn_text"] += [f"{self.textualize_schema(intent_schema, schema_type='intent')}\n"]
                        assert len(task2gens["intent_gen"]) == len(dial_ids), f"{len(task2gens['intent_gen'])} != {len(dial_ids)}"
                        record_info(
                            "int", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                    case "constraint":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['bspn']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "bspn", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["bspn"], **sampling_params), vllm_engine, tokenizer)

                        if db_res and turn_idx > 0 and self.turn_dbres:
                            self.swap_prev_turn_text(dial_meta, tokenizer)

                        # Note: remove repeat tokens
                        self.deduplicate_output(gen_results, num_gen_tokens, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} bspn Post-Processing:"):
                            task2gens["bspn_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]

                            # gen_result = self.map_bspn(gen_result, enable_capital=self.enable_capital)
                            # if isinstance(gen_result, dict):
                            #     gen_result = format_fn(gen_result)

                            bspn_use = format_fn(turn_data['bspn']) if self.use_true_curr_bspn else gen_result
                            text = f"{start_str}{bspn_use}{self.end_token_map['bspn'][0]}\n"
                            dial_meta[dial_id]["turn_prompt"] += text
                            dial_meta[dial_id]["turn_prompt_token_ids"] += tokenizer.encode(text, add_special_tokens=False)

                            bspn_use = format_fn(turn_data['bspn']) if self.use_true_prev_bspn else gen_result
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{bspn_use}{self.end_token_map['bspn'][0]}\n"]
                        assert len(task2gens["bspn_gen"]) == len(dial_ids), f"{len(task2gens['bspn_gen'])} != {len(dial_ids)}"
                        record_info(
                            "bspn", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)

                        if db_res:
                            for idx, dial_id in tqdm(enumerate(dial_ids), desc=f"Turn {turn_idx} dbres Post-Processing:"):
                                turn_data = dial_meta[dial_id]["turns"][turn_idx]
                                dbres = get_dbres(task2gens["bspn_gen"][idx], turn_data, self.use_true_curr_bspn)
                                curr_dbres = self.manage_db_result(
                                    dbres, dial_meta[dial_id]["prev_dbres"], turn_data['turn_domain'],
                                    turn_data.get("domain_mapping", None), dial_meta[dial_id]["max_token_limit"])
                                db_return = self.textualize_db_result(curr_dbres).strip()
                                task2gens["db_return"].append(db_return)
                                if db_return:
                                    text = f"{db_return}\n"
                                    dial_meta[dial_id]["turn_prompt"] += text
                                    dial_meta[dial_id]["turn_prompt_token_ids"] += tokenizer.encode(text, add_special_tokens=False)

                                dbres = get_dbres(task2gens["bspn_gen"][idx], turn_data, self.use_true_prev_bspn)
                                curr_dbres = self.manage_db_result(
                                    dbres, dial_meta[dial_id]["prev_dbres"], turn_data['turn_domain'],
                                    turn_data.get("domain_mapping", None), dial_meta[dial_id]["max_token_limit"], inplace=True)
                                db_use = self.textualize_db_result(curr_dbres).strip()
                                if not self.turn_dbres and db_use:
                                    dial_meta[dial_id]["turn_text"] += [f"{db_use}\n"]
                    case "sys_act":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['aspn']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "aspn", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["aspn"], **sampling_params), vllm_engine, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} aspn Post-Processing:"):
                            task2gens["aspn_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            aspn_use = turn_data['aspn'] if self.use_true_curr_aspn else gen_result
                            text = f"{start_str}{aspn_use}{self.end_token_map['aspn'][0]}\n"
                            dial_meta[dial_id]["turn_prompt"] += text
                            dial_meta[dial_id]["turn_prompt_token_ids"] += tokenizer.encode(text, add_special_tokens=False)

                            aspn_use = turn_data['aspn'] if self.use_true_prev_aspn else gen_result
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{aspn_use}{self.end_token_map['aspn'][0]}\n"]
                        assert len(task2gens["aspn_gen"]) == len(dial_ids), f"{len(task2gens['aspn_gen'])} != {len(dial_ids)}"
                        record_info(
                            "aspn", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                    case "delex_resp":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['delex_resp']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "delex_resp", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["delex_resp"], **sampling_params), vllm_engine, tokenizer)
                        self.filter_repeat_slot(gen_results, num_gen_tokens, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} delex_resp Post-Processing:"):
                            task2gens["delex_resp_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            text = f"{start_str}{gen_result}{self.end_token_map['delex_resp'][0]}\n"
                            dial_meta[dial_id]["turn_prompt"] += text
                            dial_meta[dial_id]["turn_prompt_token_ids"] += tokenizer.encode(text, add_special_tokens=False)

                            delex_resp_use = turn_data['delex_resp'] if self.use_true_prev_resp else gen_result
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{delex_resp_use}{self.end_token_map['delex_resp'][0]}\n"]
                        assert len(task2gens["delex_resp_gen"]) == len(dial_ids), f"{len(task2gens['delex_resp_gen'])} != {len(dial_ids)}"
                        record_info(
                            "dlx", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                    case "concrete_resp":
                        task_id += 1
                        start_str = f"{task_id}. {self.start_token_map['concrete_resp']}"
                        gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                            turn_idx, dial_meta, dial_ids, "concrete_resp", start_str if self.start_with_bos else "",
                            SamplingParams(stop=self.end_token_map["concrete_resp"], **sampling_params), vllm_engine, tokenizer)
                        # Note: remove repeat tokens
                        self.deduplicate_output(gen_results, num_gen_tokens, tokenizer)
                        for dial_id, gen_result in tqdm(zip(dial_ids, gen_results), desc=f"Turn {turn_idx} concrete_resp Post-Processing:"):
                            task2gens["concrete_resp_gen"].append(gen_result)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            text = f"{start_str}{gen_result}{self.end_token_map['concrete_resp'][0]}\n"
                            dial_meta[dial_id]["turn_prompt"] += text
                            dial_meta[dial_id]["turn_prompt_token_ids"] += tokenizer.encode(text, add_special_tokens=False)

                            # TODO: if the dataset has no concrete response, use generated concrete response
                            concrete_resp_use = turn_data['concrete_resp'] if self.use_true_prev_concrete_resp and self.has_concrete_resp else gen_result
                            dial_meta[dial_id]["turn_text"] += [f"{start_str}{concrete_resp_use}{self.end_token_map['concrete_resp'][0]}\n"]
                        assert len(task2gens["concrete_resp_gen"]) == len(dial_ids), f"{len(task2gens['concrete_resp_gen'])} != {len(dial_ids)}"
                        record_info(
                            "con", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)

            # transform task2gens (dict[list]) to list[dict]
            turn_results = []
            for i in range(len(dial_ids)):
                turn_results.append({task_name: task2gens[task_name][i] for task_name in task2gens})

            # update context
            dst_flag = self.start_token_map['bspn']
            for idx, dial_id in tqdm(enumerate(dial_ids), desc=f"Turn {turn_idx} Update Context:"):
                dial_meta[dial_id]["turn_text"][-1] = dial_meta[dial_id]["turn_text"][-1].rstrip() + f"{self.eou_token}\n"

                dial_meta[dial_id]["prev_turn_text"] = deepcopy(dial_meta[dial_id]["turn_text"])
                db_return = task2gens["db_return"][idx] if "db_return" in task2gens else ""
                if db_res and self.turn_dbres and db_return:
                    dst_idx = next((i for i, txt in enumerate(dial_meta[dial_id]["turn_text"]) if dst_flag in txt), None)
                    dial_meta[dial_id]["prev_turn_text"].insert(dst_idx + 1, f"{db_return}\n")

                turn_token_ids, turn_text = tokenize_text_list(dial_meta[dial_id]["turn_text"])
                dial_meta[dial_id]["prompt"] += turn_text
                dial_meta[dial_id]["prompt_token_ids"] += turn_token_ids
                dial_meta[dial_id]["history_turns"].append(turn_text)
                dial_meta[dial_id]["turn_lengths"].append(len(turn_token_ids))
                dial_meta[dial_id]["history_turns_token_ids"].append(turn_token_ids)
                dial_meta[dial_id]["cur_context_length"] += len(turn_token_ids)
                turn_lengths = dial_meta[dial_id]["turn_lengths"]
                if self.num_history_turns is not None and len(turn_lengths) > self.num_history_turns:
                    split_idx = len(turn_lengths) - self.num_history_turns
                    temp_len = turn_lengths[0] + sum(turn_lengths[split_idx:])
                    dial_meta[dial_id]["max_token_limit"] = min(temp_len, allow_max_token)
                for dom in dial_meta[dial_id]["prev_domain_schema"]:
                    dial_meta[dial_id]["prev_domain_schema"][dom] += len(turn_token_ids)
                for intent in dial_meta[dial_id]["prev_intent_schema"]:
                    dial_meta[dial_id]["prev_intent_schema"][intent] += len(turn_token_ids)
                for dom in dial_meta[dial_id]["prev_dbres"]:
                    dial_meta[dial_id]["prev_dbres"][dom][1] += len(turn_token_ids)
                # update conv_results
                turn_data = dial_meta[dial_id]["turns"][turn_idx]
                turn_result = {
                    "turn_num": turn_data['turn_num'],
                    "user": turn_data['user'],
                    "domain": format_fn(turn_data.get('turn_domain', None)),
                    "intent": format_fn(turn_data.get('turn_intent', None)),
                    'bspn': format_fn(turn_data.get('bspn', None)),
                    "aspn": turn_data.get('aspn', None),
                    "delex_resp": turn_data.get('delex_resp', None),
                    "concrete_resp": turn_data.get('concrete_resp', None),
                }
                turn_result.update(turn_results[idx])
                conv_results[dial_id].append(turn_result)

            elapseds.append(time.time() - cur_time)
            elapsed_dict["total"] = elapseds[-1]
            elapsed_dict = {k: v / 1000 if k.endswith('_toks') else v for k, v in elapsed_dict.items()}
            detailed_elapseds.append(elapsed_dict)
            print(f"Finish turn {turn_idx} inference, consume {elapseds[-1]:.2f} seconds.\n", flush=True)
            cur_time = time.time()
            turn_idx += 1
            if turn_idx == max_turn_num:
                print(json.dumps(dial_meta[dial_ids[0]]))
                print(f"The above is the dial_meta for {dial_ids[0]}")
                with open("dial_meta.json", "w", encoding="utf-8") as fout:
                    small_dial_meta = {did: dm for did, dm in list(dial_meta.items())[:3]}
                    json.dump(small_dial_meta, fout, indent=2)
            elapsed_df = pd.DataFrame(detailed_elapseds).set_index("turn_idx")
            elapsed_df.loc['SUM'] = elapsed_df.sum()
            task_columns = [col for col in elapsed_df.columns if
                            col not in {'count', 'total'} and not col.endswith('_toks')]
            task_tok_columns = [col for col in elapsed_df.columns if col.endswith('_toks') and col != 'prompt_toks']
            print(task_columns)
            print(task_tok_columns)
            elapsed_df['vllm'] = elapsed_df[task_columns].sum(axis=1)
            elapsed_df['prompt_toks'] = elapsed_df.pop('prompt_toks')
            elapsed_df['gen_toks'] = elapsed_df[task_tok_columns].sum(axis=1)
            elapsed_df['fwd_v'] = elapsed_df['prompt_toks'] / elapsed_df['vllm']
            elapsed_df['gen_v'] = elapsed_df['gen_toks'] / elapsed_df['vllm']
            # elapsed_df['avg_vllm'] = elapsed_df['vllm'] / elapsed_df['count']
            print(f"Detailed Elapsed (sec) and tokens (K):\n", elapsed_df.round(2), end="\n\n", flush=True)
            print(f"Min Max Gen Len:\n", pd.DataFrame(min_max_gen_len), end="\n\n", flush=True)

        print(f"Time elapsed (sec): {[round(ela, 2) for ela in elapseds]} -> {sum(elapseds):.2f}", flush=True)
        # mess up final results
        sorted_results = [(d, conv_results[d[0]['dial_id']], elapseds[:len(d)], None) for d in dials]
        print(f"Finish inference for {len(dials)} dials, consume {time.time() - start_time:.2f} seconds.", flush=True)
        return sorted_results

    def filter_repeat_slot(self, gen_results, num_gen_tokens, tokenizer, max_allow_repeat=3):
        for i, (gen_text, gen_toks) in enumerate(tqdm(zip(gen_results, num_gen_tokens), desc="Filter Repeat Slot:")):
            slot2count = Counter(re.findall(r"\[value_\w+\]", gen_text))
            repeat_slots = [slot for slot, count in slot2count.items() if count > max_allow_repeat]
            if len(repeat_slots) == 0:
                continue
            cur_text = gen_text
            for repeat_text in repeat_slots:
                spans = cur_text.split(repeat_text)
                new_spans = []
                left = 0
                repeat_count = 0
                for idx, span in enumerate(spans):
                    if span == spans[left]:
                        repeat_count += 1
                    else:
                        repeat_count = 1
                        left = idx
                    if repeat_count < max_allow_repeat:
                        new_spans.append(span)
                cur_text = repeat_text.join(new_spans)
            new_gen_toks = len(tokenizer.encode(cur_text, add_special_tokens=False))
            if gen_toks > self.max_tokens - 20:
                gen_results[i] = cur_text
                num_gen_tokens[i] = new_gen_toks
            else:
                print(utils.highlight("No Operation, if ", "yellow"), end="", flush=True)
            print(utils.highlight(f"Filter Repeat Slot, {gen_toks} -> {new_gen_toks}", "yellow") +
                  f":\n{gen_text}\n" + utils.highlight(f"-> {cur_text}\n", "yellow"), flush=True)

    def deduplicate_output(self, gen_results, num_gen_tokens, tokenizer):
        for i, (gen_text, gen_toks) in enumerate(tqdm(zip(gen_results, num_gen_tokens), desc="Deduplicate Generated Outputs:")):
            if gen_toks > self.max_tokens - 20:
                new_output = utils.deduplicate_string(gen_text)
                new_gen_toks = len(tokenizer.encode(new_output, add_special_tokens=False))
                gen_results[i] = new_output
                num_gen_tokens[i] = new_gen_toks
                print(utils.highlight(f"Deduplicate Generated Outputs, {gen_toks} -> {new_gen_toks}", "yellow") +
                  f":\n{gen_text}\n" + utils.highlight(f"-> {new_output}\n", "yellow"), flush=True)

    def swap_prev_turn_text(self, dial_meta, tokenizer):
        turn_start_str = f"{self.bou_token}{self.tod_roles['user']}\n"
        ptk = "prev_turn_text"
        for dial_id, v in dial_meta.items():
            sep_idx = v["prompt"].rfind(turn_start_str)
            old_prev_turn = v["prompt"][sep_idx:]
            prev_turn = ''.join(v[ptk]) if isinstance(v[ptk], list) else v[ptk]
            v[ptk] = old_prev_turn
            v["prompt"] = v["prompt"][:sep_idx] + prev_turn
            v["prompt_token_ids"] = tokenizer.encode(v["prompt"], add_special_tokens=False)

    def gen_and_decode(self, turn_idx, dial_meta, dial_ids, task_name, start_str,
                       sampling_params, vllm_engine, tokenizer):
        prompts, prompt_token_ids = [], []
        num_prompt_tokens = []
        for dial_id in tqdm(dial_ids, desc=f"Turn {turn_idx} {task_name} Gen Preparaion:"):
            prompts.append(dial_meta[dial_id]["prompt"] + dial_meta[dial_id]["turn_prompt"] + start_str)
            prompt_token_ids.append(
                dial_meta[dial_id]["prompt_token_ids"] + dial_meta[dial_id]["turn_prompt_token_ids"] +
                tokenizer.encode(start_str, add_special_tokens=False))
            num_prompt_tokens.append(len(prompt_token_ids[-1]))
        gen_results = []
        num_gen_tokens = []
        gen_start_time = time.time()
        # Note: detect illegal special tokens in the generated text
        status2color = {0: None, 1: "red", 2: "yellow"}
        start_sp_token = self.start_token_map[task_name]
        legal_token_ids = [] if start_sp_token in start_str else tokenizer.convert_tokens_to_ids([start_sp_token])
        legal_token_ids += tokenizer.convert_tokens_to_ids(sampling_params.stop)
        illegal_token_ids = list(set(range(tokenizer.vocab_size, len(tokenizer))) - set(legal_token_ids))
        for idx, result in enumerate(vllm_engine.generate(prompts, sampling_params, prompt_token_ids, use_tqdm=True)):
            completions = [[output.cumulative_logprob, output.text] for output in result.outputs]

            gen_token_ids = np.array(result.outputs[0].token_ids)
            illegal_flags = np.isin(gen_token_ids, illegal_token_ids)
            if illegal_flags.any():
                legal_flags = np.isin(gen_token_ids, legal_token_ids)
                colors = [status2color[_] for _ in illegal_flags + 2 * legal_flags]
                print(utils.highlight(f"Turn {turn_idx} {task_name} Gen Illegal Tokens", "yellow") +
                      f": {''.join(map(utils.highlight, tokenizer.convert_ids_to_tokens(gen_token_ids), colors))}",
                      end="\n\n", flush=True)

            if idx < 3 or len(gen_token_ids) >= self.max_tokens:
                if len(gen_token_ids) >= self.max_tokens:
                    print(utils.highlight("Exceed Max Tokens", "yellow") + f": {result.outputs[0].token_ids}", flush=True)
                print(f"[{dial_ids[idx]:<10}] [{turn_idx}] {task_name}:", completions, flush=True)

            gen_comp = completions[0][1]
            if illegal_flags.any():
                # Note: truncate from the first illegal token
                gen_token_ids = gen_token_ids[:np.argmax(illegal_flags)]
                gen_comp = tokenizer.decode(gen_token_ids, skip_special_tokens=False)
                print(utils.highlight(f"Turn {turn_idx} {task_name} Gen Truncated", "yellow") + f": {gen_comp}", flush=True)

            gen_results.append(gen_comp.split(start_sp_token)[-1].strip())
            num_gen_tokens.append(len(gen_token_ids))

        return gen_results, num_prompt_tokens, num_gen_tokens, time.time() - gen_start_time

    @staticmethod
    def manage_context(dial_id, turn_idx, meta_data, max_token_limit):
        if meta_data["cur_context_length"] <= max_token_limit or turn_idx == 0:
            return meta_data

        print(utils.highlight(f"{dial_id} Turn_idx {turn_idx}", "yellow") +
              f" - cur_context_length: {meta_data['cur_context_length']} > max_token_limit: {max_token_limit}")
        turn_lengths = meta_data["turn_lengths"]
        history_window = bisect.bisect_right(np.cumsum(turn_lengths[1:][::-1]), max_token_limit - turn_lengths[0])
        split_idx = len(turn_lengths) - history_window
        history_length = sum(turn_lengths[split_idx:])

        length_str = (f"[{', '.join(map(str, turn_lengths[:split_idx]))}, "
                      f"{', '.join(map(partial(utils.highlight, color='yellow'), turn_lengths[split_idx:]))}]")
        print(f"turn_lengths:   {length_str} -> {sum(turn_lengths)} "
              f"-> {turn_lengths[0]} + " + utils.highlight(
            f"{history_length} (history_window={history_window})", "yellow"))
        # assert history_window > 0, f"history_window: {history_window}"
        meta_data["cur_context_length"] = turn_lengths[0] + history_length
        meta_data["prompt"] = meta_data["history_turns"][0] + ''.join(meta_data["history_turns"][split_idx:])
        history_token_ids = sum(meta_data["history_turns_token_ids"][split_idx:], [])
        meta_data["prompt_token_ids"] = meta_data["history_turns_token_ids"][0] + history_token_ids
        return meta_data

