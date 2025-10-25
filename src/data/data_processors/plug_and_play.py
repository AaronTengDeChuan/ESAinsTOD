import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass

from src.utils import utils


def organize_instruct_data(self, data):
    instruct_data = []
    dial_tokens = []
    dial_lengths = []
    sample_tokens = []
    num_dials = len(data)

    if len(data) >= 10000:
        num_processes = 10
        with Pool(num_processes) as pool:
            partial_worker = partial(generate_instruct_sample, self)
            generated_data = list(tqdm(
                pool.imap(partial_worker, data, chunksize=len(data) // num_processes),
                total=len(data), desc="Generating instruct data"))
    else:
        generated_data = [generate_instruct_sample(self, dial) for dial in tqdm(
            data, desc=f"Generating instruct data for {self.data_source}")]
    assert len(generated_data) == len(data), f"{len(generated_data)}, {len(data)}"
    before_dial_ids = [dial[-1]['dial_id'] for dial in data]
    after_dial_ids = [res[0] for res in generated_data]
    equals = [a == b for a, b in zip(before_dial_ids, after_dial_ids)]
    assert before_dial_ids == after_dial_ids, f"Inconsistent happened at 'index={equals.index(False)}'"

    for didx, (dial_id, samples) in enumerate(tqdm(
            generated_data, desc=f"Organizing instruct data for {self.data_source}")):
        dial_token = 0
        for sample_id, instruct, output in samples:
            sample_lengths = [self._tokenize_text(text, return_tokens=False) for text in [instruct, output]]
            num_tokens = sum(sample_lengths)
            instruct_data.append({
                "id": f"{dial_id}-{sample_id}",
                "source": self.data_source,
                "conversations": [{"turn_texts": [instruct, output], "turn_labels": [False, True]}],
                "text_lengths": json.dumps(sample_lengths),
                "total_length": num_tokens,
            })
            sample_tokens.append(num_tokens)
            dial_token += num_tokens
        dial_tokens.append(dial_token)
        dial_lengths.append(len(data[didx]))

    sample_stats = pd.DataFrame({"sample_tokens": np.array(sample_tokens)})
    stats = pd.DataFrame({"tokens": np.array(dial_tokens), 'turns': np.array(dial_lengths)})
    stats['tokens_per_turn'] = stats['tokens'] / stats['turns']
    stats_dict = stats.describe().to_dict()
    stats_dict["sample_tokens"] = sample_stats.describe().to_dict()
    stats_dict['real_dials'] = num_dials
    stats_dict['total_tokens'] = stats_dict['tokens']['count'] * stats_dict['tokens']['mean']
    stats_dict['total_tokens (B)'] = stats_dict['total_tokens'] / 1e9
    stats_dict['train_tokens (B)'] = len(instruct_data) * self.model_max_length / 1e9
    return instruct_data, stats_dict


def get_task_list(self, dial):
    task_orders = ["domain", "intent", "constraint", "sys_act", "delex_resp", "concrete_resp"]
    field_orders = ["turn_domain", "turn_intent", "bspn", "aspn", "delex_resp", "concrete_resp"]
    if self.no_schema_info:
        del task_orders[0]
        del field_orders[0]
    task_orders, field_orders = zip(
        *[(task_name, field_name) for task_name, field_name in zip(task_orders, field_orders) if
          field_name in dial[0]])
    task_orders, field_orders = list(task_orders), list(field_orders)

    if "delex_resp" in task_orders:
        resp_field = "delex_resp"
        if "concrete_resp" in task_orders:
            task_orders.remove("concrete_resp")
            field_orders.remove("concrete_resp")
    else:
        resp_field = "concrete_resp"
        if self.disable_concrete_resp_loss:
            task_orders.remove("concrete_resp")
            field_orders.remove("concrete_resp")

    return task_orders, field_orders, resp_field


def get_schema_str(self, schema, schema_type="domain"):
    if isinstance(schema, list):
        temp = {}
        for sch in schema:
            temp.update(sch)
        schema = temp
    schema_str = self.textualize_schema(schema, schema_type=schema_type)
    if schema_str:
        schema_str = f"\n{schema_str}\n"
    return schema_str


def generate_instruct_sample(self, dial):
    """
        turn_data: list of turn_data (dict)
        PPTOD:
            input_contain_db control whether includes db result as part of the input when generating dialog act or response
            ref_db: if input contain db, whether using the reference db result
            ref_bs, ref_act, ref_db = False, False, False # pptod only consider e2e evaluation
            but with groud-truth dialog history consisting of previous user and system messages
        """
    assert {'user', 'dial_id'}.issubset(dial[0].keys()), f"turn_data keys: {list(dial[0].keys())}"
    dial_id = dial[-1]['dial_id']

    task_orders, field_orders, resp_field = get_task_list(self, dial)

    all_doms = ', '.join(self.all_domains)
    dom_cands = f"\nPossible domains include: {all_doms}.\n"

    field2type = defaultdict(set)
    # list of (sample_id, instruct, output)
    instruct_data = []
    previous_context = []
    curr_domain_schema, accm_domain_schema = {}, {}
    curr_intent_schema, accm_intent_schema = {}, {}
    for turn_idx, turn_data in enumerate(dial):
        curr_user_input = f"{self.bousr_token}{turn_data['user']}{self.eousr_token}"
        previous_context.append(curr_user_input)
        context_str = ''.join(previous_context)

        db_input = ''
        if "dbres" in turn_data:
            field2type["dbres"].add(type(turn_data["dbres"]).__name__)
            curr_dbres = self.manage_db_result(
                turn_data['dbres'], {}, turn_data['turn_domain'],
                turn_data.get("domain_mapping", None), None, inplace=False)
            db_input = self.textualize_db_result(curr_dbres)

        for task_name, field_name in zip(task_orders, field_orders):
            field2type[field_name].add(type(turn_data[field_name]).__name__)
            sample_id = f"{turn_idx}:{task_name}"
            instruct, output = None, None
            match field_name:
                case "turn_domain":
                    domains = [dom for dom in turn_data['turn_domain'] if dom in self.all_domains]
                    instruct = f"{self.pptod_prompts[task_name]}{dom_cands}{context_str}"
                    output = f"{self.bodom_token}{json.dumps(domains)}{self.eodom_token}"
                    curr_domain_schema = self.get_domain_schema(domains, turn_data)
                    accm_domain_schema.update(curr_domain_schema)
                case "turn_intent":
                    schema_str = get_schema_str(self, curr_domain_schema)
                    instruct = f"{self.pptod_prompts[task_name]}{schema_str}{context_str}"
                    output = f"{self.bointent_token}{json.dumps(turn_data['turn_intent'])}{self.eointent_token}"
                    if "domain" in task_orders:
                        curr_intent_schema = self.get_intent_schema(turn_data['turn_intent'], turn_data)
                        accm_intent_schema.update(curr_intent_schema)
                case "bspn":
                    if turn_data["bspn"] is not None:
                        dom_schema_str = get_schema_str(self, accm_domain_schema)
                        intent_schema_str = get_schema_str(self, accm_intent_schema, schema_type="intent")
                        instruct = f"{self.pptod_prompts[task_name]}{dom_schema_str}{intent_schema_str}{context_str}"
                        output = f"{self.bodst_token}{json.dumps(turn_data['bspn'])}{self.eodst_token}"
                case "aspn":
                    instruct = f"{self.pptod_prompts[task_name]}{context_str}\n{db_input}".strip()
                    output = f"{self.bosys_act_token}{turn_data['aspn']}{self.eosys_act_token}"
                case resp_field:
                    dom_schema_str = get_schema_str(self, curr_domain_schema)
                    intent_schema_str = get_schema_str(self, curr_intent_schema, schema_type="intent")
                    instruct = f"{self.pptod_prompts['delex_resp']}{dom_schema_str}{intent_schema_str}{context_str}\n{db_input}".strip()
                    output = f"{self.bodelex_resp_token}{turn_data[resp_field]}{self.eodelex_resp_token}"

            if instruct:
                instruct_data.append((
                    sample_id,
                    f"{self.bou_token}{self.tod_roles['user']}\n{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n",
                    f"{output}{self.eou_token}"
                ))

        curr_sys_resp = f"{self.boresp_token}{turn_data[resp_field]}{self.eoresp_token}"
        previous_context.append(curr_sys_resp)

    if self.first_sample:
        print(f"[{self.data_source}] Types of fields: {json.dumps(field2type, default=utils.set_to_list, indent=2)}")
        self.first_sample = False

    return dial_id, instruct_data


def inference(self, dials):
    start_time = time.time()
    vllm_engine, tokenizer, sampling_params = self.load_vllm_engine()
    cur_time = time.time()

    domain = None
    if "turn_domain" in dials[0][0]:
        domain = ', '.join(self.all_domains)
        # domain = ', '.join(sorted(self.all_domains))
        # domain = ', '.join(dom.capitalize() for dom in self.all_domains)
        if self.enable_capital:
            domain = ', '.join(dom.capitalize() for dom in sorted(self.all_domains))
        dom_cands = f"\nPossible domains include: {domain}.\n"

    dial_meta = {}
    for d in dials:
        dial_meta[d[-1]["dial_id"]] = {
            "turns": d,
            # system prompt
            "prompt": "",
            "prompt_token_ids": [],
            # task instruct
            "turn_prompt": "",
            "turn_prompt_token_ids": [],
            # conversation history
            "turn_text": "",
            "curr_domain_schema": {},
            "accm_domain_schema": {},
            "temp_domain_schema": {},
            "curr_intent_schema": {},
            "accm_intent_schema": {},
            "temp_intent_schema": {}
        }

    print(f"Finish preparing dialogue meta, consume {time.time() - cur_time:.2f} seconds.", flush=True)
    cur_time = time.time()
    task_orders, field_orders, resp_field = get_task_list(self, dials[0])
    db_res = "dbres" in dials[0][0]
    concrete_resp = "concrete_resp" in dials[0][0]

    format_fn = lambda x: json.dumps(x)

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
        # turn start
        for dial_id, v in tqdm(dial_meta.items(), desc=f"Turn {turn_idx} Turn Start:"):
            curr_user_input = f"{self.bousr_token}{v['turns'][turn_idx]['user']}{self.eousr_token}"
            v["turn_text"] += curr_user_input
        # generate turn results
        dial_ids = list(dial_meta.keys())
        elapsed_dict = {'turn_idx': turn_idx, 'count': len(dial_ids), 'prompt_toks': 0}
        task2gens = defaultdict(list)
        for dial_id in dial_ids:
            turn_data = dial_meta[dial_id]["turns"][turn_idx]
            if "domain" not in task_orders:
                task2gens["domain_gen"].append(format_fn(turn_data['turn_domain']))
            if concrete_resp:
                task2gens["concrete_resp_gen"].append('')

        for task_name in task_orders:
            match task_name:
                case "domain":
                    # construct instruct for domain
                    for dial_id in dial_ids:
                        instruct = f"{self.pptod_prompts[task_name]}{dom_cands}{dial_meta[dial_id]['turn_text']}"
                        dial_meta[dial_id]["turn_prompt"] = \
                            (f"{self.bou_token}{self.tod_roles['user']}\n"
                             f"{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
                        dial_meta[dial_id]["turn_prompt_token_ids"] = tokenizer.encode(
                            dial_meta[dial_id]["turn_prompt"], add_special_tokens=False)
                    # generate domain
                    start_str = f"{self.start_token_map['domain']}"
                    gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                        turn_idx, dial_meta, dial_ids, "domain", start_str if self.start_with_bos else "",
                        SamplingParams(stop=self.end_token_map["domain"], **sampling_params), vllm_engine, tokenizer)
                    # record generated domain
                    for dial_id, gen_result in tqdm(zip(dial_ids, gen_results),
                                                    desc=f"Turn {turn_idx} intent Post-Processing:"):
                        task2gens["domain_gen"].append(gen_result)
                        turn_data = dial_meta[dial_id]["turns"][turn_idx]
                        domain_schema = get_domain_schema(gen_result, turn_data, self.use_true_curr_domain)
                        dial_meta[dial_id]["curr_domain_schema"] = domain_schema
                        domain_schema = get_domain_schema(gen_result, turn_data, self.use_true_prev_domain)
                        dial_meta[dial_id]["temp_domain_schema"] = domain_schema

                    assert len(task2gens["domain_gen"]) == len(dial_ids), \
                        f"{len(task2gens['domain_gen'])} != {len(dial_ids)}"
                    record_info("dom", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                case "intent":
                    # construct instruct for intent
                    for dial_id in dial_ids:
                        schema_str = get_schema_str(self, dial_meta[dial_id]["curr_domain_schema"])
                        instruct = f"{self.pptod_prompts[task_name]}{schema_str}{dial_meta[dial_id]['turn_text']}"
                        dial_meta[dial_id]["turn_prompt"] = \
                            (f"{self.bou_token}{self.tod_roles['user']}\n"
                             f"{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
                        dial_meta[dial_id]["turn_prompt_token_ids"] = tokenizer.encode(
                            dial_meta[dial_id]["turn_prompt"], add_special_tokens=False)
                    # generate intent
                    start_str = f"{self.start_token_map['intent']}"
                    gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                        turn_idx, dial_meta, dial_ids, "intent", start_str if self.start_with_bos else "",
                        SamplingParams(stop=self.end_token_map["intent"], **sampling_params), vllm_engine, tokenizer)
                    # record generated intent
                    for dial_id, gen_result in tqdm(zip(dial_ids, gen_results),
                                                    desc=f"Turn {turn_idx} intent Post-Processing:"):
                        task2gens["intent_gen"].append(gen_result)
                        if "domain" in task_orders:
                            didx = dial_ids.index(dial_id)
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            intent_schema, gold_intent = get_intent_schema(
                                gen_result, turn_data, self.use_true_curr_intent,
                                task2gens["domain_gen"][didx] if "domain_gen" in task2gens else None,
                                self.use_true_curr_domain
                            )
                            dial_meta[dial_id]["curr_intent_schema"]= intent_schema
                            intent_schema, gold_intent = get_intent_schema(
                                gen_result, turn_data, self.use_true_prev_intent,
                                task2gens["domain_gen"][didx] if "domain_gen" in task2gens else None,
                                self.use_true_prev_domain
                            )
                            dial_meta[dial_id]["temp_intent_schema"] = intent_schema
                    assert len(task2gens["intent_gen"]) == len(dial_ids), \
                        f"{len(task2gens['intent_gen'])} != {len(dial_ids)}"
                    record_info("int", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                case "constraint":
                    # construct instruct for state tracking
                    for dial_id in dial_ids:
                        dom_schema_str = get_schema_str(self, [
                            dial_meta[dial_id]["accm_domain_schema"], dial_meta[dial_id]["curr_domain_schema"]])
                        intent_schema_str = get_schema_str(self, [
                            dial_meta[dial_id]["accm_intent_schema"], dial_meta[dial_id]["curr_intent_schema"]], schema_type="intent")
                        instruct = (f"{self.pptod_prompts[task_name]}"
                                    f"{dom_schema_str}{intent_schema_str}{dial_meta[dial_id]['turn_text']}")
                        dial_meta[dial_id]["turn_prompt"] = \
                            (f"{self.bou_token}{self.tod_roles['user']}\n"
                             f"{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
                        dial_meta[dial_id]["turn_prompt_token_ids"] = tokenizer.encode(
                            dial_meta[dial_id]["turn_prompt"], add_special_tokens=False)
                    # generate belief state
                    start_str = f"{self.start_token_map['bspn']}"
                    gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                        turn_idx, dial_meta, dial_ids, "bspn", start_str if self.start_with_bos else "",
                        SamplingParams(stop=self.end_token_map["bspn"], **sampling_params), vllm_engine, tokenizer)
                    # record generated state
                    for dial_id, gen_result in tqdm(zip(dial_ids, gen_results),
                                                    desc=f"Turn {turn_idx} bspn Post-Processing:"):
                        task2gens["bspn_gen"].append(gen_result)
                    assert len(task2gens["bspn_gen"]) == len(dial_ids), \
                        f"{len(task2gens['bspn_gen'])} != {len(dial_ids)}"
                    record_info("bspn", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)

                    if db_res:
                        for idx, dial_id in tqdm(enumerate(dial_ids), desc=f"Turn {turn_idx} dbres Post-Processing:"):
                            turn_data = dial_meta[dial_id]["turns"][turn_idx]
                            dbres = get_dbres(task2gens["bspn_gen"][idx], turn_data, self.use_true_curr_bspn)
                            curr_dbres = self.manage_db_result(
                                dbres, {}, turn_data['turn_domain'],
                                turn_data.get("domain_mapping", None), None, inplace=False)
                            db_return = self.textualize_db_result(curr_dbres).strip()
                            task2gens["db_return"].append(db_return)
                case "sys_act":
                    # construct instruct for system action
                    for idx, dial_id in enumerate(dial_ids):
                        instruct = (f"{self.pptod_prompts[task_name]}{dial_meta[dial_id]['turn_text']}\n"
                                    f"{task2gens['db_return'][idx]}").strip()
                        dial_meta[dial_id]["turn_prompt"] = \
                            (f"{self.bou_token}{self.tod_roles['user']}\n"
                             f"{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
                        dial_meta[dial_id]["turn_prompt_token_ids"] = tokenizer.encode(
                            dial_meta[dial_id]["turn_prompt"], add_special_tokens=False)
                    # generate system action
                    start_str = f"{self.start_token_map['aspn']}"
                    gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                        turn_idx, dial_meta, dial_ids, "aspn", start_str if self.start_with_bos else "",
                        SamplingParams(stop=self.end_token_map["aspn"], **sampling_params), vllm_engine, tokenizer)
                    # record generated system action
                    for dial_id, gen_result in tqdm(zip(dial_ids, gen_results),
                                                    desc=f"Turn {turn_idx} aspn Post-Processing:"):
                        task2gens["aspn_gen"].append(gen_result)
                    assert len(task2gens["aspn_gen"]) == len(dial_ids), \
                        f"{len(task2gens['aspn_gen'])} != {len(dial_ids)}"
                    record_info("aspn", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
                case resp_field:
                    # construct instruct for system response
                    for idx, dial_id in enumerate(dial_ids):
                        dom_schema_str = get_schema_str(self, dial_meta[dial_id]["curr_domain_schema"])
                        intent_schema_str = get_schema_str(self, dial_meta[dial_id]["curr_intent_schema"], schema_type="intent")
                        instruct = (f"{self.pptod_prompts[task_name]}"
                                    f"{dom_schema_str}{intent_schema_str}{dial_meta[dial_id]['turn_text']}\n"
                                    f"{task2gens['db_return'][idx]}").strip()
                        dial_meta[dial_id]["turn_prompt"] = \
                            (f"{self.bou_token}{self.tod_roles['user']}\n"
                             f"{instruct}{self.eou_token}\n{self.bou_token}{self.tod_roles['assistant']}\n")
                        dial_meta[dial_id]["turn_prompt_token_ids"] = tokenizer.encode(
                            dial_meta[dial_id]["turn_prompt"], add_special_tokens=False)
                    # generate system response
                    start_str = f"{self.start_token_map['delex_resp']}"
                    gen_results, num_prompt_tokens, num_gen_tokens, gen_elapsed = self.gen_and_decode(
                        turn_idx, dial_meta, dial_ids, "delex_resp", start_str if self.start_with_bos else "",
                        SamplingParams(stop=self.end_token_map["delex_resp"], **sampling_params), vllm_engine,
                        tokenizer)
                    self.filter_repeat_slot(gen_results, num_gen_tokens, tokenizer)
                    # record generated system response
                    for dial_id, gen_result in tqdm(zip(dial_ids, gen_results),
                                                    desc=f"Turn {turn_idx} delex_resp Post-Processing:"):
                        task2gens["delex_resp_gen"].append(gen_result)
                    assert len(task2gens["delex_resp_gen"]) == len(dial_ids), \
                        f"{len(task2gens['delex_resp_gen'])} != {len(dial_ids)}"
                    record_info("dlx", elapsed_dict, min_max_gen_len, gen_elapsed, num_prompt_tokens, num_gen_tokens)
        # transform task2gens (dict[list]) to list[dict]
        turn_results = []
        for i in range(len(dial_ids)):
            turn_results.append({task_name: task2gens[task_name][i] for task_name in task2gens})

        # update context
        for idx, dial_id in tqdm(enumerate(dial_ids), desc=f"Turn {turn_idx} Update Context:"):
            meta_dict = dial_meta[dial_id]
            meta_dict["accm_domain_schema"].update(meta_dict["temp_domain_schema"])
            meta_dict["accm_intent_schema"].update(meta_dict["temp_intent_schema"])
            turn_data = meta_dict['turns'][turn_idx]
            resp_use = turn_data[resp_field] if self.use_true_prev_resp else task2gens["delex_resp_gen"][idx]
            curr_sys_resp = f"{self.boresp_token}{resp_use}{self.eoresp_token}"
            meta_dict["turn_text"] += curr_sys_resp
            # update conv_results
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
        if turn_idx == max_turn_num - 2:
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