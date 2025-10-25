# coding: utf-8

import json
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter, defaultdict


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class SLUEvaluator(object):
    def __init__(self, reader):
        self.reader = reader
        self.all_domains = deepcopy(self.reader.all_domains)
        self.all_intents = self.reader.all_intents
        self.informable_slots = {domain: ds["informable slots"] for domain, ds in self.reader.schema.items()}
        # self.informable_slots = {}
        # for domain, intents in self.reader.intent_schema.items():
        #     for intent, intent_info in intents.items():
        #         self.informable_slots[f"{intent}"] = intent_info["required_slots"]
    def validation_metric(self, results, infer_samples_file=None):
        # icorr, itotal, stp, sfp, sfn, ocorr
        domain2stats = {domain: [0] * 6 for domain in ["ALL"] + self.all_domains}
        unknown_domains = []
        unknown_intents = []
        unknown_slots = defaultdict(list)

        def accumulate_stats(target, keys, index, value):
            for key in keys:
                target[key][index] += value

        for result in results:
            gold_domain = result[-1]['dspn'][0]
            if isinstance(result[-1]['ispn'], list):
                gold_intent = result[-1]['ispn'][0]
            else:
                gold_intent = result[-1]['ispn'][gold_domain][0]
            gold_spans = result[-1]['bspn']

            pred_domain = result[-1]['dspn_gen'][0] if result[-1]['dspn_gen'] else "NONE"
            ispn_gen = result[-1]['ispn_gen']
            if isinstance(ispn_gen, list):
                pred_intent = ispn_gen[0] if ispn_gen else "NONE"
            else:
                ispn_gen = ispn_gen.get(pred_domain, [])
                pred_intent = ispn_gen[0] if ispn_gen and isinstance(ispn_gen, list) else "NONE"
            pred_spans = result[-1]['bspn_gen']

            if pred_domain not in self.all_domains:
                unknown_domains.append(pred_domain)
            if pred_intent not in self.all_intents:
                unknown_intents.append(pred_intent)
            for ps, _ in pred_spans.items():
                if ps not in self.informable_slots.get(pred_intent, []):
                    unknown_slots[pred_intent].append(ps)

            keys = ["ALL", gold_domain]
            accumulate_stats(domain2stats, keys, 1, 1)
            accumulate_stats(domain2stats, keys, 0, int(gold_intent == pred_intent))
            gold_spans = set([f"{k}-{v}" for k, v in gold_spans.items()])
            pred_spans = set([f"{k}-{v}" for k, v in pred_spans.items()])
            accumulate_stats(domain2stats, keys, 2, len(gold_spans & pred_spans))
            accumulate_stats(domain2stats, keys, 3, len(pred_spans - gold_spans))
            accumulate_stats(domain2stats, keys, 4, len(gold_spans - pred_spans))
            accumulate_stats(domain2stats, keys, 5, int(gold_intent == pred_intent and gold_spans == pred_spans))


        print(f"unknown domains [{len(unknown_domains)}]: {json.dumps(Counter(unknown_domains), indent=2)}")
        print(f"unknown intents [{len(unknown_intents)}]: {json.dumps(Counter(unknown_intents), indent=2)}")
        unknown_slots = {f"{k} [{len(v)}]": Counter(v) for k, v in unknown_slots.items()}
        print(f"unknown slots: {json.dumps(unknown_slots, indent=2)}")

        domain_df = pd.DataFrame.from_dict(domain2stats, orient="index", columns=["icorr", "itotal", "stp", "sfp", "sfn", "ocorr"])
        domain_df["intent_acc"] = domain_df["icorr"] / domain_df["itotal"] * 100
        domain_df["slot_precision"] = domain_df["stp"] / (domain_df["stp"] + domain_df["sfp"] + 1e-10) * 100
        domain_df["slot_recall"] = domain_df["stp"] / (domain_df["stp"] + domain_df["sfn"] + 1e-10) * 100
        domain_df["slot_f1"] = 2 * domain_df["slot_precision"] * domain_df["slot_recall"] / (
                domain_df["slot_precision"] + domain_df["slot_recall"] + 1e-10)
        domain_df["overall_acc"] = domain_df["ocorr"] / domain_df["itotal"] * 100
        print(f"domain stats:\n{domain_df.round(3)}")

        all_rows = domain_df.loc["ALL"].to_dict()
        return_keys = ["intent_acc", "slot_precision", "slot_recall", "slot_f1", "overall_acc"]
        return {k: all_rows[k] for k in return_keys}
