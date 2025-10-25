# coding: utf-8

import os
import re
import json
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import Counter
from itertools import chain


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


class IntentEvaluator(object):
    def __init__(self, reader):
        self.reader = reader
        if getattr(self.reader, "all_services", None) is None:
            self.all_domains = deepcopy(self.reader.all_domains)
        else:
            self.all_domains = deepcopy(self.reader.all_services)
        self.all_domains += ["NONE", "UNK"]
        self.domain2id = {domain: i for i, domain in enumerate(self.all_domains)}
        self.id2domain = {i: domain for i, domain in enumerate(self.all_domains)}
        self.all_intents = [f"{domain}-{intent}" for domain, dom_schema in self.reader.schema.items() for intent in dom_schema['intents']]
        self.all_intents += ["NONE-NONE", "UNK"]
        self.intent2id = {intent: i for i, intent in enumerate(self.all_intents)}
        self.id2intent = {i: intent for i, intent in enumerate(self.all_intents)}

    def validation_metric(self, results, infer_samples_file=None):
        int_unk = []
        # TP, FP, FN, TN
        domain_confusion_matrix = np.zeros((len(self.domain2id), 4), dtype=int)
        intent_confusion_matrix = np.zeros((len(self.intent2id), 4), dtype=int)
        dom_corr, intent_corr, dom_total, intent_total = 0, 0, 0, 0
        for result in results:
            dial_id = result[-1]['dial_id']
            user = result[-1]['user']
            if isinstance(result[-1]['ispn'], list):
                gold_domain = result[-1]['dspn'][0] if result[-1]['dspn'] else "NONE"
                gold_intents = {gold_domain: result[-1]['ispn'] if result[-1]['ispn'] else ["NONE"]}
                pred_domain = result[-1]['dspn_gen'][0] if result[-1]['dspn_gen'] else "NONE"
                pred_intents = {pred_domain: result[-1]['ispn_gen'] if result[-1]['ispn_gen'] else ["NONE"]}
            elif isinstance(result[-1]['ispn'], dict):
                gold_intents = result[-1]['ispn'] if result[-1]['ispn'] else {"NONE": ["NONE"]}
                if result[-1]['ispn_gen'] and isinstance(result[-1]['ispn_gen'], dict):
                    pred_intents = {
                        k: v if isinstance(v, list) else [v]
                        for k, v in result[-1]['ispn_gen'].items() if v and isinstance(v, list | str)}
                    if not pred_intents:
                        pred_intents = {"NONE": ["NONE"]}
                else:
                    pred_intents = {"NONE": ["NONE"]}
            else:
                raise ValueError(f"Unknown type of intent prediction: {type(result[-1]['ispn'])} - {result[-1]['ispn']}")

            # if len(pred_domain) > 1:
            #     print(f"[{dial_id}] '{user}' - Warning: multiple domain predictions: {result[-1]['dspn_gen']}")
            # if len(pred_intent) > 1:
            #     print(f"[{dial_id}] '{user}' - Warning: multiple intent predictions: {result[-1]['ispn_gen']}")
            # pred_domain, pred_intent = pred_domain[0], pred_intent[0]

            for gold_domain, gis in gold_intents.items():
                pred_domain = gold_domain if gold_domain in pred_intents else ""
                dom_corr += int(gold_domain == pred_domain)
                dom_total += 1
                pis = pred_intents.get(gold_domain, [])
                for gold_intent in gis:
                    pred_intent = gold_intent if gold_intent in pis else ""
                    gold_di = f"{gold_domain}-{gold_intent}" if gold_intent != "NONE" else "NONE-NONE"
                    pred_di = f"{pred_domain}-{pred_intent}" if pred_intent != "NONE" else "NONE-NONE"
                    domain_confusion_matrix[self.domain2id[gold_domain]][0 if pred_domain == gold_domain else 2] += 1
                    intent_confusion_matrix[self.intent2id[gold_di]][0 if pred_di == gold_di else 2] += 1

                    intent_corr += int(gold_di == pred_di)
                    intent_total += 1
            for pred_domain, pis in pred_intents.items():
                if pred_domain not in gold_intents:
                    domain_confusion_matrix[self.domain2id.get(pred_domain, self.domain2id["UNK"])][1] += 1
                for pred_intent in pis:
                    if pred_intent not in gold_intents.get(pred_domain, []):
                        pred_di = f"{pred_domain}-{pred_intent}" if pred_intent != "NONE" else "NONE-NONE"
                        pred_di = "UNK" if pred_di not in self.intent2id else pred_di
                        if pred_di == "UNK":
                            int_unk.append(f"{result[-1]['dspn']}-{result[-1]['ispn']} v.s. {result[-1]['real_dspn_gen']}-{result[-1]['real_ispn_gen']}")
                        intent_confusion_matrix[self.intent2id[pred_di]][1] += 1

        detailed_dom = self.confusion_matrix_to_metric(domain_confusion_matrix, "domain", self.all_domains)
        print(f"Domain Metrics:\n", detailed_dom.round(3), end="\n\n")
        detailed_int = self.confusion_matrix_to_metric(intent_confusion_matrix, "intent", self.all_intents)
        print(f"Intent Metrics:\n", detailed_int.round(3), end="\n\n")
        print(f"UNK cases:\n{json.dumps(Counter(int_unk), indent=2)}")
        metrics = {
            "domain_acc": dom_corr / dom_total * 100,
            "intent_acc": intent_corr / intent_total * 100,
        }
        return metrics

    def confusion_matrix_to_metric(self, confusion_matrix, index_name, all_objects):
        # delete objects of which all values are 0
        index_list = [obj for idx, obj in enumerate(all_objects) if np.sum(confusion_matrix[idx]) > 0]
        confusion_matrix = confusion_matrix[[all_objects.index(obj) for obj in index_list]]

        detailed_dom = pd.DataFrame({
            f"{index_name}": index_list,
            "tp": confusion_matrix[:, 0],
            "fp": confusion_matrix[:, 1],
            "fn": confusion_matrix[:, 2],
            "tn": confusion_matrix[:, 3],
        }).set_index(index_name)
        detailed_dom["precision"] = detailed_dom["tp"] / (detailed_dom["tp"] + detailed_dom["fp"] + 1e-10)
        detailed_dom["recall"] = detailed_dom["tp"] / (detailed_dom["tp"] + detailed_dom["fn"] + 1e-10)
        detailed_dom["f1"] = 2 * detailed_dom["precision"] * detailed_dom["recall"] / (
                    detailed_dom["precision"] + detailed_dom["recall"] + 1e-10)
        detailed_dom.loc["Macro"] = detailed_dom.sum()
        detailed_dom.loc["Macro", ['precision', 'recall', 'f1']] /= (len(index_list) - int("UNK" in index_list))
        detailed_dom.loc["Micro"] = deepcopy(detailed_dom.loc["Macro"])
        detailed_dom.loc["Micro", ['precision', 'recall']] = [
            detailed_dom.loc["Micro"]['tp'] / (detailed_dom.loc["Micro"]['tp'] + detailed_dom.loc["Micro"]['fp'] + 1e-10),
            detailed_dom.loc["Micro"]['tp'] / (detailed_dom.loc["Micro"]['tp'] + detailed_dom.loc["Micro"]['fn'] + 1e-10)]
        precision, recall = detailed_dom.loc["Micro"]['precision'], detailed_dom.loc["Micro"]['recall']
        detailed_dom.loc["Micro", "f1"] = 2 * precision * recall / (precision + recall + 1e-10)
        return detailed_dom.reset_index()