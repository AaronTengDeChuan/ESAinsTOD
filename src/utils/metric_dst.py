# coding: utf-8

import os
import re
import math
import json
import numpy as np
from copy import deepcopy
from src.utils import ontology, sgd_ontology, utils
from collections import defaultdict


class GenericTrackerEvaluator(object):
    def __init__(self):
        return

    def normalize_constraint(self, constraint, dial_id, verbose=False):
        raise NotImplementedError

    def mapping_label(self, value_label, inform_label):
        raise NotImplementedError

    def run_metrics(self, data, enable_label_mapping=True):
        """
        data: List[Dict]
        """
        domain_jga = {}
        slot_confusion_matrix = {}
        for dom in self.informable_slots.keys():
            # [correct, wrong, miss, false positive]
            domain_jga[dom] = [0, 0, 0, 0]
            for slot in self.informable_slots[dom]:
                #               Pred Positive   Pred Negative
                # Gold Positive     TP              FN
                # Gold Negative     FP              TN
                slot_confusion_matrix[f"{dom}-{slot}"] = np.zeros((2, 2), dtype=int)

        jga, fgas, total = 0, defaultdict(float), 0
        for dial_id, turns in data.items():
            tracker_output, tracker_label = [], []
            for turn_id, turn in enumerate(turns):
                bspn, bspn_gen = turn.get("bspn", ""), turn.get("bspn_gen", "")
                legal_flag = isinstance(bspn, dict) and isinstance(bspn_gen, dict)
                if turn_id == 0 and not legal_flag:
                    continue
                assert legal_flag, f"bspn and bspn_gen should be dict, but got '{bspn}' and '{bspn_gen}'"
                tracker_output.append(turn["bspn_gen"])
                tracker_label.append(turn["bspn"])
            corr, flexible_corrs, num = self.tracker_eval(
                dial_id, tracker_output, tracker_label, domain_jga, slot_confusion_matrix, enable_label_mapping)
            jga += corr
            for k, v in flexible_corrs.items():
                fgas[k] += v
            total += num
        fgas = {k: v / total for k, v in fgas.items()}

        dom_f1 = {}
        for domain, matrix in domain_jga.items():
            tp, wrong, miss, fp = matrix
            if sum(matrix) == 0:
                continue
            precision = tp / (tp + wrong + fp) if tp + wrong + fp > 0 else 0
            recall = tp / (tp + wrong + miss) if tp + wrong + miss > 0 else 0
            acc = tp / (tp + wrong + miss + fp) if tp + wrong + miss + fp > 0 else 0
            dom_f1[domain] = str([tp, wrong, miss, fp, acc, precision, recall])

        slot_f1 = {}
        for dom_slot, matrix in slot_confusion_matrix.items():
            tp, fn, fp, tn = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
            if tp + fp + fn == 0:
                continue
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
            slot_f1[dom_slot] = str([tp, fp, fn, tn, precision, recall, f1])

        return jga / total, fgas, dom_f1, slot_f1

    def tracker_eval(self, dial_id, tracker_output, tracker_label, domain_jga, slot_confusion_matrix,
                     enable_label_mapping=True):
        """
        eval one dialogue per time
        tracker_output: List[Dict]
        tracker_label: List[Dcit]
        """

        def check_latest_state(gs, ps, prev_gs, prev_ps):
            # return: turn_match, type_2_error
            if gs == ps:
                return True, False
            ds_func = lambda s: '-'.join(s.split('-', maxsplit=2)[:2])
            delta_gs, delta_prev_gs = gs - prev_gs, prev_gs - gs
            delta_ps, delta_prev_ps = ps - prev_ps, prev_ps - ps
            deleted_gs = set(map(ds_func, delta_prev_gs)) - set(map(ds_func, delta_gs))
            deleted_ps = set(map(ds_func, delta_prev_ps)) - set(map(ds_func, delta_ps))

            if delta_ps.issubset(gs) and delta_gs.issubset(ps) and \
                    deleted_ps.isdisjoint(set(map(ds_func, gs))) and deleted_gs.isdisjoint(set(map(ds_func, ps))):
                return True, True
            else:
                return False, None

        def flexible_goal_accuracy(turn_records, alpha=0.5):
            # return: fga
            fga, err_idx = 0, -1
            for idx, (turn_match, type_2_error) in enumerate(turn_records):
                if turn_match:
                    fga += 1
                    if type_2_error:
                        # TODO: address special cases where some slots are deleted
                        assert err_idx >= 0, \
                            (f"Error index should be greater than 0, but got {err_idx}: {turn_records}\n"
                             f"{json.dumps([(tl, to) for tl, to in zip(tracker_label, tracker_output)], indent=2)}")
                        if isinstance(alpha, (int, float)):
                            assert alpha >= 0, f"alpha should be greater than 0, but got {alpha}"
                            fga -= math.exp(-alpha * (idx - err_idx))
                    else:
                        err_idx = -1
                else:
                    err_idx = idx
            return fga

        assert len(tracker_output) == len(tracker_label), \
            f"Length of tracker output and label is not same: {len(tracker_output)} vs {len(tracker_label)}"
        jga_count = 0
        prev_gold_dsv, prev_pred_dsv, turn_matches = set(), set(), []
        for turn_idx in range(len(tracker_label)):
            gold_state = self.normalize_constraint(tracker_label[turn_idx], dial_id)
            pred_state = self.normalize_constraint(tracker_output[turn_idx], dial_id)
            jga_correct = True
            for dom in self.informable_slots.keys():
                dom_jga_correct = True
                dom_gold_state = gold_state.get(dom, {})
                dom_pred_state = pred_state.get(dom, {})
                if not dom_gold_state and not dom_pred_state:
                    continue
                for slot in self.informable_slots[dom]:
                    dom_slot = f"{dom}-{slot}"
                    gold_value = dom_gold_state.get(slot, "")
                    pred_value = dom_pred_state.get(slot, "")
                    if not gold_value and not pred_value:
                        slot_confusion_matrix[dom_slot][1][1] += 1
                    elif not gold_value and pred_value:
                        slot_confusion_matrix[dom_slot][1][0] += 1
                        dom_jga_correct = False
                    elif gold_value and not pred_value:
                        slot_confusion_matrix[dom_slot][0][1] += 1
                        dom_jga_correct = False
                    else:
                        pred_value = self.mapping_label(gold_value, pred_value) if enable_label_mapping else pred_value
                        pred_state[dom][slot] = pred_value
                        if pred_value == gold_value:
                            slot_confusion_matrix[dom_slot][0][0] += 1
                        else:
                            slot_confusion_matrix[dom_slot][0][1] += 1
                            slot_confusion_matrix[dom_slot][1][0] += 1
                            dom_jga_correct = False
                jga_correct = jga_correct and dom_jga_correct
                if dom_jga_correct:
                    domain_jga[dom][0] += 1
                elif not dom_pred_state:
                    domain_jga[dom][2] += 1
                elif not dom_gold_state:
                    domain_jga[dom][3] += 1
                else:
                    domain_jga[dom][1] += 1
            jga_count += int(jga_correct)
            # records for flexible goal accuracy
            gold_dsv, pred_dsv = flatten_state(gold_state), flatten_state(pred_state)
            turn_matches.append(check_latest_state(gold_dsv, pred_dsv, prev_gold_dsv, prev_pred_dsv))
            prev_gold_dsv, prev_pred_dsv = gold_dsv, pred_dsv
        # calculate flexible goal accuracy
        fgas = {str(alp): flexible_goal_accuracy(turn_matches, alpha=alp) for alp in [0, 0.25, 0.5, 0.75, 1.0, '∞']}
        return jga_count, fgas, len(tracker_label)


def flatten_state(state, prefix=""):
    flat_state = set()
    for k, v in state.items():
        if isinstance(v, dict):
            flat_state.update(flatten_state(v, f"{prefix}{k}-"))
        else:
            flat_state.add(f"{prefix}{k}-{v}")
    return flat_state


def is_in_list(tok, value):
    found = False
    tok_list = [item for item in map(str.strip, re.split("(\W+)", tok)) if len(item) > 0]
    value_list = [item for item in map(str.strip, re.split("(\W+)", value)) if len(item) > 0]
    tok_len = len(tok_list)
    value_len = len(value_list)
    for i in range(tok_len + 1 - value_len):
        if tok_list[i:i + value_len] == value_list:
            found = True
            break
    return found


class MultiWozTrackerEvaluator(GenericTrackerEvaluator):
    def __init__(self, version="2.1"):
        super().__init__()
        self.label_mapping = {
            "2.0": "dataset_config/multiwoz21.json",
            "2.1": "dataset_config/multiwoz21.json",
            "2.2": "dataset_config/multiwoz22.json",
        }
        self.label_maps = self.load_dataset_config(self.label_mapping[version])
        self.informable_slots = deepcopy(ontology.informable_slots)
        self.informable_slots.pop("police", [])
        self.informable_slots.pop("hospital", [])
        self.skip_case = ["dontcare", "dont care", "don't care", "do n't care", "do not care",
                          "any", "none", "not mentioned"]

    @staticmethod
    def load_dataset_config(config_file):
        with open(config_file, "r", encoding='utf-8') as f:
            raw_config = json.load(f)
        return raw_config['label_maps']

    def normalize_constraint(self, state, dial_id, verbose=False):
        state = json.loads(json.dumps(state).lower())
        new_state = {}
        for dom, slot_values in state.items():
            if dom not in self.informable_slots or not isinstance(slot_values, dict):
                continue
            assert '-' not in dom, f"Domain name '{dom}' should not contain '-', but got {state}"
            new_state[dom] = {}
            for slot, value in slot_values.items():
                if not isinstance(value, str):
                    continue
                slot = utils.check_domain_and_slot(slot, state).strip()
                value = value.strip()
                if slot in self.informable_slots[dom] and value and value not in self.skip_case:
                    new_state[dom][slot] = value
        return new_state

    def mapping_label(self, value_label, inform_label):
        value = inform_label
        if value_label == inform_label:
            value = value_label
        elif is_in_list(inform_label, value_label):
            value = value_label
        elif is_in_list(value_label, inform_label):
            value = value_label
        elif inform_label in self.label_maps:
            for inform_label_variant in self.label_maps[inform_label]:
                if value_label == inform_label_variant:
                    value = value_label
                    break
                elif is_in_list(inform_label_variant, value_label):
                    value = value_label
                    break
                elif is_in_list(value_label, inform_label_variant):
                    value = value_label
                    break
        elif value_label in self.label_maps:
            for value_label_variant in self.label_maps[value_label]:
                if value_label_variant == inform_label:
                    value = value_label
                    break
                elif is_in_list(inform_label, value_label_variant):
                    value = value_label
                    break
                elif is_in_list(value_label_variant, inform_label):
                    value = value_label
                    break
        return value


class SGDTrackerEvaluator(GenericTrackerEvaluator):
    def __init__(self, did2meta, data_version):
        super().__init__()
        self.did2meta = did2meta
        self.informable_slots = self.load_dataset_config(f"dataset_config/sgd_schema_{data_version}.json")
        return

    @staticmethod
    def load_dataset_config(schema_file):
        with open(schema_file, "r", encoding='utf-8') as f:
            schema = json.load(f)
        informable_slots = {}
        num_slots, num_inform_slots = 0, 0
        for service_name, service in schema.items():
            # Note: all slots
            all_slots = [slot['name'] for slot in service['slots']]
            # Note: required_slots and optional_slots in all intents of each service
            inform_slots = set()
            for intent in service["intents"]:
                inform_slots.update(intent["required_slots"])
                inform_slots.update(intent["optional_slots"])
            informable_slots[service_name] = list(inform_slots)
            num_slots += len(all_slots)
            num_inform_slots += len(inform_slots)
            print(f"Service '{utils.highlight(service_name, 'yellow')}' "
                  f"has {utils.highlight(len(all_slots), 'yellow')} slots, "
                  f"containing {utils.highlight(len(inform_slots), 'yellow')} informable slots.")
        print(f"Total {utils.highlight(num_slots, 'yellow')} slots, "
              f"containing {utils.highlight(num_inform_slots, 'yellow')} informable slots.")
        return json.loads(json.dumps(informable_slots).lower())

    def normalize_constraint(self, constraint, dial_id, verbose=False):
        value_mapping = json.loads(json.dumps(self.did2meta[dial_id]['value_mapping']).lower())
        constraint = json.loads(json.dumps(constraint).lower())
        cleaned_constraint = {}
        for domain, slots in constraint.items():
            if domain not in self.informable_slots:
                if verbose: print(f"Warning: domain '{domain}' not in informable slots")
                continue
            if not isinstance(slots, dict):
                if verbose: print(f"Warning: slots of domain '{domain}' is not a dict but '{type(slots)}': {constraint}")
                continue
            for slot, value in slots.items():
                if slot not in self.informable_slots[domain]:
                    if verbose: print(f"Warning: slot '{slot}' not in informable slots of domain '{domain}'")
                    continue
                if not isinstance(value, str):
                    if verbose: print(f"Warning: value of slot '{slot}' is not a str but '{type(value)}': {constraint}")
                    continue
                cleaned_constraint.setdefault(domain, {})[slot] = value
        normalized_constraint = sgd_ontology.normalize_constraint(cleaned_constraint, value_mapping)
        return normalized_constraint
