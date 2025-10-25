import os
import json
from itertools import chain
from collections import defaultdict, Counter

from src.utils import sgd_ontology, utils, metric_dst
from src.utils.eval import BLEUScorer
from src.utils.metric_intent import IntentEvaluator

class SGDEvaluator(object):
    def __init__(self, reader):
        self.reader = reader
        self.did2meta = reader.did2meta

        self.bleu_scorer = BLEUScorer()
        self.intent_scorer = None if self.reader.disable_intent else IntentEvaluator(reader)
        self.tracker_scorer = metric_dst.SGDTrackerEvaluator(self.did2meta, data_version=self.reader.data_version)

    def pack_dial(self, data):
        dials = {}
        for turn in data:
            dial_id = turn['dial_id']
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def validation_metric(self, data, infer_samples_file=None):
        # run sgd official evaluation shell script
        try:
            data_version = self.reader.data_version
            assert data_version in infer_samples_file and infer_samples_file.endswith('.json')
            prediction_tag = os.path.basename(infer_samples_file).replace('.json', '')
            os.system(f"bash scripts/e2e/sgd_offical_evaluation.sh {data_version} {prediction_tag}")
        except Exception as e:
            print(e)
            print(utils.highlight(f"Failed to run sgd official evaluation shell script, please run it manually.", 'red'))

        # TODO: bleu for concrete response
        bleu = self.bleu_metric(data)
        dials = self.pack_dial(data)
        jga, alpha2fga, domain_jga, slot_f1 = self.tracker_scorer.run_metrics(
            dials, enable_label_mapping=False)
        print(len(domain_jga), json.dumps(domain_jga, indent=2))
        print(len(slot_f1), json.dumps(slot_f1, indent=2))
        match, success = self.match_metric(dials)
        req_f1, req_p, req_r, req_slot_prf = self.request_metric(dials)
        req_f1, req_p, req_r = req_f1 * 100, req_p * 100, req_r * 100
        req_slot = {k: f"p={v[0]}/({v[0]}+{v[1]})={v[3]:.2f} r={v[0]}/({v[0]}+{v[2]})={v[4]:.2f} f1={v[5]:.2f}"
                    for k, v in req_slot_prf.items()}
        print(f"User Request Metrics: {json.dumps(req_slot, indent=2)}")

        metrics = {
            'bleu': bleu,
            'match': match * 100,
            'success': success * 100,
            'req_f1': req_f1,
            **self.intent_metric(data),
            'joint_goal': jga * 100,
            'flexible_goals': {k: v * 100 for k, v in alpha2fga.items()},
            'req-p/r/f1': (req_p, req_r, req_f1),
            "req_slot": "; ".join([f"[{k}] {v}" for k, v in req_slot.items()]),
        }

        if self.reader.gen_concrete_resp and self.reader.has_concrete_resp:
            concrete_bleu = self.bleu_metric(data, column="concrete_resp")
            metrics['concrete_bleu'] = concrete_bleu
        return metrics

    def bleu_metric(self, data, column="resp"):
        gen, truth = [], []
        for row in data:
            gen.append(row[f"{column}_gen"])
            truth.append(row[column])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        if gen and truth:
            try:
                sc = self.bleu_scorer.score(zip(wrap_generated, wrap_truth))
            except:
                sc = 0.0
        else:
            sc = 0.0
        return sc

    def intent_metric(self, data):
        if self.intent_scorer is None:
            return {}
        flatten_data = [[turn] for turn in data]
        return self.intent_scorer.validation_metric(flatten_data)

    def match_metric(self, dials):
        def check_consistency(service, gs, ps, turn_data):
            curr_slots = set(gs.get(service, {}).keys())
            inform_slots = set(self.tracker_scorer.informable_slots[service.lower()])
            assert curr_slots.issubset(inform_slots), f"{curr_slots} not in {inform_slots}"
            return self.reader.constraint_to_DBpointer(ps, turn_data['dspn'], turn_data)

        match_methods = []
        success_methods = []
        num_match, num_success, total = 0, 0, 0
        for dial_id in dials:
            dial = dials[dial_id]
            dial_meta = self.did2meta[dial_id]
            match_stats = defaultdict(list)
            success_stats = defaultdict(list)
            for turn_id, api_call in sorted(dial_meta["api_calls"].items(), key=lambda x: x[0]):
                service, command, num_results = api_call
                turn_data = dial[turn_id]
                sys_act = turn_data["aspn"]
                delex_resp, delex_resp_gen = turn_data["resp"], turn_data["resp_gen"]
                if "[notify_failure]" not in sys_act and "[notify_success]" not in sys_act and "[value_" in delex_resp:
                    assert num_results > 0, f"{dial_id}-{turn_id}: {api_call}"
                    assert turn_data['dbres'], f"{dial_id}-{turn_id}: {api_call}"
                    match_methods.append(f"{service} -> {command['method']}")
                    # check if the system offered appropriate entities
                    if "[value_" not in delex_resp_gen:
                        match_stats[service].append((False, "[value_*] not in delex_resp_gen"))
                        continue
                    if not check_consistency(service, turn_data["bspn"], turn_data["bspn_gen"], turn_data):
                        match_stats[service].append((False, "constraint not match"))
                        continue
                    match_stats[service].append((True, ""))
                elif "[notify_success]" in sys_act:
                    assert num_results == 1, f"{dial_id}-{turn_id}: {api_call}"
                    assert turn_data['dbres'], f"{dial_id}-{turn_id}: {api_call}"
                    success_methods.append(f"{service} -> {command['method']}")
                    if not check_consistency(service, turn_data["bspn"], turn_data["bspn_gen"], turn_data):
                        success_stats[service].append((False, "constraint not match"))
                        continue
                    success_stats[service].append((True, ""))

            stats = defaultdict(list)
            for service, match_stat in match_stats.items():
                stats[service].extend(["Match:", match_stat[-1][0], match_stat[-1][1]])
            match = all([stat[1] for stat in stats.values()])

            dial_success = True
            for service, success_stat in success_stats.items():
                service_success = all([stat[0] for stat in success_stat])
                stats[service].extend(["Success:", service_success])
                dial_success &= service_success
            success = dial_success

            dial[0]['detailed_info' if match and success else 'error_info'] = {
                'stats': {dom: ', '.join(map(str, stats[dom])) for dom in stats},
            }
            num_match += int(match)
            num_success += int(success)
            total += 1
        match_method_counter = Counter(match_methods).most_common()
        print(f"{utils.highlight('Matched Methods', 'yellow')} [{len(match_method_counter)}]: {json.dumps(match_method_counter)}")
        success_method_counter = Counter(success_methods).most_common()
        print(f"{utils.highlight('Success Methods', 'yellow')} [{len(success_method_counter)}]: {json.dumps(success_method_counter)}")
        return num_match / total, num_success / total

    def request_metric(self, dials):
        def extract_inform_from_sys_act(sys_act):
            inform_slots = []
            curr_act = None
            for span in sys_act.split():
                if span.startswith("["):
                    # assert span.endswith("]"), f"{span} not endswith ']' in {sys_act}"
                    if span.endswith("]"):
                        curr_act = span[1:-1]
                    else:
                        curr_act = None
                elif curr_act == "inform":
                    inform_slots.append(span)
            return set(inform_slots)

        tp, fp, fn = 0, 0, 0
        unconsidered_gold, unconsidered_pred = defaultdict(int), defaultdict(int)
        # [tp, fp, fn, unconsidered, p, r, f]
        req_prf = defaultdict(lambda: [0, 0, 0])
        for dial_id in dials:
            dial = dials[dial_id]
            dial_meta = self.did2meta[dial_id]
            for turn_id, turn_data in enumerate(dial):
                user_req = dial_meta["user_reqs"].get(turn_id, {})
                user_req = set(chain(*user_req.values()))
                sys_act, sys_act_gen = turn_data["aspn"].lower(), turn_data["aspn_gen"].lower()
                gold_inform, pred_inform = extract_inform_from_sys_act(sys_act), extract_inform_from_sys_act(sys_act_gen)
                gold_req = user_req & gold_inform
                pred_req = pred_inform - ((gold_inform | user_req) - gold_req)
                # pred_req = pred_inform
                for slot in user_req - gold_req:
                    unconsidered_gold[slot] += 1
                for slot in pred_inform - pred_req:
                    unconsidered_pred[slot] += 1
                for slot in gold_req:
                    if slot in pred_req:
                        tp += 1
                        req_prf[slot][0] += 1
                    else:
                        fn += 1
                        req_prf[slot][2] += 1
                for slot in pred_req:
                    if slot not in gold_req:
                        fp += 1
                        req_prf[slot][1] += 1
        print(utils.highlight(f"REQUEST: TP={tp}, FP={fp}, FN={fn}", 'yellow'))
        print(f"Unconsidered Gold Req Slots [{len(unconsidered_gold)}] [SUM={sum(unconsidered_gold.values())}]: "
              f"{json.dumps(unconsidered_gold, indent=2)}")
        print(f"Unconsidered Pred Req Slots [{len(unconsidered_pred)}] [SUM={sum(unconsidered_pred.values())}]: "
              f"{json.dumps(unconsidered_pred, indent=2)}")
        precision, recall = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        for _, mat in req_prf.items():
            mat.extend([mat[0] / (mat[0] + mat[1] + 1e-10), mat[0] / (mat[0] + mat[2] + 1e-10)])
            mat.append(2 * mat[-2] * mat[-1] / (mat[-2] + mat[-1] + 1e-10))
        return f1, precision, recall, req_prf
