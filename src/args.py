"""
Parse argument.
"""

import json
import argparse
from collections import Counter


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class HParams(dict):
    """ Hyper-parameters class

    Store hyper-parameters in training / infer / ... scripts.
    """

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        for v in self.values():
            if isinstance(v, HParams):
                if name in v:
                    return v[name]
        raise AttributeError(f"'HParams' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def modify_attr(self, name, value):
        if name in self.keys():
            self[name] = value
            return
        for v in self.values():
            if isinstance(v, HParams):
                if name in v:
                    v[name] = value
                    return
        raise AttributeError(f"'HParams' object has no attribute '{name}'")

    def save(self, filename):
        with open(filename, "w", encoding="utf-8") as fp:
            json.dump(self, fp, ensure_ascii=False,
                      indent=4, sort_keys=False)

    def load(self, filename):
        if isinstance(filename, str):
            with open(filename, "r", encoding="utf-8") as fp:
                params_dict = json.load(fp)
        elif isinstance(filename, dict):
            params_dict = filename
        else:
            raise ValueError(f"Unsupported type: {type(filename)}")
        all_keys = []
        for k, v in params_dict.items():
            if isinstance(v, dict):
                all_keys.extend(list(v.keys()))
                if k not in self:
                    self[k] = HParams()
                self[k].update(HParams(v))
            else:
                all_keys.append(k)
                self[k] = v
        # check duplicate keys
        counter = Counter(all_keys)
        duplicate_keys = [k for k, v in counter.items() if v > 1]
        if len(duplicate_keys) > 0:
            raise ValueError(f"Duplicate keys: {duplicate_keys}")


def parse_args(parser):
    """ Parse hyper-parameters from cmdline. """
    parsed = parser.parse_args()
    args = HParams()
    optional_args = parser._action_groups[1]
    for action in optional_args._group_actions[1:]:
        arg_name = action.dest
        args[arg_name] = getattr(parsed, arg_name)
    for group in parser._action_groups[2:]:
        group_args = HParams()
        for action in group._group_actions:
            arg_name = action.dest
            group_args[arg_name] = getattr(parsed, arg_name)
        if len(group_args) > 0:
            args[group.title] = group_args
    return args
