# coding: utf-8
import os
import json
import argparse

from src.args import str2bool
from src.args import parse_args
from src.data import GenProcessor, MultiWOZProcessor, KvretProcessor, CamRestProcessor, SGDProcessor
from src.data import FramesProcessor, BiToDProcessor, STARProcessor
from src.data import IntentProcessor, HWUProcessor, ClincProcessor, BankingProcessor
from src.data import SnipsProcessor
from src.utils.eval import MultiWOZEvaluator, KvretEvaluator, CamRestEvaluator
from src.utils.sgd_eval import SGDEvaluator
from src.utils.metric_intent import IntentEvaluator
from src.utils.metric_slu import SLUEvaluator
from src.utils import utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--get_hparams", type=str2bool, default=False,
                        help="Whether to get hparams from the command line. If True, the following arguments will be ignored.")
    parser.add_argument("--debug", type=str2bool, default=False,
                        help="Whether to run in debug mode.")
    parser.add_argument("--do_process", type=str2bool, default=False,
                        help="Whether to process the raw data.")
    parser.add_argument("--do_infer", type=str2bool, default=False,
                        help="Whether to run inference on the test dataset.")
    parser.add_argument("--eval_split", type=str, default="test",
                        help="The split of dataset to evaluate.")
    parser.add_argument("--num_infer_samples", type=int, default=None,
                        help="The number of samples need to infer.")

    GenProcessor.add_cmdline_argument(parser)
    IntentProcessor.add_cmdline_argument(parser)

    hparams = parse_args(parser)
    if isinstance(hparams.num_infer_samples, int) and hparams.num_infer_samples <= 0:
        hparams.modify_attr("num_infer_samples", None)

    assert not (hparams.do_process and hparams.do_infer), "Cannot process and infer at the same time."

    # set save dir
    hparams.save_dir = os.path.join(hparams.project_root, "sft_data", f"{hparams.processor_version}")
    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    print(json.dumps(hparams, indent=2))

    try:
        get_hparams = hparams.get_hparams
    except AttributeError:
        get_hparams = False
    if get_hparams:
        print(json.dumps(hparams))
        return

    # set processor and evaluator
    match hparams.data_name:
        case "multiwoz":
            processor = MultiWOZProcessor(hparams)
            evaluator = MultiWOZEvaluator(processor)
        case "kvret":
            processor = KvretProcessor(hparams)
            evaluator = KvretEvaluator(processor)
        case "camrest":
            processor = CamRestProcessor(hparams)
            evaluator = CamRestEvaluator(processor)
        case "sgd":
            processor = SGDProcessor(hparams)
            evaluator = SGDEvaluator(processor)
        case "frames":
            processor = FramesProcessor(hparams)
            evaluator = None
        case "bitod":
            processor = BiToDProcessor(hparams)
            evaluator = None
        case "star":
            processor = STARProcessor(hparams)
            evaluator = None
        case "hwu":
            processor = HWUProcessor(hparams)
            evaluator = IntentEvaluator(processor)
        case "clinc":
            processor = ClincProcessor(hparams)
            evaluator = IntentEvaluator(processor)
        case "banking":
            processor = BankingProcessor(hparams)
            evaluator = IntentEvaluator(processor)
        case "snips":
            processor = SnipsProcessor(hparams)
            evaluator = SLUEvaluator(processor)
        case _:
            raise ValueError(f"Unsupported dataset: {hparams.data_name}")

    if hparams.do_process:
        success = processor.build_instruct_data()
        if success:
            # save hparams
            hparams_file = os.path.join(hparams.save_dir, processor.data_source, "hparams.json")
            hparams.save_dir = os.path.join(hparams.save_dir, processor.data_source)
            hparams.save(hparams_file)
            print(f"Saved hparams to '{hparams_file}'.")

    if hparams.do_infer:
        infer_samples_file = processor.infer(split_name=hparams.eval_split, evaluator=evaluator, num_infer_samples=hparams.num_infer_samples)
        print(infer_samples_file)


if __name__ == "__main__":
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    main()