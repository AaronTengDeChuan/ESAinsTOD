#!/bin/bash

abspath=$(cd "$(dirname "$0")";pwd)
pwd

source "${abspath}/../common.sh"

# Parameters
DATA_NAME=snips
PROJECT_ROOT=./

MODEL_MAX_LENGTH=4096
TRAINING_LEVEL=turn_level
num_shot=8
history_ratio=0
PROCESSOR_VERSION=v1.0

if [[ ${few_shot_training} != '' ]]; then
    num_shot=2
fi

if [[ ${do_infer} == true ]]; then
    TRAINING_LEVEL=turn_level
    num_shot=3
fi

#    --tokenizer_path ${TOKENIZER_PATH} \
python -u main.py \
    --get_hparams=${get_hparams} \
    --do_process=${do_process} \
    --do_infer=${do_infer} \
    --processor_version ${PROCESSOR_VERSION} \
    --project_root ${PROJECT_ROOT} \
    --data_name ${DATA_NAME} \
    --llama \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --training_level ${TRAINING_LEVEL} \
    --history_ratio=${history_ratio} \
    --use_true_prev_domain=true \
    --use_true_prev_intent=true \
    --use_true_prev_bspn=true \
    --use_true_curr_domain=false \
    --use_true_curr_intent=false \
    --use_true_curr_bspn=false \
    --few_shot_training=${few_shot_training} \
    --few_shot_evaluation=false \
    --num_shot=${num_shot} \
    --start_with_bos=true \
    --do_sample=true \
    --temperature 0 \
    --num_return_sequences 1 \
    --top_p 1 \
    --infer_backend=vllm \
    --num_processes=54 \
    --max_tokens=400 \
    "${remain_params[@]}"
