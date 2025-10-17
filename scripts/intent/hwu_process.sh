#!/bin/bash

abspath=$(cd "$(dirname "$0")";pwd)
pwd

source "${abspath}/../common.sh"

# Parameters
DATA_NAME=hwu
PROJECT_ROOT=./

MODEL_MAX_LENGTH=4096
num_duplication=2
intent_dropout_ratio=0.33
#TRAINING_LEVEL=session_level
TRAINING_LEVEL=turn_level
num_shot=8
history_ratio=0
PROCESSOR_VERSION=v1.0

#do_process=true
#do_infer=false

if [[ ${few_shot_training} != '' ]]; then
    num_shot=2
    num_duplication=4
    intent_dropout_ratio=0.33
fi

if [[ ${do_infer} == true ]]; then
    TRAINING_LEVEL=turn_level
    num_shot=3
fi

#    --num_infer_samples=100 \
#    --debug=true \
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
    --num_duplication=${num_duplication} \
    --intent_dropout_ratio=${intent_dropout_ratio} \
    --training_level ${TRAINING_LEVEL} \
    --history_ratio=${history_ratio} \
    --use_true_prev_domain=true \
    --use_true_prev_intent=true \
    --use_true_curr_domain=false \
    --use_true_curr_intent=false \
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
    --max_tokens=100 \
    "${remain_params[@]}"
