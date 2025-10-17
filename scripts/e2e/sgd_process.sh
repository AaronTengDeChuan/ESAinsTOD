#!/bin/bash

abspath=$(cd "$(dirname "$0")";pwd)
pwd
DATA_VERSION=v0
num_infer_samples=150
source "${abspath}/../common.sh"

# Parameters
DATA_NAME=sgd
PROJECT_ROOT=./

MODEL_MAX_LENGTH=4096
TRAINING_LEVEL=session_level
history_ratio=0.6
PROCESSOR_VERSION=v1.0

#do_process=false
#do_infer=true

#    --tokenizer_path ${TOKENIZER_PATH} \
python -u main.py \
    --get_hparams=${get_hparams} \
    --debug=${debug} \
    --num_infer_samples=${num_infer_samples} \
    --do_process=${do_process} \
    --do_infer=${do_infer} \
    --processor_version ${PROCESSOR_VERSION} \
    --project_root ${PROJECT_ROOT} \
    --data_name ${DATA_NAME} \
    --data_version ${DATA_VERSION} \
    --llama \
    --model_name_or_path ${MODEL_PATH} \
    --model_max_length ${MODEL_MAX_LENGTH} \
    --training_level ${TRAINING_LEVEL} \
    --history_ratio=${history_ratio} \
    --turn_schema=false \
    --inform_request_schema=true \
    --disable_intent=false \
    --detailed_db_result=true \
    --disable_sys_act=false \
    --use_true_prev_domain=true \
    --use_true_prev_intent=false \
    --use_true_prev_bspn=false \
    --use_true_prev_aspn=true \
    --use_true_prev_resp=true \
    --use_true_prev_concrete_resp=true \
    --use_true_curr_domain=true \
    --use_true_curr_intent=false \
    --use_true_curr_bspn=false \
    --use_true_curr_aspn=false \
    --gen_concrete_resp=true \
    --concrete_resp_first=false \
    --start_with_bos=true \
    --do_sample=true \
    --temperature 0 \
    --num_return_sequences 1 \
    --top_p 1 \
    --infer_backend=vllm \
    --max_tokens=400 \
    "${remain_params[@]}"
