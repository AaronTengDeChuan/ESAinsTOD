#!/bin/bash

abspath=$(cd "$(dirname "$0")";pwd)
pwd
DATA_VERSION=2.1
source "${abspath}/../common.sh"

# Parameters
DATA_NAME=multiwoz
PROJECT_ROOT=./

MODEL_MAX_LENGTH=4096
TRAINING_LEVEL=session_level
history_ratio=0.6
PROCESSOR_VERSION=v1.0

#do_process=false
#do_infer=true
ports=(8000)
# calculate tp size = number of CUDA_VISIBLE_DEVICES / number of ports
#CUDA_VISIBLE_DEVICES=(0 1 2 3 4 5 6 7)
#tp=$(echo "${#CUDA_VISIBLE_DEVICES[@]} / ${#ports[@]}" | bc)

# if do_infer equal true, start vllm api server
#if [ ${do_infer} = true ]; then
#    echo "Please start vllm api server with the following command:"
#    for port in "${ports[@]}"; do
#        server_command="python src/vllm/entrypoints/api_server.py --port ${port} --model ${MODEL_PATH} --tensor-parallel-size ${tp} --disable-log-requests"
#        echo ${server_command}
#        echo ${server_command} >> commands.txt
#    done
#    # wait confirmation from user
#    read -p "Press enter to continue"
#fi

#    --num_infer_samples=100 \
#    --tokenizer_path ${TOKENIZER_PATH} \
python -u main.py \
    --get_hparams=${get_hparams} \
    --debug=${debug} \
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
    --detailed_db_result=true \
    --disable_sys_act=false \
    --use_true_prev_domain=false \
    --use_true_prev_bspn=false \
    --use_true_prev_aspn=false \
    --use_true_prev_resp=false \
    --use_true_prev_concrete_resp=false \
    --use_true_curr_domain=false \
    --use_true_curr_bspn=false \
    --use_true_curr_aspn=false \
    --gen_concrete_resp=true \
    --few_shot_training=${few_shot_training} \
    --concrete_resp_first=false \
    --start_with_bos=true \
    --do_sample=true \
    --temperature 0 \
    --num_return_sequences 1 \
    --top_p 1 \
    --infer_backend=vllm \
    --num_processes=54 \
    --port "${ports[@]}" \
    --max_tokens=300 \
    "${remain_params[@]}"
