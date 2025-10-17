#!/bin/bash

echo "#################### common.sh STARTING ####################"
echo "common.sh: '$0' '$@'"


echo "------------------- Specified Parameters -------------------"

get_hparams=false
debug=false
do_process=false
do_infer=false
few_shot_training=''
checkpoint_idx=0
params=($@)
remain_params=()
for param in ${params[@]}; do
    if [[ ${param} == "--get_hparams" ]]; then
        get_hparams=true
        echo "get_hparams = true"
    elif [[ ${param} == "--debug" ]]; then
        debug=true
        echo "debug = true"
    elif [[ ${param} == "--do_process" ]]; then
        do_process=true
        echo "do_process = true"
    elif [[ ${param} == "--do_infer" ]]; then
        do_infer=true
        echo "do_infer = true"
    elif [[ ${param} == --checkpoint_idx=* ]]; then
        checkpoint_idx=${param#*=}
        echo "checkpoint_idx = ${checkpoint_idx}"
    elif [[ ${param} == --data_version=* ]]; then
        DATA_VERSION=${param#*=}
        echo "DATA_VERSION = ${DATA_VERSION}"
    elif [[ ${param} == --num_infer_samples=* ]]; then
        num_infer_samples=${param#*=}
        echo "num_infer_samples = ${num_infer_samples}"
    elif [[ ${param} == --few_shot_training=* ]]; then
        few_shot_training=${param#*=}
        echo "few_shot_training = ${few_shot_training}"
    elif [[ ${param} == --model_name_or_path=* ]]; then
        model_name_or_path=${param#*=}
        echo "model_name_or_path = ${model_name_or_path}"
    else
        remain_params+=(${param})
    fi
done


echo "---------------- Post-Processing Parameters ----------------"

# if model_name_or_path is set, use it
if [[ -n ${model_name_or_path} ]]; then
    MODEL_PATH=${model_name_or_path}
    echo "MODEL_PATH is changed to ${MODEL_PATH}"
fi

echo "-------------------- Remain Parameters ---------------------"

echo "Parse ${remain_params[@]}"
for param in ${remain_params[@]}; do
    echo ${param}
done

echo "#################### common.sh ENDDING ####################"
