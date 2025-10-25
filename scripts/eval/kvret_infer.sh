#!/bin/bash

choices=("qwen2.5-3b-instruct" "llama2-7b")

model_name=$1
# check model_name validity
if [[ ! " ${choices[@]} " =~ " ${model_name} " ]]; then
    echo "Invalid model_name: ${model_name}. Please choose from: ' ${choices[@]} '"
    exit 1
fi

if [ "${model_name}" == "llama2-7b" ]; then
    model_params="--use_raw_utterance=false"
    model_path="saved_models/llama2-7b-e2e/checkpoint-1656"
elif [ "${model_name}" == "qwen2.5-3b-instruct" ]; then
    model_params="--use_raw_utterance=true --processor_version=qwen25 --tokenizer_path= --special_tokens_file=qwen25_special_tokens.json"
    model_path="saved_models/qwen25-3b-instruct-e2e/checkpoint-930"
fi

bash scripts/e2e/kvret_process.sh \
    --do_infer \
    --gen_concrete_resp=false \
    ${model_params} \
    --model_name_or_path=${model_path}
