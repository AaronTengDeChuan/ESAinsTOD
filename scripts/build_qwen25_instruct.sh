#!/bin/bash


model_path=/home/dcteng/work/LLM-Engines/LLaMA-Factory/saves/qwen25-0.5b-instruct/full/sft
model_specific_args="--processor_version=qwen25 --model_name_or_path=${model_path} --tokenizer_path= --special_tokens_file=qwen25_special_tokens.json"
echo "model_specific_args: ${model_specific_args}"

common_args="--do_process ${model_specific_args}"
#common_args="--do_process ${model_specific_args} --turn_schema=true"
echo "common_args: ${common_args}"

use_raw_uttr=true


bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true
exit 0


# kvret
bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true
## no schema info
#bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_task_instruction=true
### simple task instructions
#bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --simple_instruction=true
### no task instructions and no schema info
#bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
### simple task instructions and no schema info
#bash scripts/e2e/kvret_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true

# camrest
bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false
bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true
## no schema info
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false --no_schema_info=true
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false --no_task_instruction=true
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false --simple_instruction=true
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false --no_task_instruction=true --no_schema_info=true
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=false --simple_instruction=true --no_schema_info=true
#bash scripts/e2e/camrest_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --disable_sys_act=true --simple_instruction=true --no_schema_info=true

# multiwoz 2.1
bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true
## no schema info
#bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/multiwoz_process.sh ${common_args} --use_raw_utterance=${use_raw_uttr} --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true

# sgd
bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true
## no schema info
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
#bash scripts/e2e/sgd_process.sh ${common_args} --data_version=mix_1 --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true

# frames
# Note: since disable_concrete_resp_loss is set to True, no need to remain system actions
bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true
## no schema info
#bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
#bash scripts/e2e/frames_process.sh ${common_args} --disable_sys_act=true --simple_instruction=true --no_schema_info=true

# bitod
bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true
## no schema info
#bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
#bash scripts/e2e/bitod_process.sh ${common_args} --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true

# star
bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false
bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true
## no schema info
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false --no_schema_info=true
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true --no_schema_info=true
## no task instructions
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false --no_task_instruction=true
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true --no_task_instruction=true
## simple task instructions
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false --simple_instruction=true
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false --no_task_instruction=true --no_schema_info=true
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=false --simple_instruction=true --no_schema_info=true
#bash scripts/e2e/star_process.sh ${common_args} --disable_sys_act=true --simple_instruction=true --no_schema_info=true