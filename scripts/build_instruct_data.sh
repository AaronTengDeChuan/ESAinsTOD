#!/bin/bash

# NOTE: For all mutliwoz sft data with "simple task instructions" option, the most relevant entity is found based on both delex_resp and concrete_resp

# kvret
#bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true
#bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=false
## no schema info
bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true --no_schema_info=true
## no task instructions
bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/kvret_process.sh --do_process --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true


# camrest
#bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false
#bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true
## no schema info
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false --no_schema_info=true
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true --no_schema_info=true
## no task instructions
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false --no_task_instruction=true
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false --simple_instruction=true
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
#bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false --no_task_instruction=true --no_schema_info=true
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=false --simple_instruction=true --no_schema_info=true
bash scripts/e2e/camrest_process.sh --do_process --disable_sys_act=true --simple_instruction=true --no_schema_info=true

# multiwoz 2.0
#bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true
## few shot (5%)
#bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --few_shot_training=5_0
## few shot (10%)
#bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --few_shot_training=10_0
## few shot (10%) + no schema info
#bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --few_shot_training=10_0 --no_schema_info=true
## no schema info
bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --no_schema_info=true
## no task instructions
bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/multiwoz_process.sh --data_version=2.0 --do_process --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true


# multiwoz 2.1
#bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true
#bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=false
# find relevant entity from db results based on both delex_resp and concrete_resp
bash scripts/e2e/multiwoz_process.sh --processor_version=v1.1 --do_process --gen_concrete_resp=true
## no schema info
bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true --no_schema_info=true
## no task instructions
bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## few shot (10%)
#bash scripts/e2e/multiwoz_process.sh --do_process --gen_concrete_resp=true --few_shot_training=10_0

# sgd
#bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true
#bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=false
## no schema info
bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true --no_schema_info=true
## no task instructions
bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/sgd_process.sh --do_process --data_version=mix_1 --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true

# frames
# Note: since disable_concrete_resp_loss is set to True, no need to remain system actions
#bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true
## no schema info
bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true --no_schema_info=true
## no task instructions
bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/frames_process.sh --do_process --disable_sys_act=true --simple_instruction=true --no_schema_info=true

# bitod
#bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true
#bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=false
## no schema info
bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true --no_schema_info=true
## no task instructions
bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/bitod_process.sh --do_process --gen_concrete_resp=true --simple_instruction=true --no_schema_info=true

# star
#bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false
#bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true
## no schema info
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false --no_schema_info=true
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true --no_schema_info=true
## no task instructions
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false --no_task_instruction=true
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true --no_task_instruction=true
## simple task instructions
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false --simple_instruction=true
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true --simple_instruction=true
## no task instructions and no schema info
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false --no_task_instruction=true --no_schema_info=true
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true --no_task_instruction=true --no_schema_info=true
## simple task instructions and no schema info
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=false --simple_instruction=true --no_schema_info=true
bash scripts/e2e/star_process.sh --do_process --disable_sys_act=true --simple_instruction=true --no_schema_info=true

# intent
## (turn-level-concat)
#bash scripts/intent/banking_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0
#bash scripts/intent/clinc_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0
#bash scripts/intent/hwu_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0

#bash scripts/intent/banking_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0.33
#bash scripts/intent/clinc_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0.33
#bash scripts/intent/hwu_process.sh --do_process --num_duplication=2 --intent_dropout_ratio=0.33

#bash scripts/intent/banking_process.sh --do_process --num_duplication=4 --intent_dropout_ratio=0.33
#bash scripts/intent/clinc_process.sh --do_process --num_duplication=4 --intent_dropout_ratio=0.33
#bash scripts/intent/hwu_process.sh --do_process --num_duplication=4 --intent_dropout_ratio=0.33
## (single-turn)
#bash scripts/intent/banking_process.sh --do_process --training_level=session_level --num_shot=1 --num_duplication=1 --intent_dropout_ratio=0
#bash scripts/intent/clinc_process.sh --do_process --training_level=session_level --num_shot=1 --num_duplication=1 --intent_dropout_ratio=0
#bash scripts/intent/hwu_process.sh --do_process --training_level=session_level --num_shot=1 --num_duplication=1 --intent_dropout_ratio=0

#bash scripts/intent/banking_process.sh --do_process --training_level=session_level --num_shot=1 --num_duplication=2 --intent_dropout_ratio=0.33
#bash scripts/intent/banking_process.sh --do_process --training_level=session_level --num_shot=1 --num_duplication=2 --intent_dropout_ratio=0.20

# slu
#bash scripts/slu/snips_process.sh --do_process --training_level=session_level --num_shot=1
#bash scripts/slu/snips_process.sh --do_process --num_duplication=1
#bash scripts/slu/snips_process.sh --do_process --num_duplication=2