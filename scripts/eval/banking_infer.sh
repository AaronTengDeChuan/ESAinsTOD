#!/bin/bash

model_path="saved_models/llama2-7b-e2e-lu/checkpoint-2080"

bash scripts/intent/banking_process.sh \
    --do_infer \
    --model_name_or_path=${model_path}
