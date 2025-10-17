#!/bin/bash

abspath=$(cd "$(dirname "$0")";pwd)
cd "${abspath}/../../../"
pwd

data_version=$1
echo "data version: ${data_version}"
prediction_dir=InstructToD/infer_results/v1.0/sgd/${data_version}
prediction_tag=${prediction_dir}/$2
echo "prediction file: ${prediction_tag}.json"

sgd_data_dir=InstructToD/raw_data/sgd/${data_version}
eval_set=test
use_fuzzy_match=True

python InstructToD/tools/convert_sgd_predictions.py \
    --dataset_dir ${sgd_data_dir}/${eval_set} \
    --prediction_file ${prediction_tag}.json

python -m schema_guided_dst.evaluate \
    --dstc8_data_dir ${sgd_data_dir} \
    --prediction_dir ${prediction_tag} \
    --eval_set ${eval_set} \
    --output_metric_file ${prediction_tag}/_offical_metrics_fz-${use_fuzzy_match}.json \
    --use_fuzzy_match=${use_fuzzy_match}