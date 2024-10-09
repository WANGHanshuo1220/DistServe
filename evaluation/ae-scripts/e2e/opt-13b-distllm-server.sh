#!/bin/bash
# . /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

root_path="/root/DistServe"
model_name="facebook/opt-125m"

echo "Starting distllm server... (for full evaluation, OPT-13B)"
python3 ${root_path}/evaluation/2-benchmark-serving/2-start-api-server.py \
    --backend distserve \
    --model ${model_name}
    # --model /root/models/facebook/opt-125m

rm -r ${root_path}/output