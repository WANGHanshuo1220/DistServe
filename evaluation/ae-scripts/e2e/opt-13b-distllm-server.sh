#!/bin/bash
# . /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

root_path="/root/DistServe"

echo "Starting distllm server... (for full evaluation, OPT-13B)"
python3 ${root_path}/evaluation/2-benchmark-serving/2-start-api-server.py \
    --backend distserve \
    --model facebook/opt-125m
    # --model /root/models/facebook/opt-125m
