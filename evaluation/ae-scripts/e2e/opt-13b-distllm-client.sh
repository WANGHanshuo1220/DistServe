#!/bin/bash
# . /app/distserve/distserve/evaluation/ae-scripts/e2e/common.sh

root_path="/root/DistServe"
dataset_path="/root/ShareGPT_V3_unfiltered_cleaned_split.json"
tokenizer="/root/models/facebook/opt-125m"

echo "Starting distllm client... (for full evaluation, OPT-13B)"
python3 ${root_path}/evaluation/2-benchmark-serving/2-benchmark-serving.py \
    --backend distserve \
    --tokenizer ${tokenizer} \
    --dataset ${dataset_path} \
    --num-prompts-req-rates "[(1, 10)]" \
    --exp-result-root ${root_path}/output \
    --exp-result-dir opt-13b-sharegpt

    # --num-prompts-req-rates "[(2, 0.75), (300, 1.5), (300, 3), (300, 4.5), (300, 6), (300, 6.75), (300, 7.5), (300, 9)]" \
