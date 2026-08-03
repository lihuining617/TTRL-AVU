#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# Set these paths through environment variables before running the script.
# Example:
#   DATASET_PATH=/path/to/videos \
#   MODEL_PATH=/path/to/model \
#   TEST_GT_PATH=/path/to/annotations.csv \
#   bash scripts/evaluation/evaluation_cls_qwen.sh
dataset_path="${DATASET_PATH:?Please set DATASET_PATH to the video directory}"
model_path="${MODEL_PATH:?Please set MODEL_PATH to the model checkpoint}"
test_gt_path="${TEST_GT_PATH:?Please set TEST_GT_PATH to the annotation CSV}"
use_think=false  # set to false to disable --think

# init output path
model_tag=$(echo "$model_path" | cut -d'/' -f2- | tr '/' '_')
think_tag=$( [ "$use_think" = true ] && echo "w_think" || echo "no_think" )
output_csv="./results/cls/${model_tag}/output_$(basename "$dataset_path")_${think_tag}.csv"

# Run inference script
python src/evaluation/inference_cls_qwen.py \
    --dataset "ecva" \
    --video_folder "$dataset_path" \
    --model_path "$model_path" \
    --test_gt_path "$test_gt_path" \
    --output_csv "$output_csv" \
    $( [ "$use_think" = true ] && echo "--think" )

# Evaluate the results
python src/evaluation/evaluation_cls.py \
    --test_gt_path "$test_gt_path" \
    --pred_path "$output_csv"
