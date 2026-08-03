#!/usr/bin/env sh

# Run GRPO training for video multiple-choice QA.
# Activate the intended Python/Conda environment before launching this script.

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT" || exit 1

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Make sure torchrun comes from the currently activated Conda environment.
if [ -n "${CONDA_PREFIX:-}" ]; then
    export PATH="$CONDA_PREFIX/bin:$PATH"
    # DeepSpeed CPUAdam may need Intel OpenMP symbols from the active Conda env.
    if [ -f "$CONDA_PREFIX/lib/libiomp5.so" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libiomp5.so${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

WANDB_NAME=$(basename "$0")_$(date +"%Y%m%d_%H%M%S")
OUTDIR="./checkpoints/$WANDB_NAME"
export LOG_PATH="./logs/${WANDB_NAME}.log"
mkdir -p ./checkpoints ./logs

# Select visible GPUs. Keep this consistent with --nproc_per_node.
export CUDA_VISIBLE_DEVICES=0,1

# Parameters you usually need to adjust:
#   --nproc_per_node: number of training processes on this node, usually the GPU count.
#   --model_name_or_path: base model checkpoint.
#   --train_data_path: CSV annotation file.
#   --train_video_folder: root folder containing videos.
#   --dataset_name: dataset tag used in output JSONL filenames.
#   --use_think_prompt: true for <think>...</think><answer>...</answer>, false for answer only.
#   --num_generations: sampled completions per prompt for GRPO.
#   --generation_batch_size: total rollout generation batch size.
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=10675 \
    src/open_r1/grpo_qa_baseline_expansion_threshold_redistribution_rewrite.py \
    --deepspeed scripts/training/zero3_offload.json \
    --output_dir "$OUTDIR" \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --train_data_path annotations/test_ucf_crime_rewritten.csv \
    --train_video_folder ../ucf_videos \
    --dataset_name ucf \
    --use_think_prompt false \
    --max_completion_length 20 \
    --num_generations 2 \
    --generation_batch_size 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 true \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation sdpa \
    --num_train_epochs 1 \
    --run_name "$WANDB_NAME" \
    --report_to tensorboard \
    --save_steps 40 \
    --save_total_limit 11 \
    --save_only_model true \
    --dataloader_num_workers 0
