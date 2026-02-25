#!/bin/bash
#SBATCH --job-name=coco_eval
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=230000
#SBATCH --time=1:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate verl

cd /users/$USER/verl-main

# Environment setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export RAY_TMPDIR="/users/$USER/ray_$USER"
export PYTHONPATH="/users/$USER/verl-main:${PYTHONPATH:-}"

# Ray settings
export RAY_worker_env_HF_HOME=$HF_HOME
export RAY_worker_env_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
export RAY_DEDUP_LOGS=0
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1

# Chat template
export VERL_CUSTOM_CHAT_TEMPLATE_PATH="/users/$USER/verl-main/examples/sglang_multiturn/config/qwen_vl_tool_chat_template.jinja2"
export RAY_worker_env_VERL_CUSTOM_CHAT_TEMPLATE_PATH="$VERL_CUSTOM_CHAT_TEMPLATE_PATH"

echo "==================================================================="
echo "Crop Evaluation - Testing Spatial Awareness Transfer"
echo "==================================================================="

# Model to evaluate - default to base Qwen model
# For Hydra, we need to quote paths with special characters
MODEL_PATH="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
echo "Model: $MODEL_PATH"

# Dataset paths
CROP_EVAL_FILE="/users/$USER/verl-main/data/refcoco_tasks/crop/test.parquet"
TOOL_CONFIG="/users/$USER/verl-main/examples/sglang_multiturn/config/crop_only_tool_config.yaml"

echo "Dataset: $CROP_EVAL_FILE"
echo "Tool config: $TOOL_CONFIG"
echo "==================================================================="

ulimit -n 65535

# Config path for Hydra
CONFIG_PATH="/users/$USER/verl-main/examples/sglang_multiturn/config"

# Determine output directory based on model - use unique temp dir to avoid resume
if [[ "$MODEL_PATH" == *"checkpoint"* ]]; then
    # Extract step info from checkpoint path for output naming
    STEP_NAME=$(basename $(dirname "$MODEL_PATH"))
    OUTPUT_DIR="$SCRATCH/checkpoint/crop_eval_results/${STEP_NAME}_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="$SCRATCH/checkpoint/crop_eval_results/baseline_$(date +%Y%m%d_%H%M%S)"
fi

echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run evaluation using verl trainer in eval-only mode
python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='bbox_grpo' \
    data.train_files="$CROP_EVAL_FILE" \
    data.val_files="$CROP_EVAL_FILE" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
    trainer.total_training_steps=0 \
    trainer.val_before_train=True \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="$OUTPUT_DIR"

echo "Crop evaluation completed."
echo "Results saved to: $OUTPUT_DIR"
