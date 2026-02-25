#!/bin/bash
#SBATCH --job-name=refcoco_bbox
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=230000
#SBATCH --time=12:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail
set -o pipefail
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate verl

# --- Environment setup ---
cd /users/$USER/verl-main

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_SOCKET_IFNAME=lo

export HF_HOME=${HF_HOME:-$SCRATCH/hf_home}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME}

export RAY_worker_env_HF_HOME=$HF_HOME
export RAY_worker_env_TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE
export RAY_worker_env_HF_DATASETS_CACHE=$HF_DATASETS_CACHE
export RAY_worker_env_SSL_CERT_FILE=${SSL_CERT_FILE:-}
export RAY_worker_env_REQUESTS_CA_BUNDLE=${REQUESTS_CA_BUNDLE:-}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}

export RAY_IGNORE_UNHANDLED_ERRORS=0
export RAY_TASK_ERROR_ON_RPC_ERROR=1
export RAY_TASK_IGNORE_ERRORS=0

export RAY_TMPDIR="/users/$USER/ray_$USER"
export RAY_SESSION_DIR="$RAY_TMPDIR"

export PYTHONPATH="/users/$USER/data_postprocess:/users/$USER/verl-main:${PYTHONPATH:-}"

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus

ulimit -n 65535
BASE="/users/$USER/verl-main"

# --- Configuration paths ---
CONFIG_PATH="$BASE/examples/sglang_multiturn/config"
TOOL_CONFIG_PATH="$BASE/examples/sglang_multiturn/config/bbox_only_tool_config.yaml"
CHAT_TEMPLATE_PATH="$BASE/examples/sglang_multiturn/config/qwen_vl_tool_chat_template.jinja2"


TRAIN_FILE="/users/abollengier/verl-main/data/refcoco/bbox/train.parquet"
VAL_FILE="/users/abollengier/verl-main/data/refcoco/bbox/test.parquet"

ERROR_COUNT=0

echo "Checking required files..."

if [ -f "$TRAIN_FILE" ]; then
    SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
    echo "Train file exists: $TRAIN_FILE ($SIZE, $COUNT samples)"
else
    echo "Train file NOT found: $TRAIN_FILE"
    ((ERROR_COUNT++))
fi

if [ -f "$VAL_FILE" ]; then
    SIZE=$(du -h "$VAL_FILE" | cut -f1)
    COUNT=$(python3 -c "import pandas as pd; print(len(pd.read_parquet('$VAL_FILE')))" 2>/dev/null || echo "?")
    echo "Test file exists: $VAL_FILE ($SIZE, $COUNT samples)"
else
    echo "Test file NOT found: $VAL_FILE"
    ((ERROR_COUNT++))
fi

if [ -f "$TOOL_CONFIG_PATH" ]; then
    echo "Tool config exists: $TOOL_CONFIG_PATH"
else
    echo "Tool config NOT found: $TOOL_CONFIG_PATH"
    ((ERROR_COUNT++))
fi

if [ -f "$CHAT_TEMPLATE_PATH" ]; then
    echo "Chat template exists: $CHAT_TEMPLATE_PATH"
else
    echo "Chat template NOT found: $CHAT_TEMPLATE_PATH"
    ((ERROR_COUNT++))
fi

if [ $ERROR_COUNT -gt 0 ]; then
    echo "ERROR: $ERROR_COUNT required files are missing!"
    exit 1
fi

echo "Training Configuration:"
echo "  Train file: $TRAIN_FILE"
echo "  Validation file: $VAL_FILE"
echo "  Tool config: $TOOL_CONFIG_PATH"
echo "  Chat template: $CHAT_TEMPLATE_PATH"
echo ""

# Set environment variables for template
export VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"
export RAY_worker_env_VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"
echo "Custom chat template set: $VERL_CUSTOM_CHAT_TEMPLATE_PATH"

# --- Run training ---
python3 /users/$USER/verl-main/data_postprocess/verl_metrics.py \
    --config-path="$CONFIG_PATH" \
    --config-name='refcoco_bbox_grpo' \
    data.prompt_key='prompt' \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.rollout.trace.backend=tensorboard \
    trainer.default_local_dir="$SCRATCH/checkpoint/refcoco_bbox" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \

EXPERIMENT_NAME="refcoco_bbox_spatial_$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT_NAME

    trainer.project_name='refcoco_bbox_training' \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    "$@"

echo "Training completed!"
