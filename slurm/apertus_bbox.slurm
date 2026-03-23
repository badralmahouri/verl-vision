#!/bin/bash
#SBATCH --job-name=verl_image_bbox_tool
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=230000
#SBATCH --time=12:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail
set -o pipefail

source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source /opt/conda/etc/profile.d/conda.sh
conda activate verl

# --- Environment setup ---
cd /users/$USER/verl-vision

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

export PYTHONPATH="/users/$USER/data_postprocess:/users/$USER/verl-vision:${PYTHONPATH:-}"


echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus
ulimit -n 65535
BASE="/users/$USER/verl-vision"

# configs
CONFIG_PATH="$BASE/examples/sglang_multiturn/config"
TOOL_CONFIG_PATH="$BASE/examples/sglang_multiturn/config/bbox_only_tool_config.yaml"
CHAT_TEMPLATE_PATH="$BASE/examples/sglang_multiturn/config/qwen_vl_tool_chat_template.jinja2"

# data
TRAIN_FILE="/users/$USER/verl-vision/data/bbox/train.parquet"
VAL_FILE="/users/$USER/verl-vision/data/bbox/test.parquet"

ERROR_COUNT=0
# data presence check
if [ -f "$TRAIN_FILE" ]; then
    SIZE=$(du -h "$TRAIN_FILE" | cut -f1)
    echo "Train file exists: $TRAIN_FILE ($SIZE)"
else
    echo "Train file NOT found: $TRAIN_FILE"
    ((ERROR_COUNT++))
fi

if [ -f "$VAL_FILE" ]; then
    SIZE=$(du -h "$VAL_FILE" | cut -f1)
    echo "Test file exists: $VAL_FILE ($SIZE)"
else
    echo "Test file NOT found: $VAL_FILE"
    ((ERROR_COUNT++))
fi

# tool config checks
if [ -f "$TOOL_CONFIG_PATH" ]; then
    echo "Tool config exists: $TOOL_CONFIG_PATH"
else
    echo "Tool config NOT found: $TOOL_CONFIG_PATH"
    ((ERROR_COUNT++))
fi

# apertus checks
EMU3_CHECKPOINT="${EMU3_CHECKPOINT:-/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF}"
EMU3_TOKENIZER="${EMU3_TOKENIZER:-$EMU3_CHECKPOINT}"

echo "=== tokenizer checks ==="
if [ -d "$EMU3_CHECKPOINT" ]; then
    echo "✓ Apertus checkpoint dir: $EMU3_CHECKPOINT"
    # Verify key files exist in checkpoint
    for f in config.json; do
        if [ -f "$EMU3_CHECKPOINT/$f" ]; then
            echo "  ✓ $f found"
        else
            echo "  ✗ $f NOT found in checkpoint — model loading may fail"
            ((ERROR_COUNT++))
        fi
    done
else
    echo "✗ Apertus checkpoint NOT found: $EMU3_CHECKPOINT"
    echo "  Set EMU3_CHECKPOINT env var or edit this script."
    ((ERROR_COUNT++))
fi

# Check tokenizer (Apertus uses standard HF tokenizer bundled in checkpoint)
if [ -d "$EMU3_TOKENIZER" ]; then
    echo "✓ Tokenizer dir: $EMU3_TOKENIZER"
    for f in tokenizer_config.json; do
        if [ -f "$EMU3_TOKENIZER/$f" ]; then
            echo "  ✓ $f found"
        else
            echo "  ✗ $f NOT found — tokenizer loading may fail"
            ((ERROR_COUNT++))
        fi
    done
else
    echo "✗ Tokenizer NOT found: $EMU3_TOKENIZER"
    ((ERROR_COUNT++))
fi

# Custom chat template for tool calling support for Apertus
if [ -f "$CHAT_TEMPLATE_PATH" ]; then
    echo "Chat template exists: $CHAT_TEMPLATE_PATH"
else
    echo "Chat template NOT found: $CHAT_TEMPLATE_PATH"
    ((ERROR_COUNT++))
fi

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed. Fix the above and retry."
    exit 1
fi

echo ""
echo "=== Training Configuration ==="
echo "  Apertus checkpoint : $EMU3_CHECKPOINT"
echo "  Apertus tokenizer  : $EMU3_TOKENIZER"
echo "  Train data         : $TRAIN_FILE"
echo "  Val data           : $VAL_FILE"
echo "  Tool config        : $TOOL_CONFIG_PATH"
echo "  Chat template      : $CHAT_TEMPLATE_PATH"
echo "==================================================================="
echo ""

export VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"
export RAY_worker_env_VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"
echo "Custom chat template set: $VERL_CUSTOM_CHAT_TEMPLATE_PATH"

# --- Run training ---
python3 /users/$USER/verl-vision/data_postprocess/verl_metrics.py \
    --config-path="$CONFIG_PATH" \
    --config-name='emu3_bbox_grpo' \
    data.prompt_key='prompt' \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.model.path="$EMU3_CHECKPOINT" \
    actor_rollout_ref.model.tokenizer_path="$EMU3_CHECKPOINT" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.trace.backend=tensorboard \
    trainer.default_local_dir="$SCRATCH/checkpoint/emu3_bbox" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    "$@"
echo "Training completed!"

# --- Metrics extraction ---
# Strips ANSI color codes from SLURM output, keeps only step metric lines.
# Named with job ID so re-runs never overwrite. Delete stale logs manually.
# To plot: python slurm/plot.py slurm/logs/metrics_apertus_bbox_${SLURM_JOB_ID}.log
OUT_FILE="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
METRICS_LOG="${BASE}/slurm/logs/metrics_apertus_bbox_${SLURM_JOB_ID}.log"
mkdir -p "${BASE}/slurm/logs"
sed -r 's/\x1b\[[0-9;]*m//g' "$OUT_FILE" | grep -E "step:[0-9]+" > "$METRICS_LOG"
echo "Metrics saved to: $METRICS_LOG"
echo "To plot: python slurm/plot.py $METRICS_LOG"
