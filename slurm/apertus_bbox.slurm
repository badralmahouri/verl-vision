#!/bin/bash
#SBATCH --job-name=verl_image_bbox_tool
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=460000
#SBATCH --time=12:00:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=4
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

# Emu3.5 src needed for IBQ vision tokenizer (build_vision_tokenizer)
EMU3_SRC="/users/$USER/Emu3.5/src"
export PYTHONPATH="${EMU3_SRC}:/users/$USER/verl-vision/data_postprocess:/users/$USER/verl-vision:${PYTHONPATH:-}"
export RAY_worker_env_PYTHONPATH="$PYTHONPATH"

# IBQ vision tokenizer for discrete image encoding
VQ_MODEL_PATH="/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/Emu3.5-VisionTokenizer"
export VQ_MODEL_PATH
export RAY_worker_env_VQ_MODEL_PATH="$VQ_MODEL_PATH"

export PYTHONUNBUFFERED=1

# Persistent per-sample reward debug log (survives after job ends)
export REWARD_DEBUG_LOG="/users/$USER/verl-vision/slurm/logs/reward_debug_${SLURM_JOB_ID}.jsonl"
export RAY_worker_env_REWARD_DEBUG_LOG="$REWARD_DEBUG_LOG"

# Clean stale torch extension caches that cause import hangs on GH200 (SD-68483)
rm -rf ~/.cache/torch_extensions/ 2>/dev/null || true

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --list-gpus
ulimit -n 65535
BASE="/users/$USER/verl-vision"

# Modified template: model's native format (developer_start + TypeScript) with added
# explicit tool-call format instructions so the model knows to use <|tools_prefix|>
CHAT_TEMPLATE_PATH="$BASE/examples/sglang_multiturn/config/apertus_tool_chat_template.jinja2"
export VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"
export RAY_worker_env_VERL_CUSTOM_CHAT_TEMPLATE_PATH="$CHAT_TEMPLATE_PATH"

# configs
CONFIG_PATH="$BASE/examples/sglang_multiturn/config"
TOOL_CONFIG_PATH="$BASE/examples/sglang_multiturn/config/bbox_only_tool_config.yaml"

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
APERTUS_CHECKPOINT="${APERTUS_CHECKPOINT:-/capstor/store/cscs/swissai/infra01/MLLM/ablations/apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-seqlen8192-s2onlytxtloss/HF}"
APERTUS_TOKENIZER="${APERTUS_TOKENIZER:-$APERTUS_CHECKPOINT}"

echo "=== tokenizer checks ==="
if [ -d "$APERTUS_CHECKPOINT" ]; then
    echo "✓ Apertus checkpoint dir: $APERTUS_CHECKPOINT"
    # Verify key files exist in checkpoint
    for f in config.json; do
        if [ -f "$APERTUS_CHECKPOINT/$f" ]; then
            echo "  ✓ $f found"
        else
            echo "  ✗ $f NOT found in checkpoint — model loading may fail"
            ((ERROR_COUNT++))
        fi
    done
else
    echo "✗ Apertus checkpoint NOT found: $APERTUS_CHECKPOINT"
    echo "  Set APERTUS_CHECKPOINT env var or edit this script."
    ((ERROR_COUNT++))
fi

# Check tokenizer (Apertus uses standard HF tokenizer bundled in checkpoint)
if [ -d "$APERTUS_TOKENIZER" ]; then
    echo "✓ Tokenizer dir: $APERTUS_TOKENIZER"
    for f in tokenizer_config.json; do
        if [ -f "$APERTUS_TOKENIZER/$f" ]; then
            echo "  ✓ $f found"
        else
            echo "  ✗ $f NOT found — tokenizer loading may fail"
            ((ERROR_COUNT++))
        fi
    done
else
    echo "✗ Tokenizer NOT found: $APERTUS_TOKENIZER"
    ((ERROR_COUNT++))
fi

# IBQ vision tokenizer check
if [ -d "$VQ_MODEL_PATH" ]; then
    echo "✓ VQ model dir: $VQ_MODEL_PATH"
else
    echo "✗ VQ model NOT found: $VQ_MODEL_PATH"
    echo "  IBQ vision tokenizer needed for discrete image encoding."
    ((ERROR_COUNT++))
fi

# Emu3.5 source check (for vision_tokenizer module)
if [ -d "$EMU3_SRC/vision_tokenizer" ]; then
    echo "✓ Emu3.5 vision_tokenizer module: $EMU3_SRC/vision_tokenizer"
else
    echo "✗ Emu3.5 vision_tokenizer NOT found: $EMU3_SRC/vision_tokenizer"
    echo "  Clone https://github.com/baaivision/Emu3.5 to /users/$USER/Emu3.5"
    ((ERROR_COUNT++))
fi

# Chat template check
if [ -f "$CHAT_TEMPLATE_PATH" ]; then
    echo "✓ Chat template: $CHAT_TEMPLATE_PATH"
else
    echo "✗ Chat template NOT found: $CHAT_TEMPLATE_PATH"
    ((ERROR_COUNT++))
fi

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "ERROR: $ERROR_COUNT pre-flight checks failed. Fix the above and retry."
    exit 1
fi

echo ""
echo "=== Training Configuration ==="
echo "  Apertus checkpoint : $APERTUS_CHECKPOINT"
echo "  Apertus tokenizer  : $APERTUS_TOKENIZER"
echo "  VQ model (IBQ)     : $VQ_MODEL_PATH"
echo "  Train data         : $TRAIN_FILE"
echo "  Val data           : $VAL_FILE"
echo "  Tool config        : $TOOL_CONFIG_PATH"
echo "  Chat template      : $CHAT_TEMPLATE_PATH"
echo "==================================================================="
echo ""

# --- Run training ---
python3 /users/$USER/verl-vision/data_postprocess/verl_metrics.py \
    --config-path="$CONFIG_PATH" \
    --config-name='emu3_bbox_grpo' \
    data.prompt_key='prompt' \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    actor_rollout_ref.model.path="$APERTUS_CHECKPOINT" \
    actor_rollout_ref.model.tokenizer_path="$APERTUS_CHECKPOINT" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.rollout.trace.backend=tensorboard \
    trainer.default_local_dir="$SCRATCH/checkpoint/apertus_bbox_v3" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    trainer.total_training_steps=50 \
    "$@"
echo "Training completed!"

# --- Metrics extraction ---
# Strips ANSI color codes from SLURM output, keeps only step metric lines.
# Named with job ID so re-runs never overwrite. Delete stale logs manually.
# To plot: python slurm/plot.py slurm/logs/metrics_apertus_bbox_${SLURM_JOB_ID}.log
OUT_FILE="${SLURM_SUBMIT_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out"
METRICS_LOG="${BASE}/slurm/logs/apertus_bbox_${SLURM_JOB_ID}.log"
mkdir -p "${BASE}/slurm/logs"
sed -r 's/\x1b\[[0-9;]*m//g' "$OUT_FILE" | grep -E "step:[0-9]+" > "$METRICS_LOG"
echo "Metrics saved to: $METRICS_LOG"

# --- Auto-generate plots ---
PLOT_DIR="${BASE}/slurm/plots/apertus_bbox_${SLURM_JOB_ID}"
echo "Generating plots to: $PLOT_DIR"
python3 "${BASE}/slurm/plot.py" "$METRICS_LOG" "$PLOT_DIR" || echo "WARNING: Plot generation failed"
echo "Plots saved to: $PLOT_DIR"
echo "To re-plot: python slurm/plot.py $METRICS_LOG $PLOT_DIR"