#!/bin/bash
#SBATCH --job-name=convert_ckpt
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000
#SBATCH --time=0:15:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=%x.out
#SBATCH --error=%x.err

set -euo pipefail

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate verl

cd /users/$USER/verl-main
export PYTHONPATH="/users/$USER/verl-main:${PYTHONPATH:-}"

STEP="${1:-30}"
INPUT_DIR="/iopsstor/scratch/cscs/arichelle/checkpoint/bbox/global_step_${STEP}/actor"
OUTPUT_DIR="/iopsstor/scratch/cscs/arichelle/checkpoint/bbox/global_step_${STEP}_hf"

echo "Converting FSDP checkpoint to HuggingFace format"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$INPUT_DIR" \
    --target_dir "$OUTPUT_DIR"

echo "Conversion completed!"

# Fix config.json for SGLang compatibility
# SGLang doesn't recognize "qwen2_5_vl_text" model_type in text_config
echo "Patching config.json for SGLang compatibility..."
python3 << 'EOF'
import json
import sys

config_path = "$OUTPUT_DIR/config.json"
with open(config_path, 'r') as f:
    config = json.load(f)

modified = False

# Fix text_config model_type if it exists
if 'text_config' in config:
    if config['text_config'].get('model_type') == 'qwen2_5_vl_text':
        # Remove the problematic model_type from text_config
        # SGLang only recognizes "qwen2_5_vl" at the top level
        del config['text_config']['model_type']
        modified = True
        print(f"Removed 'model_type: qwen2_5_vl_text' from text_config")

if modified:
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config patched successfully: {config_path}")
else:
    print("No patching needed")
EOF

echo "Final checkpoint contents:"
ls -la "$OUTPUT_DIR"
