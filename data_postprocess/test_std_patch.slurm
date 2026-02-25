#!/bin/bash
#SBATCH --job-name=test_std_patch
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=0:10:00
#SBATCH --environment=verl_env
#SBATCH --partition=normal
#SBATCH --gpus-per-node=1
#SBATCH --output=test_std_patch.out
#SBATCH --error=test_std_patch.err

set -euo pipefail

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Environment setup
export PYTHONPATH="/users/$USER/verl-main:${PYTHONPATH:-}"

echo "==================================================================="
echo "Testing reward_std_patch"
echo "==================================================================="

cd /users/$USER/data_postprocess
python test_std_patch.py

echo "==================================================================="
echo "Test completed"
echo "==================================================================="
