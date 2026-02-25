#!/bin/bash
#SBATCH --job-name=tensorboard
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=normal
#SBATCH -A infra01
#SBATCH --output=tensorboard.out

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate verl

# Get node name
HOSTNAME=$(hostname)
echo "TensorBoard is running on: $HOSTNAME"
echo "Port: 6006"
echo ""
echo "To access from your local machine, run:"
echo "  ssh -L 6006:${HOSTNAME}:6006 ${USER}@clariden"
echo ""
echo "Then open: http://localhost:6006"

cd /users/$USER/verl-main/

# Launch TensorBoard
tensorboard --logdir=tensorboard_log \
    --port=6006 \
    --bind_all \
    --reload_interval=5