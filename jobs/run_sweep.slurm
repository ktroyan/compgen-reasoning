#!/bin/bash
# run_sweep.slurm
# SLURM script to run one WandB sweep agent for a given sweep ID

# To run on RTX 3090 GPU:
# #SBATCH --gres=gpu:rtx3090:1

# To run on RTX 5090 GPU:
# #SBATCH -p rtx5090
# #SBATCH --gres=gpu:rtx5090:1


#SBATCH --job-name=compgen-wandb_sweep  # Job name
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out        # Standard output
#SBATCH --error=logs/%x_%j.err         # Standard error
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64             # Number of CPU cores
#SBATCH --time=24:00:00                # Max runtime (HH:MM:SS)
#SBATCH --mem=128G                     # Max memory
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/ktroyan/compgen-reasoning

set -e
mkdir -p logs

# -----------------------
# Load environment
# -----------------------
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "No virtualenv found!"
    exit 1
fi

export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)

echo "Running on $(hostname)"
nvidia-smi || true

# Argument required is the sweep ID
if [ -z "$1" ]; then
    echo "Usage: sbatch run_sweep.slurm <username>/<project>/<sweep_id>"
    exit 1
fi

SWEEP_ID=$1

# -----------------------
# Run WandB sweep agent (for the given sweep ID)
# -----------------------
srun -u wandb agent "$SWEEP_ID"