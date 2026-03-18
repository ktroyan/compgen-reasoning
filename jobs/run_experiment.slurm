#!/bin/bash
# run_experiment.slurm
# SLURM script to run a single experiment using main.py
# This file:
# - Loads environment and activates the venv
# - Sets SLURM options for GPU nodes
# - Runs the Python experiment with Hydra config
# - Designed to work with single or multiple GPUs
# #SBATCH --gres=gpu:1

# To run on RTX 3090 GPU:
# #SBATCH --gres=gpu:rtx3090:1

# To run on RTX 5090 GPU:
# #SBATCH -p rtx5090
# #SBATCH --gres=gpu:rtx5090:1

#SBATCH --job-name=compgen-experiment  # Job name
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out        # Standard output
#SBATCH --error=logs/%x_%j.err         # Standard error
#SBATCH -p rtx5090
#SBATCH --gres=gpu:rtx5090:1
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

# Set WANDB API key (better to do it like this here for SLURM)
export WANDB_API_KEY=$(cat ~/.secrets/wandb_api_key)

# Info
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES from SLURM: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# -----------------------
# Run Python experiment
# -----------------------
# Pass additional Hydra CLI overrides and launch SLURM job for the Python script
srun -u uv run main.py "$@"
