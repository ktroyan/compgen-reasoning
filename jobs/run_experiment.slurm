#!/bin/bash
# run_experiment.slurm
# SLURM script to run a single experiment using main.py
# This file:
# - Loads environment and activates the venv
# - Sets SLURM options for GPU nodes
# - Runs the Python experiment with Hydra config

# To run on RTX 3090 GPU:
# #SBATCH --gres=gpu:rtx3090:1
# #SBATCH --nodelist=node04                 # node04 has 8 3090 GPUs, 128 CPUs and 252GB RAM
# #SBATCH --cpus-per-task=16                # Number of CPU cores
# #SBATCH --mem=62G                         # Max memory


# To run on RTX 5090 GPU:
# #SBATCH -p rtx5090
# #SBATCH --gres=gpu:rtx5090:1
# #SBATCH --nodelist=node10              # node10 has 4 RTX5090 GPUs, 96 CPUs, and 382GB RAM
# #SBATCH --cpus-per-task=24             # Number of CPU cores
# #SBATCH --mem=92G                      # Max memory


#SBATCH --job-name=compgen-experiment  # Job name
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out           # Standard output
#SBATCH --error=logs/%x_%j.err            # Standard error
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodelist=node04                 # node04 has 8 3090 GPUs, 128 CPUs and 252GB RAM
#SBATCH --cpus-per-task=16                # Number of CPU cores
#SBATCH --mem=62G                         # Max memory
#SBATCH --ntasks=1
#SBATCH --time=96:00:00                   # Max runtime (HH:MM:SS)
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
