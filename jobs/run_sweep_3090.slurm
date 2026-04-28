#!/bin/bash
# run_sweep.slurm
# SLURM script to run one WandB sweep agent for a given sweep ID

# To run on RTX 3090 GPU:
# #SBATCH --gres=gpu:rtx3090:1
# #SBATCH --nodelist=node04                 # node04 has 8 3090 GPUs, 128 CPUs and 252GB RAM
# #SBATCH --cpus-per-task=16                # Number of CPU cores
# #SBATCH --mem=30G                         # Max memory


# To run on RTX 5090 GPU:
# #SBATCH -p rtx5090
# #SBATCH --gres=gpu:rtx5090:1
# #SBATCH --nodelist=node10              # node10 has 4 RTX5090 GPUs, 96 CPUs, and 382GB RAM
# #SBATCH --cpus-per-task=24             # Number of CPU cores
# #SBATCH --mem=92G                      # Max memory


#SBATCH --job-name=compgen-wandb_sweep  # Job name
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out        # Standard output
#SBATCH --error=logs/%x_%j.err         # Standard error
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --nodelist=node06                 # node04 has 8 3090 GPUs, 128 CPUs and 252GB RAM
#SBATCH --cpus-per-task=16                # Number of CPU cores
#SBATCH --mem=62G                         # Max memory
#SBATCH --ntasks=1
#SBATCH --time=96:00:00                # Max runtime (HH:MM:SS)
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
COUNT=$2  # optional: number of runs to execute (e.g. 1); if not specified, agent runs until the sweep is complete or the time limit is hit

# -----------------------
# Run WandB sweep agent (for the given sweep ID)
# -----------------------
if [ -n "$COUNT" ]; then
    srun -u wandb agent --count "$COUNT" "$SWEEP_ID"
else
    srun -u wandb agent "$SWEEP_ID"
fi