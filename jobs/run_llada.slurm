#!/bin/bash
#SBATCH --job-name=compgen-experiment  # Job name
#SBATCH -D ./
#SBATCH --output=logs/%x_%j.out           # Standard output
#SBATCH --error=logs/%x_%j.err            # Standard error
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=16                # Number of CPU cores
#SBATCH --mem=64G                         # Max memory
#SBATCH --ntasks=1
#SBATCH --time=96:00:00                   # Max runtime (HH:MM:SS)
#SBATCH --export=ALL,SRC_DIR=/cluster/raid/data/yassine

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
# Experiment Configuration
# -----------------------
MODEL="llada_model"
ENCODER="llada_encoder"

# These can be overridden via --export when calling sbatch
SEED="${SEED:-1997}"
DATASET="${DATASET:-hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-1/experiment_1_3M}"

MAX_EPOCHS=20

# -----------------------
# Run Python experiment
# -----------------------
# Pass additional Hydra CLI overrides and launch SLURM job for the Python script
srun -u uv run main.py \
    seed="${SEED}" \
    model=${MODEL} \
    network/encoder=${ENCODER} \
    data.data_path="${DATASET}" \
    training.max_epochs="${MAX_EPOCHS}" \
    "$@"
