#!/bin/bash

# cd to the project root (parent of jobs/)
cd "$(dirname "$0")/.." || exit 1

SEEDS=(1997 42 123)
# SEEDS=(1997)
# DATASETS=(
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-1/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-1/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-2/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-3/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-3/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-4/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-5/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-5/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_2-1/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_2-2/experiment_1_3M"
# )
# DATASETS=(
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-1/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-1/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-2/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-3/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-3/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-4/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-5/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_1-5/experiment_2_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_2-1/experiment_1_3M"
#     "hf://datasets/ktroyan/COGITAO/compgen/exp_setting_2-2/experiment_1_3M"
# )

DATASETS=("${DATASETS[@]}" "hf://datasets/yassinetb/COGITAO/CompGen/exp_setting_1/experiment_1")

echo "Launching ${#DATASETS[@]} datasets x ${#SEEDS[@]} seeds = $((${#DATASETS[@]} * ${#SEEDS[@]})) jobs"

for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Submitting: seed=$seed dataset=$dataset"
        SEED="$seed" DATASET="$dataset" sbatch jobs/run_llada.slurm
    done
done
