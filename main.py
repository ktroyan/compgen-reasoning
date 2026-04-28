"""
main.py

Entry point for running a single experiment or a sweep of experiments (e.g., over hyperparameters, over models, over experiments, etc.).
Built on PyTorch, PyTorch Lightning (PTL) 2.x, Hydra, OmegaConf, and WandB.

- Loads and resolves the experiment config via Hydra (defaults → sweep overrides → CLI overrides).
- Optionally sets up a WandB run and merges sweep parameters into the config.
- Instantiates the data module (GridDataModule) and model (TransformerModel or TRMModel).
- Dispatches to run_training and/or run_inference based on config flags.
- Saves the final resolved config locally and syncs it to WandB.
"""

import os
import hydra
import wandb
import torch
from typing import Type
from omegaconf import OmegaConf, DictConfig
from hydra.utils import get_original_cwd

## Personal imports
# Training and Inference entry points
from training import run_training
from inference import run_inference

# Data
from data.cogitao_data import GridDataModule  

# Models
from models.transformer_model import TransformerModel
from models.trm_model import TRMModel
from models.resnet_model import ResNetModel

# Utilities
from utility.wandb_utils import save_num_params_to_wandb, save_num_samples_to_wandb, setup_wandb
from utility.visualization_utils import visualize_datamodule_samples
from utility.logging_utils import logger

# PyTorch matmul precision (medium for better performance vs. precision)
torch.set_float32_matmul_precision("medium")

# Map config strings to actual Model classes
MODEL_MAP = {
    "resnet_model": ResNetModel,
    "transformer_model": TransformerModel,
    "trm_model": TRMModel
}

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------
    # Setup output folder
    # ------------------------------------------------------------------
    # Hydra has already loaded Defaults and applied CLI overrides into 'cfg'.
    # Hydra changes CWD to the run dir for this run
    run_dir = os.getcwd()   # get the timestamped run directory created by Hydra for this run
    orig_cwd = get_original_cwd()   # get the original working directory (where the code and configs are located, before Hydra changes it)
    
    logger.info(f"Run Directory (Output): {run_dir}")
    logger.info(f"Original Working Directory: {orig_cwd}")
        
    # ------------------------------------------------------------------
    # WandB Setup (+ config merging if in a sweep)
    # ------------------------------------------------------------------
    use_wandb = cfg.get("wandb", {}).get("enabled", False)
    
    # If WandB is enabled and we are in a WandB run (e.g., launched from a WandB sweep), set up the WandB logger and
    # merge sweep parameters into the config
    if use_wandb:
        cfg = setup_wandb(cfg, run_dir)

    # Log the final resolved config
    logger.info(f"Experiment Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ------------------------------------------------------------------
    # Instantiate Data module
    # ------------------------------------------------------------------

    orig_cwd = get_original_cwd()

    if cfg.data.name == "cogitao_data":
        dm = GridDataModule(cfg=cfg)
    else:
        raise ValueError(f"DataModule '{cfg.data}' not recognized. Please check the config defaults and ensure the corresponding DataModule is implemented and imported in main.py.")

    # Sanity check to ensure the DataModule can be set up without errors before proceeding to training/inference
    try:
        dm.setup(stage="fit")

        if use_wandb and wandb.run:
            save_num_samples_to_wandb(dm)
    
    except Exception as e:
        logger.error(f"DataModule setup failed. Cannot proceed: {e}")
        raise e

    ## Visualize some samples from the dataset for sanity checking
    visualize_datamodule_samples(dm, cfg)


    # ------------------------------------------------------------------
    # Instantiate Model module
    # ------------------------------------------------------------------
    model_name = cfg.get("model", {}).get("name", None)
    logger.info(f"Instantiating Model: {model_name}...")
    
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model '{model_name}' not found in MODEL_MAP. Available: {list(MODEL_MAP.keys())}")
    
    try:
        ModelClass = MODEL_MAP[model_name]
        model = ModelClass(cfg)
    except Exception as e:
        logger.error(f"Failed to initialize Model '{model_name}': {e}")
        raise e


    # ------------------------------------------------------------------
    # Training and/or Inference
    # ------------------------------------------------------------------

    # Check if running on GPU and log device info
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Using CPU.")
    
    # Check flags in the 'training' and 'inference' config sections
    do_train = cfg.get("experiment", {}).get("run_training", False)
    do_inference = cfg.get("experiment", {}).get("run_inference", False)

    if do_inference and not do_train:
        if not cfg.inference.get("checkpoint_path", None):
            raise ValueError(
                "Running inference without training requires 'inference.checkpoint_path' to be set in config."
            )
        
    if do_train:
        logger.info("--- Starting Training Phase ---")
        # NOTE: model is modified in-place by the training function to be the last training step model
        #       So the run_training() function returns the best model (w.r.t. the model selection criterion)
        model, _latest_model, _ckpt_dict = run_training(cfg, model, dm)
    else:
        logger.info("Skipping Training (cfg.experiment.run_training is False)")

    if do_inference:
        logger.info("--- Starting Inference Phase ---")
        model, _results = run_inference(cfg, model, dm)
    else:
        logger.info("Skipping Inference (cfg.experiment.run_inference is False)")

    # Save number of parameters after training/inference so the count reflects
    # the actual model used (important when loading from a checkpoint in inference-only mode)
    if use_wandb and wandb.run:
        save_num_params_to_wandb(model)


    # ------------------------------------------------------------------
    # Log and Save Results
    # ------------------------------------------------------------------
    
    # Log the final config (with initial resolution and parameters dynamically set)
    logger.info(f"Final Experiment Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # Save final config to the Hydra output directory
    with open("final_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    if use_wandb and wandb.run:
        # Sync the config file from the local Hydra output directory to WandB cloud
        wandb.save("final_config.yaml")
        wandb.finish()
        logger.info("WandB run finished.")

    logger.info(f"Experiment completed successfully. Outputs saved in: {run_dir}")

if __name__ == "__main__":
    main()