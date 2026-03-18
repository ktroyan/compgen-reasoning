"""
main.py
Main script serving as an entrypoint to run an experiment or a sweep of experiments.
It is based on PyTorch, PyTorch Lightning (PTL) 2.x, Hydra, OmegaConf, WandB. 

TODO:
1) Set the exact config to be used for the experiment, updating config parameters in this order:
   (i) default values using Hydra; (ii) WandB sweep values; (iii) CLI values.

2) Initialize WandB with the final config, setting up the project name, experiment name, and other relevant parameters.

3) Instanciate and initialize the data module and model based on the final config.

4) Run training and/or inference, logging results to WandB. 
   The PTL Trainer is built in training.py and inference.py.

5) Log and save results, checkpoints, and config of the experiment locally (/outputs folder) and to WandB.

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

# WandB utilities
from utility.wandb_utils import setup_wandb

# Data
from data.cogitao_data import GridDataModule  

# Models
from models.transformer_model import TransformerModel
from models.trm_model import TRMModel

# Utilities
from utility.visualization_utils import visualize_datamodule_samples

# Setup Logger
from utility.logging_utils import logger

# Set PyTorch matmul precision to medium for better performance vs. precision
torch.set_float32_matmul_precision("medium")

# Map config strings to actual Model classes
MODEL_MAP = {
    "transformer_model": TransformerModel,
    "trm_model": TRMModel,
}

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------
    # 0) Setup output folder
    # ------------------------------------------------------------------
    # Hydra has already loaded Defaults and applied CLI overrides into 'cfg'.
    # Hydra changes CWD to the run dir for this run
    run_dir = os.getcwd()   # get the timestamped run directory created by Hydra for this run
    orig_cwd = get_original_cwd()   # get the original working directory (where the code and configs are located, before Hydra changes it)
    
    logger.info(f"Run Directory (Output): {run_dir}")
    logger.info(f"Original Directory: {orig_cwd}")
        
    # ------------------------------------------------------------------
    # 1) & 2) Configuration & WandB Setup
    # ------------------------------------------------------------------
    use_wandb = cfg.get("wandb", {}).get("enabled", False)    # check if WandB is enabled in the config
    
    if use_wandb:
        cfg = setup_wandb(cfg, run_dir)

    # Log the final resolved config
    logger.info(f"Final Experiment Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ------------------------------------------------------------------
    # 3) Instantiate Data module
    # ------------------------------------------------------------------

    orig_cwd = get_original_cwd()

    if cfg.data.name == "cogitao_data":
        dm = GridDataModule(cfg=cfg)
    else:
        raise ValueError(f"DataModule '{cfg.data}' not recognized. Please check the config defaults and ensure the corresponding DataModule is implemented and imported in main.py.")

    # Just a sanity check to ensure the DataModule can be set up without errors before proceeding to visualization and training
    try:
        dm.setup(stage="fit")
    except Exception as e:
        logger.error(f"DataModule setup failed. Cannot proceed: {e}")
        raise e

    ## 3.1 Visualize some samples from the dataset for sanity checking
    visualize_datamodule_samples(dm, cfg)

    # ------------------------------------------------------------------
    # 4) Instantiate Model module
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
    # 5) Run Training and/or Inference
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
        run_inference(cfg, model, dm)
    else:
        logger.info("Skipping Inference (cfg.experiment.run_inference is False)")

    # ------------------------------------------------------------------
    # 6) Log and Save Results
    # ------------------------------------------------------------------
    
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