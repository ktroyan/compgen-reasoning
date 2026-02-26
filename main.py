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
import logging
import hydra
import wandb
import torch
from typing import Type
from omegaconf import OmegaConf, DictConfig, open_dict
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig 

## Personal imports
# Importing the run functions
from training import run_training
from inference import run_inference

# Importing DataModule
from data.data import GridDataModule  

# Importing Models
from models.transformer_model import TransformerModel
from models.trm_model import TRMModel

# Importing Utilities
from utility.visualization_utils import visualize_data_samples

# Setup Logger
logger = logging.getLogger(__name__)

# Map config strings to actual Model classes
MODEL_MAP = {
    "transformer_model": TransformerModel,
    "trm_model": TRMModel,
}

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):

    # ------------------------------------------------------------------
    # 0) Setup Output Directories
    # ------------------------------------------------------------------
    # Hydra has already loaded Defaults and applied CLI overrides into 'cfg'.
    # Hydra changes CWD to the run dir. We can also get it explicitly.
    # Because 'hydra.job.chdir: True', os.getcwd() IS the timestamped folder.
    run_dir = os.getcwd()
    orig_cwd = get_original_cwd()
    
    logger.info(f"Run Directory (Output): {run_dir}")
    logger.info(f"Original Directory: {orig_cwd}")
        
    # ------------------------------------------------------------------
    # 1) & 2) Configuration & WandB Setup
    # ------------------------------------------------------------------
    use_wandb = cfg.get("wandb", {}).get("enabled", False)    # check if WandB is enabled in the config
    
    if use_wandb:
        # Initialize WandB
        # If this is part of a Sweep, WandB automatically picks up the sweep config
        # and ignores the 'config' argument passed here for those specific keys.
        wandb.init(
            project=cfg.wandb.get("project_name", "compgen-reasoning-project"),
            entity=cfg.wandb.get("entity_name", None),
            group=cfg.wandb.get("group", None),
            name=cfg.wandb.get("run_name", None),
            dir=run_dir,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)    # pass the full Hydra config as an initial state
        )
    
        # (ii) Update Hydra config with WandB Sweep values
        # If a sweep is running, wandb.config contains the hyperparams chosen by the agent.
        # We overlay these onto the Hydra config so the rest of the code uses the sweep values.
        if wandb.config:
            logger.info("Updating config with WandB Sweep parameters...")
            with open_dict(cfg):
                for key, value in wandb.config.items():
                    # Handle nested keys if WandB uses dot notation (e.g. "model.n_layers")
                    if "." in key:
                        parts = key.split(".")
                        sub_conf = cfg
                        for part in parts[:-1]:
                            # Create nested dict if it doesn't exist
                            if part not in sub_conf:
                                sub_conf[part] = {}
                            sub_conf = sub_conf[part]
                        sub_conf[parts[-1]] = value
                    else:
                        # Top level keys
                        cfg[key] = value

    # Log the final resolved config
    logger.info(f"Final Experiment Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # ------------------------------------------------------------------
    # 3) Instantiate Data module
    # ------------------------------------------------------------------

    orig_cwd = get_original_cwd()
    logger.info("Instantiating Data Module (GridDataModule)...")
    
    dm = GridDataModule(cfg=cfg)

    # Just a sanity check to ensure the DataModule can be set up without errors before proceeding to visualization and training
    try:
        dm.setup(stage="fit")
    except Exception as e:
        logger.error(f"DataModule setup failed. Cannot proceed: {e}")
        raise e

    ## 3.1 Visualize some samples from the dataset for sanity checking
    if cfg.get("logging", {}).get("visualize_data_samples", False):
        logger.info("Generating data sample visualizations...")
        
        try:
            # Ensure setup is called (loads data into the DM)
            dm.setup(stage="fit")
            dm.setup(stage="test") 
            
            indices_to_check = [0, 1, 2] # Visualize first 3 samples

            # Helper function to handle Single vs List of Dataloaders
            def safe_visualize(loaders, phase_name):
                # If it's a single dataloader, we wrap it in a list for consistent processing
                if not isinstance(loaders, list):
                    loaders = [loaders]
                
                for i, loader in enumerate(loaders):
                    # Determine a suffix. 
                    # Convention in DataModule: 0=ID, 1=OOD
                    if len(loaders) > 1:
                        suffix = "ID" if i == 0 else "OOD"
                        filename = f"vis_{phase_name}_{i}_{suffix}.png"
                    else:
                        filename = f"vis_{phase_name}.png"
                    
                    # Check if loader is valid (might be None in some configs)
                    if loader is not None:
                        full_path = os.path.join(os.getcwd(), filename)
                        logger.info(f"Visualizing {phase_name} (Loader {i}) to {filename}")
                        visualize_data_samples(loader, indices_to_check, save_path=full_path)
                        
                        # Optional: Log to WandB
                        if wandb.run:
                            wandb.log({f"vis_{phase_name}_{i}": wandb.Image(full_path)})

            # Visualize Train
            safe_visualize(dm.train_dataloader(), "train")

            # Visualize Validation (Handles ID & OOD)
            safe_visualize(dm.val_dataloader(), "val")

            # Visualize Test (Handles ID & OOD)
            safe_visualize(dm.test_dataloader(), "test")

        except Exception as e:
            logger.warning(f"Could not visualize samples: {e}")
            # Print stack trace for debugging if needed
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # 3) Instantiate Model module
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
    # 4) Run Training and/or Inference
    # ------------------------------------------------------------------

    # Check if running on GPU and log device info
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available. Using CPU.")
    
    # Check flags in the 'training' and 'inference' config sections
    do_train = cfg.get("experiment", {}).get("run_training", False)
    do_inference = cfg.get("experiment", {}).get("run_inference", False)

    if do_train:
        logger.info("--- Starting Training Phase ---")
        run_training(cfg, model, dm)
    else:
        logger.info("Skipping Training (cfg.experiment.run_training is False)")

    if do_inference:
        logger.info("--- Starting Inference Phase ---")
        # Note: If training just finished, the model object already contains the trained weights.
        # If running inference-only, the run_inference function should handle checkpoint loading.
        run_inference(cfg, model, dm)
    else:
        logger.info("Skipping Inference (cfg.experiment.run_inference is False)")

    # ------------------------------------------------------------------
    # 5) Log and Save Results
    # ------------------------------------------------------------------
    
    # Save final config to the Hydra output directory
    with open("final_config.yaml", "w") as f:
        OmegaConf.save(cfg, f)
    
    if use_wandb and wandb.run:
        # Sync the config file from the local Hydra output directory to WandB cloud
        wandb.save("final_config.yaml")
        
        # WandB automatically captures stdout/stderr. 
        # Checkpoints are usually handled by PTL ModelCheckpoint callback (logged to wandb if configured).
        wandb.finish()
        logger.info("WandB run finished.")

    logger.info(f"Experiment completed successfully. Outputs saved in: {run_dir}")

if __name__ == "__main__":
    main()