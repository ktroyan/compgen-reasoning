"""
inference.py
Performs model inference on a model checkpoint.

TODO:
1) Load model to be evaluated from the checkpoint path given if no model is directly provided as input.

2) Instantiate the PTL Trainer w.r.t. the config.

3) Given the model and test data modules received as input,
   perform inference, log results to WandB, and save results locally and to WandB.

"""

import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from omegaconf import DictConfig

# Custom Logger import
from utility.logging_utils import logger

def run_inference(cfg: DictConfig, model: pl.LightningModule, datamodule: pl.LightningDataModule):
    """
    Orchestrates the inference/testing process using PyTorch Lightning.

    """

    # Hydra sets the CWD to the timestamped output directory.
    # We use this directory to save logs and predictions.
    save_dir = os.getcwd()
    logger.info(f"Inference output directory: {save_dir}")

    # ------------------------------------------------------------------
    # 1) Load model from checkpoint (if explicitly provided)
    # ------------------------------------------------------------------
    # If "checkpoint_path" is set in config (e.g. via CLI or sweep), load that specific file.
    # Otherwise, assume 'model' already contains the weights (from the training phase).
    ckpt_path = cfg.get("inference", {}).get("checkpoint_path", None)
    
    if ckpt_path:
        if os.path.exists(ckpt_path):
            logger.info(f"Loading model weights from checkpoint: {ckpt_path}")
            # We use the class of the passed model object to load the checkpoint
            # weights_only=False is needed to load Hydra DictConfig params often saved in hparams
            model = type(model).load_from_checkpoint(ckpt_path, cfg=cfg, weights_only=False)
        else:
            logger.error(f"Checkpoint path provided but not found: {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        logger.info("No specific 'inference.checkpoint_path' provided. Using the current model state (e.g. from training).")

    # Set to evaluation mode
    model.eval()

    # ------------------------------------------------------------------
    # 2) Configure Loggers
    # ------------------------------------------------------------------
    loggers = []
    
    # CSV Logger: Saves metrics to {save_dir}/inference_logs/version_0/metrics.csv
    csv_logger = CSVLogger(save_dir=save_dir, name="inference_logs")
    loggers.append(csv_logger)
    
    # WandB Logger: Attach to the existing run if active
    if cfg.get("wandb", {}).get("enabled", False) and wandb.run:
        wb_logger = WandbLogger(experiment=wandb.run, save_dir=save_dir)
        loggers.append(wb_logger)

    # ------------------------------------------------------------------
    # 3) Instantiate the PTL Trainer
    # ------------------------------------------------------------------
    trainer_args = {
        "default_root_dir": save_dir,
        "accelerator": cfg.inference.get("accelerator", "auto"),
        "devices": cfg.inference.get("devices", 1),
        "logger": loggers,
        "enable_checkpointing": False,
        "enable_progress_bar": cfg.logging.get("use_progress_bar", False) and "SLURM_JOB_ID" not in os.environ,
    }

    trainer = pl.Trainer(**trainer_args)

    # ------------------------------------------------------------------
    # 4) Perform Inference (Test Loop)
    # ------------------------------------------------------------------
    logger.info("Starting Trainer.test()...")

    # Run the test loop. This triggers `test_step` and `on_test_epoch_end` in the model.
    # Returns a list of dictionaries (one dict per dataloader)
    results = trainer.test(model, datamodule=datamodule)

    # ------------------------------------------------------------------
    # 5) Display and Log Results
    # ------------------------------------------------------------------
    logger.success("--- Inference Results ---")
    
    if isinstance(results, list):
        for i, res in enumerate(results):
            # Logic assumes DataModule returns [ID_Loader, OOD_Loader] in that order
            eval_domain = "ID (In-Distribution)" if i == 0 else "OOD (Out-Of-Distribution)"
            logger.info(f"Dataset {i} [{eval_domain}]:")
            for k, v in res.items():
                logger.info(f"  {k:<20}: {v:.5f}")
    else:
        # Fallback for single dataloader
        logger.info(f"Results: {results}")

    # ------------------------------------------------------------------
    # 6) Artifact Management
    # ------------------------------------------------------------------
    # The model's `on_test_epoch_end` should have saved 'test_predictions.json' to CWD.
    prediction_file = os.path.join(save_dir, "test_predictions.json")
    
    if os.path.exists(prediction_file):
        logger.info(f"Found prediction file: {prediction_file}")
        if wandb.run:
            logger.info("Uploading predictions to WandB...")
            # base_path ensures it's uploaded to the root of the run in the cloud
            wandb.save(prediction_file, base_path=save_dir)
    else:
        logger.warning("No 'test_predictions.json' found. Ensure 'on_test_epoch_end' in your model saves it.")

    return results