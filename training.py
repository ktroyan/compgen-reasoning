"""
training.py

Contains run_training, which builds the PTL Trainer and runs trainer.fit.

- Configures loggers: CSV and WandB (if enabled).
- Configures callbacks: ID and OOD model checkpointing, optional early stopping,
  optional EMA, LR monitor, and a SLURM-friendly epoch summary logger.
- Builds the PTL Trainer with parameters from config (precision, gradient clipping, accumulation, etc.).
- Returns the best model by set metric, the latest model, and a dict of checkpoint paths.

"""

import os
import torch
import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from omegaconf import DictConfig

## Personal imports
# Utilities
from utility.logging_utils import logger
from utility.callbacks.ema_callback import EMACallback
from utility.callbacks.epoch_summary_callback import EpochSummaryCallback

# For flash / memory-efficient kernels when available (e.g., A100, RTX40xx, etc.), otherwise
# use math kernels for older GPUs.
torch.nn.attention.sdpa_kernel("auto")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def run_training(cfg: DictConfig, model: pl.LightningModule, datamodule: pl.LightningDataModule):

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    
    # Get the current working directory (set by Hydra)
    save_dir = os.getcwd()    

    # Loggers
    loggers = []

    # CSV Logger: Saves to {save_dir}/csv_logs
    csv_logger = CSVLogger(save_dir=save_dir, name="csv_logs")
    loggers.append(csv_logger)

    # WandB Logger: Saves to {save_dir}/wandb
    if cfg.get("wandb", {}).get("enabled", False) and wandb.run:
        wb_logger = WandbLogger(
            experiment=wandb.run,
            save_dir=save_dir,
            log_model="all" if cfg.wandb.get("log_model", False) else False
        )
        loggers.append(wb_logger)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    callbacks = []
    
    ## Checkpointing
    # Saves to {save_dir}/checkpoints
    ckpt_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # ID Checkpointing
    id_metric = cfg.training.get("id_metric", "val/id_acc")
    id_mode = cfg.training.get("id_metric_mode", "max")
    
    logger.info(f"Checkpointing ID based on: {id_metric} ({id_mode})")
    ckpt_callback_id = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-id-{epoch:02d}",
        monitor=id_metric,
        mode=id_mode,
        save_top_k=1,
        save_last=True,
        verbose=True,
    )
    callbacks.append(ckpt_callback_id)

    # OOD Checkpointing
    use_ood = cfg.data.get("use_ood_val", False)
    ckpt_callback_ood = None
    
    if use_ood:
        ood_metric = cfg.training.get("ood_metric", "val/ood_acc")
        ood_mode = cfg.training.get("ood_metric_mode", "max")
        logger.info(f"Checkpointing OOD based on: {ood_metric} ({ood_mode})")
        
        ckpt_callback_ood = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best-ood-{epoch:02d}",
            monitor=ood_metric,
            mode=ood_mode,
            save_top_k=1,
            save_last=False, 
            verbose=True,
        )
        callbacks.append(ckpt_callback_ood)

    ## Early Stopping
    if cfg.training.get("early_stopping", False):
        monitor_metric = cfg.training.get("monitor_metric", "val/id_loss")
        monitor_metric_mode = cfg.training.get("monitor_metric_mode", "min")
        logger.info(f"Early stopping on monitor metric: {monitor_metric} ({monitor_metric_mode})")

        patience = cfg.training.get("patience", 20)
        logger.info(f"Early stopping on {monitor_metric}, Patience: {patience}")
        callbacks.append(EarlyStopping(monitor=monitor_metric, patience=patience, mode=monitor_metric_mode, verbose=True))

    ## EMA
    if cfg.training.get("ema", {}).get("enabled", False):
        ema_decay = cfg.training.ema.get("decay", 0.999)
        ema_cpu_offload = cfg.training.ema.get("cpu_offload", True)
        logger.info(f"EMA enabled with decay={ema_decay}, cpu_offload={ema_cpu_offload}")
        callbacks.append(EMACallback(decay=ema_decay, cpu_offload=ema_cpu_offload))

    ## Learning Rate Monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch", log_momentum=True))

    ## Simple epoch summary (useful for SLURM / non-TTY log files)
    callbacks.append(EpochSummaryCallback())


    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer_args = {
        "default_root_dir": save_dir,
        "logger": loggers,
        "callbacks": callbacks,
        "max_epochs": cfg.training.get("max_epochs", 10),
        "precision": cfg.training.get("precision", "32"),
        "accelerator": cfg.training.get("accelerator", "auto"),
        "devices": cfg.training.get("devices", "auto"),
        # With manual optimization, gradient clipping is handled in training_step; passing it to the Trainer raises an error
        **({} if cfg.training.get("use_manual_optimization", False) else {"gradient_clip_val": cfg.training.get("gradient_clip_val", 1.0)}),
        "accumulate_grad_batches": cfg.training.get("accumulate_grad_batches", 1),
        "check_val_every_n_epoch": cfg.training.get("check_val_every_n_epoch", 1),
        # Disable the progress bar under SLURM (to not interfere with logging). SLURM_JOB_ID is always set for batch jobs.
        "enable_progress_bar": cfg.logging.get("use_progress_bar", False) and "SLURM_JOB_ID" not in os.environ,
        "log_every_n_steps": cfg.logging.get("log_every_n_steps", 5),
        "deterministic": False
    }
    
    trainer = pl.Trainer(**trainer_args)

    ## Model Fitting
    # [Optional] Resume from ckpt
    resume_ckpt = cfg.experiment.get("checkpoint_path", None)
    if resume_ckpt:
        logger.warning(f"Resuming training from checkpoint: {resume_ckpt}")
    
    logger.info("Starting Trainer.fit()...")
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt, weights_only=False)

    # ------------------------------------------------------------------
    # Results Handling
    # ------------------------------------------------------------------
    best_id_path = ckpt_callback_id.best_model_path
    last_path = ckpt_callback_id.last_model_path
    best_ood_path = ckpt_callback_ood.best_model_path if ckpt_callback_ood else None
    
    checkpoints_dict = {"best_id": best_id_path, "best_ood": best_ood_path, "last": last_path}
    logger.success(f"Training finished. Checkpoints: {checkpoints_dict}")

    if best_id_path and os.path.exists(best_id_path):
        logger.info(f"Loading best ID model from {best_id_path}")
        best_model = type(model).load_from_checkpoint(best_id_path, cfg=cfg, weights_only=False)
        latest_model = model
    
    else:
        best_model = model
        latest_model = model
        logger.warning("Best ID checkpoint not found. Returning the latest model from training.")

    # Log best scores
    wb_logger = next((lg for lg in trainer.loggers if isinstance(lg, WandbLogger)), None)
    if wb_logger:
        if ckpt_callback_id.best_model_score:
            wandb.summary["best_id_score"] = ckpt_callback_id.best_model_score.item()
        if ckpt_callback_ood and ckpt_callback_ood.best_model_score:
            wandb.summary["best_ood_score"] = ckpt_callback_ood.best_model_score.item()

    return best_model, latest_model, checkpoints_dict