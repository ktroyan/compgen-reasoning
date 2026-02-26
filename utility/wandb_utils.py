"""
utility/wandb_utils.py

WandB utilities for logging experiments and saving artifacts

"""

import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger


# -------------------------------------------------------
# Initialization
# -------------------------------------------------------
def initialize_wandb(cfg):
    """
    Initialize WandB run if not already initialized.
    """

    if wandb.run is not None:
        return

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


# -------------------------------------------------------
# Sweep override
# -------------------------------------------------------
def apply_wandb_sweep_overrides(cfg):
    """
    Override Hydra config using WandB sweep parameters.
    """

    if wandb.run is None:
        return cfg

    sweep_config = dict(wandb.config)
    cfg = OmegaConf.merge(cfg, sweep_config)

    return cfg


# -------------------------------------------------------
# Lightning Logger
# -------------------------------------------------------
def build_wandb_logger(cfg):
    """
    Build PyTorch Lightning WandB logger.
    """

    return WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.run_name,
        save_dir=cfg.output_dir,
        log_model=True,
    )


# -------------------------------------------------------
# Artifact logging
# -------------------------------------------------------
def save_checkpoint_artifact(logger, checkpoint_path, artifact_name):
    """
    Save a checkpoint as a WandB artifact.
    """

    if logger is None:
        return

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
    )
    artifact.add_file(checkpoint_path)

    logger.experiment.log_artifact(artifact)
