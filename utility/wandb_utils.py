"""
utility/wandb_utils.py

WandB utilities for logging experiments and saving artifacts

"""

import wandb
from omegaconf import OmegaConf, open_dict
from pytorch_lightning.loggers import WandbLogger

from utility.logging_utils import logger

# -------------------------------------------------------
# Parse data path to save the names of subfolders for convenient grouping and filtering in WandB
# -------------------------------------------------------
def parse_data_path_folders(data_path: str) -> dict:
    """Extract meaningful folder components from an HF dataset path as WandB config keys."""
    path = data_path.replace("hf://datasets/", "").lstrip("./")
    parts = [p for p in path.split("/") if p]
    # Skip HF entity and HF repo (e.g. COGITAO)
    field_for_wanbd_table = parts[2:] if len(parts) > 2 else parts
    
    return {f"subfolder-{i+1}": folder for i, folder in enumerate(field_for_wanbd_table)}


# -------------------------------------------------------
# Initialization + Sweep override (combined entry point)
# -------------------------------------------------------
def setup_wandb(cfg, run_dir: str):
    """
    Initialize a WandB run and overlay any sweep parameters onto cfg.

    Returns the (potentially updated) cfg.
    """
    wandb.init(
        project=cfg.wandb.get("project_name", "compgen-reasoning"),
        entity=cfg.wandb.get("entity_name", None),
        group=cfg.wandb.get("group", None),
        name=cfg.wandb.get("run_name", None),
        dir=run_dir,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
    )

    logger.info(f"WandB run URL: {wandb.run.url if wandb.run else 'No run!'}")

    # Log path folder metadata to run summary (not config) for grouping/filtering in WandB
    wandb.run.summary.update(parse_data_path_folders(cfg.data.get("data_path", "")))

    # Overlay sweep parameters if a sweep agent is running
    if wandb.config:
        logger.info("Updating config with WandB Sweep parameters...")
        with open_dict(cfg):
            for key, value in wandb.config.items():
                if "." in key:
                    parts = key.split(".")
                    sub_conf = cfg
                    for part in parts[:-1]:
                        if part not in sub_conf:
                            sub_conf[part] = {}
                        sub_conf = sub_conf[part]
                    sub_conf[parts[-1]] = value
                else:
                    cfg[key] = value

    return cfg

def save_num_samples_to_wandb(dm):
    """
    Save the number of samples in the training set to WandB run summary.
    """
    wandb.run.summary.update({"num_train_samples": len(dm.train_ds)})

def save_num_params_to_wandb(model):
    """
    Save the number of trainable parameters in the model to WandB run summary.
    """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.run.summary.update({"num_trainable_params": num_params})


# -------------------------------------------------------
# Lightning Logger
# -------------------------------------------------------
def build_wandb_logger(cfg):
    """
    Build PyTorch Lightning WandB logger.
    """

    return WandbLogger(
        project=cfg.wandb.project_name,
        entity=cfg.wandb.entity_name,
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
