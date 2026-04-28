"""
models/model_helpers.py

"""

import math
import os
from typing import Dict, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig

## Personal imports
from utility.logging_utils import logger

try:    # to avoid the program to crash if wandb is not installed and user doesn't want to use it
    import wandb
except ImportError:
    wandb = None

# ------------------------------------------------------------------
# Network Factory
# ------------------------------------------------------------------
_NETWORK_REGISTRY = {
    "trm_encoder":         ("networks.trm_encoder",         "TRMEncoder"),
    "trm_decoder":         ("networks.trm_decoder",         "TRMDecoder"),
    "transformer_encoder": ("networks.transformer_encoder", "TransformerEncoder"),
    "resnet_encoder":      ("networks.resnet_encoder",      "ResNetEncoder"),
    "mlp_decoder":         ("networks.mlp_decoder",         "MLPDecoder"),
}

def build_network(name: str, cfg: DictConfig) -> nn.Module:
    """ Instantiate a network by its registry name (as specified in the model config). """
    
    if name not in _NETWORK_REGISTRY:
        raise ValueError(f"Unknown network '{name}'. Registered networks: {list(_NETWORK_REGISTRY)}")
    module_path, class_name = _NETWORK_REGISTRY[name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(cfg)

def maybe_compile(module: nn.Module, cfg: DictConfig) -> nn.Module:
    """ Wrap a module with torch.compile if use_torch_compile is set in the training config. """
    
    if cfg.training.get("use_torch_compile", False):
        logger.info(f"Compiling {type(module).__name__} with torch.compile...")
        return torch.compile(module, fullgraph=False)
    return module


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, pad_id: int, per_sample: bool = False) -> Dict[str, torch.Tensor]:
    """
    Compute base metrics.

    Args:
        preds:      [B, S] predicted token ids
        targets:    [B, S] ground truth token ids
        pad_id:     token id used for padding (ignored in no-pad metrics)
        per_sample: if True, return [B] tensors; if False (default), return scalars averaged across samples

    NOTE: We use macro-averaged metrics (average over samples) rather than micro-averaged (overall token accuracy for the batch) as it is more informative for object metrics and grid metrics.
          For example, this is important for the grid accuracy where one sample with a single token error would disproportionately affect the overall metric if micro-averaged.
    """
    correct = (preds == targets)   # [B, S]

    acc             = correct.float().mean(dim=1)                                                          # [B]
    grid_acc        = correct.all(dim=1).float()                                                           # [B]
    non_pad         = targets != pad_id                                                                    # [B, S]
    acc_no_pad      = (correct & non_pad).sum(dim=1).float() / non_pad.sum(dim=1).float().clamp(min=1)    # [B]
    grid_acc_no_pad = ((correct | ~non_pad).all(dim=1)).float()                                            # [B]
    obj_mask        = (targets >= 1) & (targets <= 9)                                                     # [B, S]
    obj_acc         = (correct & obj_mask).sum(dim=1).float() / obj_mask.sum(dim=1).float().clamp(min=1)  # [B]

    result = {
        "acc": acc,
        "acc_no_pad": acc_no_pad,
        "grid_acc": grid_acc,
        "grid_acc_no_pad": grid_acc_no_pad,
        "obj_acc": obj_acc,
    }

    if per_sample:
        return result
    
    return {k: v.mean() for k, v in result.items()}


# ------------------------------------------------------------------
# StableMax (StCE loss from paper "Grokking at the Edge of Numerical Stability")
# ------------------------------------------------------------------
def _s(x: torch.Tensor) -> torch.Tensor:
    """ Stable ramp function: s(x) = x+1 if x>=0, else 1/(1-x). Always > 0. """
    return torch.where(x >= 0, x + 1.0, 1.0 / (1.0 - x))


def stce_loss(logits_flat: torch.Tensor, targets_flat: torch.Tensor, pad_id: int, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    StableMax Cross-Entropy: -log(StableMax(z_y)) = log(sum_j s(z_j)) - log(s(z_y)).

    L_StCE = -log(StableMax(z_y))
       = -log( s(z_y) / Σ_j s(z_j) )    # expand definition from eq. (5) of the paper
       = -log(s(z_y)) + log(Σ_j s(z_j)) # log of a ratio is the difference of logs
       = log(Σ_j s(z_j)) - log(s(z_y))  # what we actually use since we do not want to materialize the full prob. distr. (since in s(z_j) for all j, only s(z_y) is needed)

    Returns [B*S] with 0.0 at padding positions.

    """
    # Compute the stable ramp function for all logits and the log of the sum across classes
    sx = _s(logits_flat)                                                # [B*S, C]
    log_sum_sx = torch.log(sx.sum(dim=-1))                              # [B*S]
    
    # Gather the s(z_y) values for the true classes using advanced indexing
    idx = torch.arange(len(targets_flat), device=logits_flat.device)
    true_sx = sx[idx, targets_flat.clamp(min=0)]
    log_true_sx = torch.log(true_sx.clamp(min=1e-12))                   # [B*S]
    
    # Difference between log_sum_sx and log_true_sx, which corresponds to -log(StableMax(z_y))
    loss = log_sum_sx - log_true_sx                                     # [B*S]
    
    # Apply class weights if provided (only on non-padding positions)
    if weight is not None:
        loss = loss * weight[targets_flat.clamp(min=0)]  # scale by class weight, same as F.cross_entropy
    
    # Mask out padding positions (where targets_flat == pad_id) by zeroing out the loss there
    loss = loss * (targets_flat != pad_id).float()                      # zero out padding positions
    
    return loss

# ------------------------------------------------------------------
# Truncated Normal Initialization (as per JAX since the PyTorch version is not mathematically correct)
# Used for initialization of z latent state in TRM code.
# See https://github.com/olivkoch/nano-trm/blob/main/src/nn/modules/utils.py#L13
# ------------------------------------------------------------------
def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    # NOTE: PyTorch nn.init.trunc_normal_ is not mathematically correct, the std dev is not actually the std dev of initialized tensor
    # This function is a PyTorch version of jax truncated normal init (default init method in flax)
    # https://github.com/jax-ml/jax/blob/main/jax/_src/random.py#L807-L848
    # https://github.com/jax-ml/jax/blob/main/jax/_src/nn/initializers.py#L162-L199

    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2

            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower**2)
            pdf_l = c * math.exp(-0.5 * upper**2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2
            )

            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)

    return tensor

# ------------------------------------------------------------------
# Extract Evolution Samples (useful for analysis)
# ------------------------------------------------------------------
def _extract_evolution_samples(model_module, src, tgt, preds):
    records =[]
    for idx in model_module.evol_indices:
        if idx < len(src):
            # decode_sample already calls .detach().cpu().tolist(), converting to string safely
            records.append({
                "sample_idx": idx,
                "input_raw": src[idx].detach().cpu().tolist(),
                "target_raw": tgt[idx].detach().cpu().tolist(),
                "prediction_raw": preds[idx].detach().cpu().tolist(),
                "input": model_module.decode_sample(src[idx]),
                "target": model_module.decode_sample(tgt[idx]),
                "prediction": model_module.decode_sample(preds[idx])
            })

    return records


# ------------------------------------------------------------------
# Plot Evolution Grids (useful for analysis)
# ------------------------------------------------------------------
def _plot_epoch_grids(model_module, phase_name: str, epoch: int, records: list):
    """ 
    NOTE: Generated this function with AI; see for review.
    """

    if not model_module.visualize_predictions or not records:
        return
        
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        from utility.visualization_utils import get_arc_colormap
    
    except ImportError as e:
        logger.warning(f"Skipping prediction grid visualization. Missing dependency: {e}")
        return

    h = model_module.cfg.model.max_h
    w = model_module.cfg.model.max_w
    use_tasks = model_module.cfg.model.get("use_task_tokens", False)
    max_task_len = model_module.cfg.model.get("max_task_seq_len", 0) if use_tasks else 0
    
    # Calculate where the grid tokens start in the source sequence
    src_start_idx = 1 + max_task_len

    n_samples = len(records)
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = [axes] # ensure iterable

    # Get the standard ARC colormap
    try:
        cmap, norm = get_arc_colormap()
    except Exception as e:
        logger.warning(f"Could not load ARC colormap, falling back to tab20: {e}")
        cmap = plt.get_cmap('tab20')
        norm = mcolors.Normalize(vmin=0, vmax=19)

    def plot_pretty_grid(ax, data, title):
        """ Helper to plot a single grid cleanly with annotations. """
        # Clip data to max 13 (UNK) to prevent colormap crashes if the model 
        # accidentally predicts a task token ID or OOD token.
        disp_data = np.clip(data, 0, 13)
        
        ax.imshow(disp_data, cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=12, pad=10)
        
        # Create gridlines between pixels
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        
        # Hide ticks and tick labels
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        
        # Annotate numbers
        if data.size < 400:
            for (j, i), label in np.ndenumerate(data):
                # Only explicitly color standard tokens, fallback to black for weird tokens
                text_color = 'white' if label in [0, 1, 9, 13] else 'black'
                ax.text(i, j, str(label), ha='center', va='center', color=text_color, fontsize=8)

    for i, record in enumerate(records):
        src_raw = record["input_raw"]
        tgt_raw = record["target_raw"]
        pred_raw = record["prediction_raw"]

        # Safely extract the grid tokens
        # src_raw structure: BOS (1) + Tasks (K) + Grid (h*w) + EOS (1)
        src_grid_1d = src_raw[src_start_idx : src_start_idx + (h * w)]
        
        # tgt and pred are typically just the grid (shape S_grid)
        tgt_grid_1d = tgt_raw[: (h * w)]
        pred_grid_1d = pred_raw[: (h * w)]

        # Safety pad with background (0) if somehow shorter
        if len(src_grid_1d) < h * w:
            src_grid_1d += [0] * (h * w - len(src_grid_1d))
        if len(tgt_grid_1d) < h * w:
            tgt_grid_1d += [0] * (h * w - len(tgt_grid_1d))
        if len(pred_grid_1d) < h * w:
            pred_grid_1d += [0] * (h * w - len(pred_grid_1d))

        src_grid = np.array(src_grid_1d).reshape(h, w)
        tgt_grid = np.array(tgt_grid_1d).reshape(h, w)
        pred_grid = np.array(pred_grid_1d).reshape(h, w)

        ax_row = axes[i]
        
        plot_pretty_grid(ax_row[0], src_grid, f"Input {record['sample_idx']}")
        plot_pretty_grid(ax_row[1], tgt_grid, f"Target {record['sample_idx']}")
        plot_pretty_grid(ax_row[2], pred_grid, f"Prediction {record['sample_idx']}")

    plt.tight_layout()
    
    # Save Dir inside Hydra's run output
    save_dir = os.path.join(os.getcwd(), "prediction_grids")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{phase_name}_epoch_{epoch}.png"
    filepath = os.path.join(save_dir, filename)
    
    plt.savefig(filepath, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    # Log to wandb if enabled
    if model_module.cfg.get("wandb", {}).get("enabled", False) and wandb is not None and wandb.run is not None:
        wandb.log({f"grids/{phase_name}": wandb.Image(filepath), "epoch": epoch}, commit=False)


# ------------------------------------------------------------------
# Plot Metrics History
# ------------------------------------------------------------------
def _plot_metrics(model_module, save_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed. Skipping metric plots generation.")
        return

    if not model_module.epoch_metrics:
        return

    epochs = [m["epoch"] for m in model_module.epoch_metrics]
    
    # Identify all unique metric keys
    all_keys = set()
    for m in model_module.epoch_metrics:
        all_keys.update(m.keys())
    
    all_keys.discard("epoch")

    # Helper function to extract base metric name
    def get_base(key_str):
        return key_str.replace("train/", "").replace("val/id_", "").replace("val/ood_", "")

    # Group metrics into sub-categories by stripping prefixes
    bases = {get_base(k) for k in all_keys}

    for base in bases:
        plt.figure(figsize=(10, 6))
        plotted = False
        
        # Find all keys that map strictly to this base
        for key in sorted(all_keys):
            if get_base(key) == base:
                values = [m.get(key, None) for m in model_module.epoch_metrics]
                
                # Filter missing values
                valid_epochs =[e for e, v in zip(epochs, values) if v is not None]
                valid_values =[v for v in values if v is not None]
                
                if valid_values:
                    plt.plot(valid_epochs, valid_values, marker='o', label=key)
                    plotted = True
                    
        if plotted:
            plt.title(f"Evolution of {base}")
            plt.xlabel("Epoch")
            plt.ylabel(base)
            plt.legend()
            plt.grid(True)
            safe_base = base.replace("/", "_").replace("\\", "_")
            plot_path = os.path.join(save_dir, f"plot_{safe_base}.png")
            plt.savefig(plot_path, bbox_inches='tight')
            logger.info(f"Saved metric plot to {plot_path}")

        plt.close()