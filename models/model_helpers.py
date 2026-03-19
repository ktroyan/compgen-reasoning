"""
models/model_helpers.py

"""

import os

try:    # to avoid the program to crash if wandb is not installed and user doesn't want to use it
    import wandb
except ImportError:
    wandb = None

from utility.logging_utils import logger

# ------------------------------------------------------------------
# Extract Evolution Samples
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
# Helper: Plot Evolution Grids
# ------------------------------------------------------------------
def _plot_epoch_grids(model_module, phase_name: str, epoch: int, records: list):
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
    use_tasks = model_module.cfg.model.get("use_task_tokens", True)
    max_task_len = model_module.cfg.model.get("max_task_seq_len", 0) if use_tasks else 0
    
    # Calculate exactly where the grid tokens start in the source sequence
    src_start_idx = 1 + max_task_len

    n_samples = len(records)
    fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3 * n_samples))
    if n_samples == 1:
        axes = [axes] # Ensure iterable

    # Get the standard ARC colormap
    try:
        cmap, norm = get_arc_colormap()
    except Exception as e:
        logger.warning(f"Could not load ARC colormap, falling back to tab20: {e}")
        cmap = plt.get_cmap('tab20')
        norm = mcolors.Normalize(vmin=0, vmax=19)

    def plot_pretty_grid(ax, data, title):
        """Helper to plot a single grid cleanly with annotations."""
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
# Helper: Plot Metrics History
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