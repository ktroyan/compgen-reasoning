"""
utility/visualization_utils.py

"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from torch.utils.data import DataLoader

def get_arc_colormap():
    """
    Returns an ARC-like colormap and norm for grid tasks + Special Tokens.
    """
    # 0-9: Standard ARC colors
    hex_colors = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
    ]
    
    # Special Token Colors (10, 11, 12, 13)
    # NOTE: refer to config values
    # 10 PAD: Very Light Gray
    # 11 BOS: Neon Green
    # 12 EOS: Neon Red
    # 13 UNK: Deep Purple
    hex_colors.extend(['#F5F5F5', '#00FF00', '#FF0000', '#800080'])
    
    cmap = colors.ListedColormap(hex_colors)
    bounds = list(range(15)) 
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    return cmap, norm

def to_numeric_grid(data, pad_val=10):
    """
    Robustly converts data (List, Numpy Object, Ragged Array) into a 
    rectangular numeric numpy array. Pads shorter rows if necessary.

    NOTE: refer to config values for pad_val (should match pad_token_id)
    """
    # If it's already a numeric array, return it
    if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
        return data
        
    # Convert to list structure if it's a numpy object/array
    if isinstance(data, np.ndarray):
        data = data.tolist()
        
    if not isinstance(data, list):
        return np.array(data) # handle scalar edge cases
        
    # Handle nesting and ragged arrays
    if len(data) == 0:
        return np.array(data)
        
    # Check if inner elements are iterables (2D grid)
    if isinstance(data[0], (list, np.ndarray)):
        # Find max width
        max_width = 0
        rows = []
        for row in data:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, list):
                row = [row]
            rows.append(row)
            max_width = max(max_width, len(row))
            
        # Pad rows to make rectangular
        padded_data = []
        for row in rows:
            pad_len = max_width - len(row)
            padded_data.append(row + [pad_val] * pad_len)
            
        return np.array(padded_data, dtype=int)
    
    # Flat list
    return np.array(data, dtype=int)

def plot_grid(ax, data, cmap, norm, title="Grid"):
    """
    Helper to plot a single grid/sequence on a matplotlib axis.
    """
    # Ensure data is numeric and rectangular
    data = to_numeric_grid(data)

    # If 1D sequence, reshape to (1, L) for imshow
    if len(data.shape) == 1:
        data = data[None, :]
        
    ax.imshow(data, cmap=cmap, norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.1)
    
    # Annotate values if grid is small
    if data.size < 400: 
        for (j, i), label in np.ndenumerate(data):
            # Only annotate standard colors
            if 0 <= label <= 13:
                text_color = 'white' if label in [0, 1, 9, 13] else 'black'
                ax.text(i, j, str(label), ha='center', va='center', 
                        color=text_color, fontsize=6)

def visualize_data_samples(dataloader: DataLoader, indices: list[int], save_path: str = None):
    dataset = dataloader.dataset
    cmap, norm = get_arc_colormap()
    
    num_samples = len(indices)
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        # Get Tokenized Tensors (what the model gets)
        src_tensor, tgt_tensor, task_tensor = dataset[idx]
        
        src_seq = src_tensor.numpy()
        tgt_seq = tgt_tensor.numpy()
        task_seq = task_tensor.numpy()
        
        # Try to get Raw 2D Grids (ground truth)
        raw_input_grid = None
        raw_output_grid = None
        
        try:
            if hasattr(dataset, 'data'):
                item = dataset.data.iloc[idx] if hasattr(dataset.data, 'iloc') else dataset.data[idx]
                raw_input_grid = item['input']
                raw_output_grid = item['output']
        except Exception:
            pass 

        # --- Plotting ---
        row_axes = axes[i]
        
        # Raw Input (2D)
        if raw_input_grid is not None:
            plot_grid(row_axes[0], raw_input_grid, cmap, norm, title=f"Sample {idx}\nRaw Input (2D)")
        else:
            row_axes[0].text(0.5, 0.5, "Raw Not Avail", ha='center')
            row_axes[0].axis('off')

        # Tokenized Input (1D)
        plot_grid(row_axes[1], src_seq, cmap, norm, title=f"Tokenized Input (1D)\n(Len: {len(src_seq)})")

        # Raw Target (2D)
        if raw_output_grid is not None:
            plot_grid(row_axes[2], raw_output_grid, cmap, norm, title="Raw Target (2D)")
        else:
            row_axes[2].text(0.5, 0.5, "Raw Not Avail", ha='center')
            row_axes[2].axis('off')

        # Tokenized Target (1D)
        plot_grid(row_axes[3], tgt_seq, cmap, norm, title=f"Tokenized Target (1D)\n(Len: {len(tgt_seq)})")

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    
    plt.close()