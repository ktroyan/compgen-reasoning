"""
utility/general_utils.py

General utility functions for reproducibility, saving, and memory handling.
"""

import os
import json
import random
import numpy as np
import torch


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------
def set_seed(seed: int):
    """
    Set all random seeds for full reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium') 


# -------------------------------------------------------
# Directory helpers
# -------------------------------------------------------
def ensure_dir_exists(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# -------------------------------------------------------
# JSON saving
# -------------------------------------------------------
def save_json(data, path: str, indent: int = 2):
    ensure_dir_exists(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


# -------------------------------------------------------
# Memory handling
# -------------------------------------------------------
def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
