"""
data/data_helpers.py

Generic data utilities reusable across data environments.

- DataFrame subsetting (size-limited sampling from dataset).
- Augmentation routing: routes an augmentation config to the right function
  and applies it to a DataFrame with 'input' / 'output' columns.
- Grid-specific augmentations: color permutation, flip, rotation.
- Grid format conversion and tensor utilities (any format -> numpy / LongTensor).
- GridTokenizer: encodes/decodes flattened integer grids with configurable
  special tokens (BOS, EOS, PAD, UNK).

"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional

from utility.logging_utils import logger


# ---------------------------------------------------------------------------
# DataFrame utilities
# ---------------------------------------------------------------------------

def subset_dataframe(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """
    Return a random subset of `df` of size `n` (reproducible via `seed`).

    If `n` is None or >= len(df), the full DataFrame is returned unchanged.

    """
    if n is None or n >= len(df):
        if n is not None:
            logger.info(f"num_samples={n} >= dataset size ({len(df)}); using full set.")
        return df

    out = df.sample(n=n, random_state=seed).reset_index(drop=True)
    logger.info(f"Dataset limited to {n} samples (random subset, seed={seed}).")

    return out

# ---------------------------------------------------------------------------
# Grid format conversion
# ---------------------------------------------------------------------------

def grid_to_numpy(grid) -> np.ndarray:
    """Convert any supported grid format to a 2D numpy int array."""
    if isinstance(grid, np.ndarray):
        if grid.dtype == object:
            grid = np.stack([np.array(row) for row in grid])
        return grid.astype(int)

    if isinstance(grid, list):
        return np.array(grid, dtype=int)

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def to_grid_tensor(grid) -> torch.Tensor:
    """
    Convert any supported grid format to a 2D LongTensor.

    Handles: 2D numpy arrays, numpy object arrays of rows, list of lists,
    list of numpy arrays.
    """
    if isinstance(grid, np.ndarray) and grid.dtype != object:
        return torch.from_numpy(grid).long()

    if isinstance(grid, np.ndarray) and grid.dtype == object:
        grid = np.stack([np.array(row) for row in grid])
        return torch.from_numpy(grid).long()

    if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], np.ndarray):
        return torch.from_numpy(np.stack(grid)).long()

    if isinstance(grid, list):
        return torch.from_numpy(np.array(grid)).long()

    raise TypeError(f"Unsupported grid type: {type(grid)}")


def pad_2d_grid(grid_tensor: torch.Tensor, max_h: int, max_w: int, pad_value: int) -> torch.Tensor:
    """
    Pad a 2D grid tensor (H, W) to (max_h, max_w) with pad_value.

    Silently truncates if the grid exceeds the target dimensions.
    """
    h, w = grid_tensor.shape
    pad_bottom = max_h - h
    pad_right = max_w - w

    if pad_bottom < 0 or pad_right < 0:
        grid_tensor = grid_tensor[:max_h, :max_w]
        pad_bottom = max(0, max_h - grid_tensor.shape[0])
        pad_right = max(0, max_w - grid_tensor.shape[1])

    return F.pad(grid_tensor, (0, pad_right, 0, pad_bottom), mode='constant', value=pad_value)


# ---------------------------------------------------------------------------
# Data Augmentation
# ---------------------------------------------------------------------------

AUGMENTATION_REGISTRY = {
    "color_swap": lambda df, n, s: augment_color_swap(df, n, s),
    "grid_flip": lambda df, n, s: augment_grid_flip(df, n, s),
    "grid_rotation": lambda df, n, s: augment_grid_rotation(df, n, s),
}

def apply_augmentation(df: pd.DataFrame, aug_cfg, seed: int) -> pd.DataFrame:
    """
    Apply a configured augmentation to a DataFrame and return the augmented copy.

    `aug_cfg` must have:
      - use_data_augmentation (bool)
      - type (str): one of the keys in AUGMENTATION_REGISTRY
      - num_copies (int): number of augmented copies per original row

    Returns the original DataFrame unchanged if augmentation is disabled or
    the config is None.

    """
    if aug_cfg is None or not aug_cfg.use_data_augmentation:
        return df

    aug_type = aug_cfg.get("type", None)
    if aug_type not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unsupported augmentation type: '{aug_type}'. "
            f"Supported: {list(AUGMENTATION_REGISTRY.keys())}"
        )

    num_copies = int(aug_cfg.get("num_copies", 1))
    original_len = len(df)

    logger.info(
        f"Applying '{aug_type}' augmentation "
        f"(num_copies={num_copies}, seed={seed})..."
    )

    df = AUGMENTATION_REGISTRY[aug_type](df, num_copies, seed)

    logger.info(f"Dataset size after augmentation: {original_len} -> {len(df)} samples.")
    
    return df


def _apply_color_permutation(grid_np: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    """
    Apply a color permutation to a 2D grid copy.

    Value 0 is never remapped (background); only values 1-9 are remapped
    according to color_map.
    """
    out = grid_np.copy()
    for old_color, new_color in color_map.items():
        out[grid_np == old_color] = new_color
    return out


def augment_color_swap(df: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Augment by randomly permuting cell values 1-9.

    For each original row, generates `num_copies` new rows where every 1-9
    value in both 'input' and 'output' grids is consistently remapped via the
    same random permutation. Value 0 (background) is never changed.
    Identity-equivalent permutations are skipped.
    """
    rng = np.random.RandomState(seed)
    new_rows = []

    for _, row in df.iterrows():
        src_np = grid_to_numpy(row['input'])
        tgt_np = grid_to_numpy(row['output'])

        present_colors = (
            set(np.unique(src_np).tolist()) | set(np.unique(tgt_np).tolist())
        )
        present_colors.discard(0)

        generated, attempts = 0, 0
        max_attempts = num_copies * 20

        while generated < num_copies and attempts < max_attempts:
            attempts += 1
            perm = rng.permutation(9) + 1
            color_map = {int(i + 1): int(perm[i]) for i in range(9)}

            if all(color_map[c] == c for c in present_colors):
                continue

            new_row = row.copy()
            new_row['input'] = _apply_color_permutation(src_np, color_map)
            new_row['output'] = _apply_color_permutation(tgt_np, color_map)
            new_rows.append(new_row)
            generated += 1

    if not new_rows:
        return df

    return pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)


def _apply_flip(grid_np: np.ndarray, axis: int) -> np.ndarray:
    """Return a flipped copy of a 2D grid along the given axis (0=vertical, 1=horizontal)."""
    return np.flip(grid_np, axis=axis).copy()


def augment_grid_flip(df: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Augment by randomly flipping grids.

    For each original row, generates `num_copies` new rows where both 'input'
    and 'output' grids are flipped along the same randomly chosen axis:
    vertical (0), horizontal (1), or both axes (180° rotation).
    """
    rng = np.random.RandomState(seed)
    flip_modes = [(0,), (1,), (0, 1)]
    new_rows = []

    for _, row in df.iterrows():
        src_np = grid_to_numpy(row['input'])
        tgt_np = grid_to_numpy(row['output'])

        for _ in range(num_copies):
            mode = flip_modes[rng.randint(len(flip_modes))]
            src_aug, tgt_aug = src_np.copy(), tgt_np.copy()

            for axis in mode:
                src_aug = _apply_flip(src_aug, axis)
                tgt_aug = _apply_flip(tgt_aug, axis)

            new_row = row.copy()
            new_row['input'] = src_aug
            new_row['output'] = tgt_aug
            new_rows.append(new_row)

    if not new_rows:
        return df

    return pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)


def _apply_rotation(grid_np: np.ndarray, rot_amount: int) -> np.ndarray:
    """Return a rotated copy of a 2D grid (rot_amount: 1-3 counterclockwise 90° steps)."""
    return np.rot90(grid_np, k=rot_amount).copy()


def augment_grid_rotation(df: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Augment by randomly rotating grids by 90°, 180°, or 270°.

    For each original row, generates `num_copies` new rows where both 'input'
    and 'output' grids are rotated by the same randomly chosen angle.

    Note: 90° / 270° rotations swap H and W; so run before computing global max
    dimensions and before any padding.

    """
    rng = np.random.RandomState(seed)
    new_rows = []

    for _, row in df.iterrows():
        src_np = grid_to_numpy(row['input'])
        tgt_np = grid_to_numpy(row['output'])

        for _ in range(num_copies):
            k = rng.randint(1, 4)  # 90°, 180°, or 270° counterclockwise

            new_row = row.copy()
            new_row['input'] = _apply_rotation(src_np, rot_amount=k)
            new_row['output'] = _apply_rotation(tgt_np, rot_amount=k)
            new_rows.append(new_row)

    if not new_rows:
        return df

    return pd.concat([df, pd.DataFrame(new_rows, columns=df.columns)], ignore_index=True)


# ---------------------------------------------------------------------------
# Grid tokenizer
# ---------------------------------------------------------------------------

class GridTokenizer:
    """
    Tokenizer for 2D grids with integer cell values 0-9.

    Handles grid tokens (0-9) and predefined special tokens (BOS, EOS, PAD, UNK). 
    Environment-specific tokens (e.g., task tokens) are intentionally
    excluded and should be managed by the DataModule.

    """

    def __init__(self, cfg):
        self.grid_tokens = {str(i): i for i in range(10)}

        self.pad_token_id = cfg.data.pad_token_id
        self.bos_token_id = cfg.data.bos_token_id
        self.eos_token_id = cfg.data.eos_token_id
        self.unk_token_id = cfg.data.unk_token_id

        # Base vocabulary: 0-9 + PAD + BOS + EOS + UNK
        self.base_vocab_size = cfg.data.base_vocab_size

        self.idx2token = {v: k for k, v in self.grid_tokens.items()}
        self.idx2token[self.pad_token_id] = "<pad>"
        self.idx2token[self.bos_token_id] = "<bos>"
        self.idx2token[self.eos_token_id] = "<eos>"
        self.idx2token[self.unk_token_id] = "<unk>"

    def encode(self, grid_flat: List[int]) -> List[int]:
        """ Encode a flattened grid as [BOS] + token_ids + [EOS]. """
        ids = [self.bos_token_id]
        for val in grid_flat:
            if hasattr(val, "item"):
                val = val.item()

            if 0 <= val <= 9:
                ids.append(int(val))
            elif val == self.pad_token_id:
                ids.append(self.pad_token_id)
            else:
                ids.append(self.unk_token_id)

        ids.append(self.eos_token_id)
        return ids

    def decode(self, indices: List[int]) -> List[str]:
        """ Decode a list of token IDs back to string tokens. """
        return [self.idx2token.get(idx, "<unk>") for idx in indices]
