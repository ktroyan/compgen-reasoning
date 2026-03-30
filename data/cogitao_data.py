"""
data/data.py

TODO:
1) Implement a PTL DataModule (based on the config) to be used by a PTL Trainer.
2) The DataModule should support loading from local files or HuggingFace datasets based on the config.
3) It should handle preprocessing of grids (e.g., grid size padding, other special tokens, encoding as sequences, task embedding, etc.).
4) The DataModule should provide train, ID/OOD validation, and ID/OOD test dataloaders.

"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Tuple, Set
from omegaconf import OmegaConf

from utility.logging_utils import logger

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

def _grid_to_numpy(grid) -> np.ndarray:
    """ Convert any supported grid format to a 2D numpy int array. """
    
    if isinstance(grid, np.ndarray):
        if grid.dtype == object:
            grid = np.stack([np.array(row) for row in grid])
        return grid.astype(int)
    
    if isinstance(grid, list):
        return np.array(grid, dtype=int)
    
    raise TypeError(f"Unsupported grid type for augmentation: {type(grid)}")


def _apply_color_permutation(grid_np: np.ndarray, color_map: Dict[int, int]) -> np.ndarray:
    """ 
    Apply a color permutation to a 2D grid copy.
    
    NOTE: Value 0 is never remapped as it is background value; only values 1-9 are remapped according to the provided color_map.
    """

    color_swapped_grid = grid_np.copy()
    for old_color, new_color in color_map.items():
        color_swapped_grid[grid_np == old_color] = new_color
    
    return color_swapped_grid


def _augment_color_swap(df_train: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Create augmented training samples by randomly permuting values 1-9 (colors in the grid).

    For each original sample, generates `num_copies` new samples where every
    1-9 grid cell value is consistently remapped (same permutation applied
    to both input and output grids). Value 0 (background) is never changed.

    Args:
        df_train: Original training DataFrame.
        num_copies: Number of augmented copies to generate per sample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame train set with original rows followed by augmented rows.
    """
    rng = np.random.RandomState(seed)
    new_rows = []

    for _, row in df_train.iterrows():
        src_np = _grid_to_numpy(row['input'])
        tgt_np = _grid_to_numpy(row['output'])

        # Colors actually present (excluding background 0)
        present_colors = set(np.unique(src_np).tolist()) | set(np.unique(tgt_np).tolist())
        present_colors.discard(0)

        generated = 0
        attempts = 0
        max_attempts = num_copies * 20  # guard against infinite loops on degenerate grids

        while generated < num_copies and attempts < max_attempts:
            attempts += 1
            perm = rng.permutation(9) + 1  # random permutation of [1..9]
            color_map = {int(i + 1): int(perm[i]) for i in range(9)}

            if all(color_map[c] == c for c in present_colors):
                continue  # skip identity-equivalent permutations

            new_row = row.copy()
            new_row['input'] = _apply_color_permutation(src_np, color_map)
            new_row['output'] = _apply_color_permutation(tgt_np, color_map)
            new_rows.append(new_row)
            generated += 1

    if not new_rows:
        return df_train

    augmented_df = pd.DataFrame(new_rows, columns=df_train.columns)
    
    return pd.concat([df_train, augmented_df], ignore_index=True)

def _apply_flip(grid_np: np.ndarray, axis: int) -> np.ndarray:
    """ Return a flipped copy of a 2D grid along the given numpy axis (0=vertical, 1=horizontal). """
    return np.flip(grid_np, axis=axis).copy()

def _augment_grid_flip(df_train: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Create augmented training samples by randomly flipping grids.

    For each original sample, generates `num_copies` new samples where both
    the input and output grids are flipped along the same randomly chosen axis:
      - axis 0: vertical flip (top <-> bottom)
      - axis 1: horizontal flip (left <-> right)
      - both axes: equivalent to 180° rotation

    The same flip is applied to both input and output, preserving the
    input -> output relationship. Grid dimensions are unchanged.
    All other metadata (e.g. transformation_suite) is copied from the original row.

    NOTE: This augmentation should be ok for any COGITAO transformation.

    Args:
        df_train: Original training DataFrame.
        num_copies: Number of augmented copies to generate per sample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with original rows followed by augmented rows.

    """
    rng = np.random.RandomState(seed)
    
    # Possible flip modes: vertical, horizontal, both
    flip_modes = [
        (0,),     # vertical
        (1,),     # horizontal
        (0, 1),   # both axes
    ]

    new_rows = []

    for _, row in df_train.iterrows():
        src_np = _grid_to_numpy(row['input'])
        tgt_np = _grid_to_numpy(row['output'])

        for _ in range(num_copies):
            mode = flip_modes[rng.randint(len(flip_modes))]
            src_aug = src_np.copy()
            tgt_aug = tgt_np.copy()
            
            for axis in mode:
                src_aug = _apply_flip(src_aug, axis)
                tgt_aug = _apply_flip(tgt_aug, axis)

            new_row = row.copy()
            new_row['input'] = src_aug
            new_row['output'] = tgt_aug
            new_rows.append(new_row)

    if not new_rows:
        return df_train

    augmented_df = pd.DataFrame(new_rows, columns=df_train.columns)
    
    return pd.concat([df_train, augmented_df], ignore_index=True)

def _apply_rotation(grid_np: np.ndarray, rot_amount: int) -> np.ndarray:
    """ Return a rotated copy of a 2D grid by the given amount (1-3 in 90° counterclockwise rotations). """
    return np.rot90(grid_np, k=rot_amount).copy()

def _augment_grid_rotation(df_train: pd.DataFrame, num_copies: int, seed: int) -> pd.DataFrame:
    """
    Create augmented training samples by randomly rotating grids by 90°, 180°, or 270°.

    For each original sample, generates `num_copies` new samples where both
    the input and output grids are rotated by the same randomly chosen angle.
    
    NOTE: 90° and 270° rotations swap H and W dimensions, but this is ok even for non-square grids
          since the augmentation runs before the splits are analyzed for max dimensions and before any padding is applied.

    Args:
        df_train: Original training DataFrame.
        num_copies: Number of augmented copies to generate per sample.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with original rows followed by augmented rows.
    
    """
    rng = np.random.RandomState(seed)
    new_rows = []

    for _, row in df_train.iterrows():
        src_np = _grid_to_numpy(row['input'])
        tgt_np = _grid_to_numpy(row['output'])

        for _ in range(num_copies):
            k_times_90 = rng.randint(1, 4)  # 1, 2, or 3 -> 90°, 180°, 270° counterclockwise

            new_row = row.copy()
            new_row['input'] = _apply_rotation(src_np, rot_amount=k_times_90)
            new_row['output'] = _apply_rotation(tgt_np, rot_amount=k_times_90)
            new_rows.append(new_row)

    if not new_rows:
        return df_train

    augmented_df = pd.DataFrame(new_rows, columns=df_train.columns)
    
    return pd.concat([df_train, augmented_df], ignore_index=True)


class GridTokenizer:
    """
    Handles standard Grid tokens (0-9) and structural special tokens.
    Task tokens are handled dynamically via offset mapping in the DataModule.
    """
    def __init__(self, cfg):
        self.grid_tokens = {str(i): i for i in range(10)}
        
        # Grid padding
        self.pad_token_id = cfg.data.pad_token_id
        
        # Sequence tokens
        self.bos_token_id = cfg.data.bos_token_id
        self.eos_token_id = cfg.data.eos_token_id
        
        self.unk_token_id = cfg.data.unk_token_id

        # Base vocabulary size (0–9 + PAD + BOS + EOS + UNK)
        self.base_vocab_size = cfg.data.base_vocab_size

        self.idx2token = {v: k for k, v in self.grid_tokens.items()}
        self.idx2token[self.pad_token_id] = "<pad>"
        self.idx2token[self.bos_token_id] = "<bos>"
        self.idx2token[self.eos_token_id] = "<eos>"
        self.idx2token[self.unk_token_id] = "<unk>"

    def encode(self, grid_flat: List[int]) -> List[int]:
        ids = [self.bos_token_id]
        for val in grid_flat:
            if hasattr(val, "item"):
                val = val.item()
            
            if 0 <= val <= 9:
                ids.append(int(val))
            else:
                # If we encounter padding from the 2D step, map to PAD token
                if val == self.pad_token_id:
                    ids.append(self.pad_token_id)
                
                else:
                    logger.warning(f"Unexpected grid value {val} encountered during encoding. Mapping to UNK token.")
                    ids.append(self.unk_token_id)
        
        ids.append(self.eos_token_id)
        
        return ids

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2token.get(idx, "<unk>") for idx in indices]

class GridDataset(Dataset):
    def __init__(self,
                 data,
                 tokenizer: GridTokenizer, 
                 max_h: int, max_w: int, 
                 task_map: Dict[str, int], 
                 max_task_seq_len: int,
                 task_pad_token_id: int,
                 use_task_tokens: bool
                 ):
        """
        Args:
            data: The dataset split (Pandas DF).
            tokenizer: GridTokenizer.
            max_h, max_w: Global maximum dimensions to pad 2D grids to.
            task_map: Dictionary mapping transformation strings to Token IDs.
            max_task_seq_len: Global maximum task sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_h = max_h
        self.max_w = max_w
        self.task_map = task_map
        self.max_task_seq_len = max_task_seq_len
        self.task_pad_token_id = task_pad_token_id
        self.use_task_tokens = use_task_tokens


    def __len__(self):
        return len(self.data)

    def _to_tensor(self, grid):
        """
        Robust conversion of HF-loaded grids to 2D LongTensor.
        Handles:
            - numpy 2D numeric arrays
            - numpy object arrays containing row arrays
            - list of lists
            - list of numpy arrays
        """

        # Case 1: Proper numeric 2D numpy array
        if isinstance(grid, np.ndarray) and grid.dtype != object:
            return torch.from_numpy(grid).long()

        # Case 2: numpy object array of rows
        if isinstance(grid, np.ndarray) and grid.dtype == object:
            # Each element is a row array
            grid = np.stack([np.array(row) for row in grid])
            return torch.from_numpy(grid).long()

        # Case 3: list of numpy arrays
        if isinstance(grid, list) and len(grid) > 0 and isinstance(grid[0], np.ndarray):
            grid = np.stack(grid)
            return torch.from_numpy(grid).long()

        # Case 4: list of lists
        if isinstance(grid, list):
            grid = np.array(grid)
            return torch.from_numpy(grid).long()

        raise TypeError(f"Unsupported grid type: {type(grid)}")

    def _pad_2d_grid(self, grid_tensor: torch.Tensor) -> torch.Tensor:
        """
        Pads a 2D grid (H, W) to (self.max_h, self.max_w) using PAD_TOKEN.
        """
        h, w = grid_tensor.shape
        
        # Calculate padding amounts
        pad_bottom = self.max_h - h
        pad_right = self.max_w - w
        
        # Safety truncation if grid exceeds max dims (shouldn't happen if scanned correctly)
        if pad_bottom < 0 or pad_right < 0:
            grid_tensor = grid_tensor[:self.max_h, :self.max_w]
            pad_bottom = max(0, self.max_h - grid_tensor.shape[0])
            pad_right = max(0, self.max_w - grid_tensor.shape[1])

        # F.pad format: (pad_left, pad_right, pad_top, pad_bottom)
        grid_padded = F.pad(
            grid_tensor, 
            (0, pad_right, 0, pad_bottom), 
            mode='constant', 
            value=self.tokenizer.pad_token_id
        )
        return grid_padded

    def __getitem__(self, idx):
        """
        Fetches grid samples, preprocesses them (2D padding, flattening, tokenization), processes task tokens, and builds the input and target sequences for the model.
        """

        if hasattr(self.data, "iloc"):
            item = self.data.iloc[idx]
        else:
            item = self.data[idx]

        # -------------------------
        # Process Grids (2D padding + flattening + tokenization)
        # -------------------------
        src_raw = self._to_tensor(item['input'])
        tgt_raw = self._to_tensor(item['output'])

        # 2D Padding
        src_padded_2d = self._pad_2d_grid(src_raw)
        tgt_padded_2d = self._pad_2d_grid(tgt_raw)

        # Flatten
        src_flat = src_padded_2d.flatten().tolist()
        tgt_flat = tgt_padded_2d.flatten().tolist()

        # Tokenize WITHOUT adding BOS/EOS
        src_grid_tokens = []
        for val in src_flat:
            if 0 <= val <= 9:
                src_grid_tokens.append(int(val))
            else:
                if val == self.tokenizer.pad_token_id:
                    src_grid_tokens.append(self.tokenizer.pad_token_id)
                else:
                    src_grid_tokens.append(self.tokenizer.unk_token_id)

        tgt_grid_tokens = []
        for val in tgt_flat:
            if 0 <= val <= 9:
                tgt_grid_tokens.append(int(val))
            else:
                if val == self.tokenizer.pad_token_id:
                    tgt_grid_tokens.append(self.tokenizer.pad_token_id)
                else:
                    tgt_grid_tokens.append(self.tokenizer.unk_token_id)

        # -------------------------
        # Process Task Tokens
        # -------------------------
        transforms = item.get('transformation_suite', [])
        if isinstance(transforms, np.ndarray):
            transforms = transforms.tolist()

        task_tokens = [
            self.task_map.get(t, self.tokenizer.unk_token_id)
            for t in transforms
        ]

        # Pad task sequence globally
        if len(task_tokens) < self.max_task_seq_len:
            pad_len = self.max_task_seq_len - len(task_tokens)
            task_tokens += [self.task_pad_token_id] * pad_len
        else:
            task_tokens = task_tokens[:self.max_task_seq_len]

        # -------------------------
        # Build final encoder sequence
        # -------------------------
        # Structure:
        # - Input: [BOS] (+ task_tokens) + grid_tokens + [EOS]
        # - Target: grid_tokens + [EOS]

        if self.use_task_tokens:
            # If using task tokens, we can keep task tokens in the sequence, let the model learn embeddings for them and how to use them (e.g., for OOD generalization). 
            # The task tokens are prepended to the grid tokens in the input sequence
            src_sequence = (
                [self.tokenizer.bos_token_id]
                + task_tokens
                + src_grid_tokens
                + [self.tokenizer.eos_token_id]
            )
        else:
            # If not using task tokens, we can choose to exclude task tokens from the sequence here but eventually encode them in a special way later in the pipeline 
            src_sequence = (
                [self.tokenizer.bos_token_id]
                + src_grid_tokens
                + [self.tokenizer.eos_token_id]
            )

        # For the target sequence, we do not use a <BOS> token
        tgt_sequence = (
            tgt_grid_tokens
            + [self.tokenizer.eos_token_id]
        )

        src_tokens = torch.tensor(src_sequence, dtype=torch.long)
        tgt_tokens = torch.tensor(tgt_sequence, dtype=torch.long)
        task_tokens = torch.tensor(task_tokens, dtype=torch.long)

        return src_tokens, tgt_tokens, task_tokens

class GridDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()

        logger.info("Instantiating Data Module (GridDataModule)...")

        self.cfg = cfg
        self.data_source = cfg.data.get("data_source", None).lower()
        self.data_path = cfg.data.get("data_path", "")

        if self.data_source not in ["huggingface", "local"]:
            raise ValueError(f"Unsupported data source: {self.data_source}")

        if self.data_path.startswith("./https"):
            self.data_path = self.data_path[2:]

        self.train_batch_size = cfg.training.get("batch_size", 32)
        self.inference_batch_size = cfg.get("inference", {}).get("batch_size", 10)
        self.num_workers = cfg.data.get("num_workers", 4)

        self.use_ood_val = cfg.data.get("use_ood_val", False)
        self.use_ood_test = cfg.data.get("use_ood_test", False)

        # Placeholders for computed stats
        self.max_h = 0
        self.max_w = 0
        self.max_task_seq_len = 0
        self.task_map = {}

        self.vocab_size = cfg.data.get("vocab_size", None)

        self.tokenizer = None
        self.train_ds = None
        self.val_id_ds = None
        self.val_ood_ds = None
        self.test_id_ds = None
        self.test_ood_ds = None

    def prepare_data(self):
        if self.data_source == "huggingface" and not HF_AVAILABLE:
            raise ImportError("datasets library not installed.")

    def _log_sample_structure(self, df, split_name="train"):
        """
        Logs detailed structure of the first sample of a split.
        """

        message = ""
        
        if df is None or len(df) == 0:
            logger.warning(f"{split_name} split is empty.")
            return

        sample = df.iloc[0]

        message += f"Full sample:\n{sample}"
        message += f"\n--- Sample Structure ({split_name}) ---"

        for key in sample.index:
            value = sample[key]

            message += f"\nField: {key}"
            message += f"Type: {type(value)}"

            if isinstance(value, np.ndarray):
                message += f"\nndim: {value.ndim}"
                message += f"\nshape: {value.shape}"
                message += f"\ndtype: {value.dtype}"

                if value.ndim == 1 and len(value) > 0:
                    message += f"\nFirst element type: {type(value[0])}"

            elif isinstance(value, list):
                message += f"\nLength: {len(value)}"

                if len(value) > 0:
                    message += f"\nFirst element type: {type(value[0])}"

                    if isinstance(value[0], list):
                        message += f"\nNested length: {len(value[0])}"

            else:
                message += f"\nValue: {value}"

        logger.info(message)

    def _analyze_split(self, df) -> Tuple[int, int, Set[str], Set[Tuple[str]], int]:
        """
        Helper to scan a single dataframe split.
        Returns: (max_h, max_w, atomic_tasks_set, sequence_set, max_seq_len)
        """
        split_max_h, split_max_w = 0, 0
        split_atomic_tasks = set()
        split_sequences = set()
        split_max_task_seq_len = 0

        def get_dims(grid):
            # Handle Numpy Arrays
            if isinstance(grid, np.ndarray):
                if grid.ndim == 2:
                    return grid.shape
                # Numpy array of lists (Ragged) -> shape (H,)
                if grid.ndim == 1:
                    if grid.size > 0 and isinstance(grid[0], (list, np.ndarray)):
                        return (grid.shape[0], len(grid[0])) # (H, W)
                    return (1, grid.shape[0]) # 1D
            
            # Handle Lists
            if isinstance(grid, list):
                if not grid:
                    return (0, 0)
                
                if isinstance(grid[0], list):
                    return (len(grid), len(grid[0]))
                
                return (1, len(grid))
                
            return (0, 0)

        for col in ['input', 'output']:
            if col in df.columns:
                dims = df[col].apply(get_dims).tolist()
                for d in dims:
                    # Robust unpacking check
                    if isinstance(d, (tuple, list)) and len(d) == 2:
                        h, w = d
                        split_max_h = max(split_max_h, h)
                        split_max_w = max(split_max_w, w)
                    elif isinstance(d, (tuple, list)) and len(d) == 1:
                        # Fallback for weird 1D shape return
                        split_max_w = max(split_max_w, d[0])

        if 'transformation_suite' in df.columns:
            def process_task(suite):
                if isinstance(suite, np.ndarray):
                    suite = suite.tolist()
                if not isinstance(suite, list):
                    return 0
                for t in suite:
                    split_atomic_tasks.add(t)
                if len(suite) > 0:
                    split_sequences.add(tuple(suite))
                return len(suite)

            lengths = df['transformation_suite'].apply(process_task)
            if not lengths.empty:
                split_max_task_seq_len = lengths.max()

        return split_max_h, split_max_w, split_atomic_tasks, split_sequences, split_max_task_seq_len

    def setup(self, stage=None):
        logger.info(f"Setting up DataModule (Stage: {stage})")
        self.tokenizer = GridTokenizer(cfg=self.cfg)

        if self.data_source == "huggingface":
            hf_url = self.data_path
            
            # Load Dataframes
            def get_df(filename):
                base = hf_url.rstrip("/")
                file_url = f"{base}/{filename}"
                try:
                    ds = load_dataset("parquet", data_files={"data": file_url}, split="data")
                    return ds.to_pandas()
                except Exception:
                    logger.warning(f"File not found or load failed: {filename}")
                    return None

            df_train = get_df("train.parquet")
            df_val = get_df("val.parquet")
            df_test = get_df("test.parquet")
            
            if df_train is None:
                raise ValueError("Train data is missing!")
            if df_val is None and self.use_id_val:
                logger.warning("ID Validation data is missing but use_id_val is True.")
            if df_test is None and self.use_id_test:
                logger.warning("ID Test data is missing but use_id_test is True.")

            df_val_ood = get_df("val_ood.parquet") if self.use_ood_val else None
            df_test_ood = get_df("test_ood.parquet") if self.use_ood_test else None

            if self.use_ood_val and df_val_ood is None:
                logger.warning("OOD Validation data is missing but use_ood_val is True.")
            if self.use_ood_test and df_test_ood is None:
                logger.warning("OOD Test data is missing but use_ood_test is True.")

            # ------------------------------
            # Data augmentation (training set only, applied before any stats or tokenization)
            # ------------------------------
            aug_cfg = self.cfg.data.get("data_augmentation", None)

            if aug_cfg is not None and aug_cfg.use_data_augmentation:
                aug_type = aug_cfg.get("type", None)
                
                aug_types = {"color_swap", "grid_flip", "grid_rotation"}
                if aug_type not in aug_types:
                    raise ValueError(f"Unsupported data augmentation type: '{aug_type}'. Supported: {aug_types}")
                
                num_copies = int(aug_cfg.get("num_copies", 1))
                original_df_train_len = len(df_train)
                
                logger.info(f"Applying '{aug_type}' augmentation to training set "
                            f"(num_copies={num_copies}, seed={self.cfg.seed}, avoid_identity=True)...")
                
                if aug_type == "color_swap":
                    df_train = _augment_color_swap(df_train, num_copies=num_copies, seed=self.cfg.seed)
                
                elif aug_type == "grid_flip":
                    df_train = _augment_grid_flip(df_train, num_copies=num_copies, seed=self.cfg.seed)
                
                elif aug_type == "grid_rotation":
                    df_train = _augment_grid_rotation(df_train, num_copies=num_copies, seed=self.cfg.seed)
                
                logger.info(f"Training set size after augmentation: {original_df_train_len} -> {len(df_train)} samples.")

            # Log sample structure for debugging
            self._log_sample_structure(df_train, "train")

            # Scan for Stats (ID vs OOD)
            id_dfs = [df for df in [df_train, df_val, df_test] if df is not None]
            ood_dfs = [df for df in [df_val_ood, df_test_ood] if df is not None]

            # Aggregate ID Stats
            id_atoms, id_seqs = set(), set()
            
            for df in id_dfs:
                mh, mw, atoms, seqs, m_task_len = self._analyze_split(df)
                self.max_h = max(self.max_h, mh)
                self.max_w = max(self.max_w, mw)
                self.max_task_seq_len = max(self.max_task_seq_len, m_task_len)
                id_atoms.update(atoms)
                id_seqs.update(seqs)

            # Aggregate OOD Stats
            ood_atoms, ood_seqs = set(), set()
            if self.use_ood_val or self.use_ood_test:

                for df in ood_dfs:
                    mh, mw, atoms, seqs, m_task_len = self._analyze_split(df)
                    self.max_h = max(self.max_h, mh)
                    self.max_w = max(self.max_w, mw)
                    
                    # OOD usually determines the max depth
                    self.max_task_seq_len = max(self.max_task_seq_len, m_task_len)
                    ood_atoms.update(atoms)
                    ood_seqs.update(seqs)

            # Logging & Analysis
            logger.info("--- Data Statistics Analysis ---")
            logger.info(f"Global Max Grid Size: {self.max_h}x{self.max_w}")
            logger.info(f"Global Max Task Depth: {self.max_task_seq_len}")
            
            logger.info(f"ID Atomic Transformations ({len(id_atoms)}): {sorted(list(id_atoms))}")
            if self.use_ood_val or self.use_ood_test:
                logger.info(f"OOD Atomic Transformations ({len(ood_atoms)}): {sorted(list(ood_atoms))}")
            
            if self.use_ood_val or self.use_ood_test:
                new_atoms = ood_atoms - id_atoms
                if new_atoms:
                    logger.info(f"New Atomic Transformations in OOD: {new_atoms}")
                else:
                    logger.warning("No new atomic transformations in OOD set.")

            logger.info(f"Unique ID Task Sequences: {len(id_seqs)}")

            if self.use_ood_val or self.use_ood_test:
                logger.info(f"Unique OOD Task Sequences: {len(ood_seqs)}")
            
                new_seqs = ood_seqs - id_seqs
                logger.info(f"New Task Sequences in OOD: {len(new_seqs)}")
                
                # Sample a few new sequences to log
                if len(new_seqs) > 0:
                    logger.info(f"Example OOD Sequences: {list(new_seqs)[:5]}")

            # Build global task map
            all_atoms = sorted(list(id_atoms | ood_atoms))
            vocab_start_id = self.tokenizer.base_vocab_size

            self.task_map = {
                task: vocab_start_id + i
                for i, task in enumerate(all_atoms)
            }

            # Task padding token ID (last ID in input vocab)
            self.task_pad_token_id = vocab_start_id + len(all_atoms)

            # Input vocab size = base + atomic tasks + task_pad
            self.vocab_size = self.tokenizer.base_vocab_size + len(self.task_map) + 1  # +1 for task_pad_token_id

            # Update config
            logger.info(f"Updating global config data.vocab_size to {self.vocab_size}")
            
            self.total_special_tokens = 2 + self.max_task_seq_len   # BOS + EOS [+ task_tokens]
            max_seq_len = (self.max_h * self.max_w) + self.total_special_tokens # for encoder input (grid flattened + special tokens)
            
            OmegaConf.set_struct(self.cfg, False) 
            
            # Update Data config
            self.cfg.data.vocab_size = int(self.vocab_size)
            self.cfg.data.max_h = int(self.max_h)
            self.cfg.data.max_w = int(self.max_w)
            self.cfg.data.max_task_seq_len = int(self.max_task_seq_len)
            self.cfg.data.max_seq_len = int(max_seq_len) # this is the max length of the input sequence given to the model 

            if self.cfg.model.get("use_task_tokens", False):
                self.total_seq_special_tokens_prepended = 1 + self.max_task_seq_len  # BOS + task tokens
            else:
                self.total_seq_special_tokens_prepended = 1  # BOS only

            self.total_seq_special_tokens_appended = 1  # EOS
            self.cfg.model.total_seq_special_tokens_prepended = int(self.total_seq_special_tokens_prepended)
            self.cfg.model.total_seq_special_tokens_appended = int(self.total_seq_special_tokens_appended)

            # Update Model config
            self.cfg.model.input_vocab_size = int(self.vocab_size)
            self.cfg.model.max_h = int(self.max_h)
            self.cfg.model.max_w = int(self.max_w)
            self.cfg.model.max_task_seq_len = int(self.max_task_seq_len)
            self.cfg.model.max_seq_len = int(max_seq_len) # encoder network needs this (e.g., for PE)
            
            OmegaConf.set_struct(self.cfg, True)

            # Instantiate Datasets
            def make_ds(df):
                if df is None:
                    return None
                
                return GridDataset(
                    df,
                    self.tokenizer,
                    self.max_h,
                    self.max_w,
                    self.task_map,
                    self.max_task_seq_len,
                    self.task_pad_token_id,
                    use_task_tokens=self.cfg.model.get("use_task_tokens", False)
                )

            self.train_ds = make_ds(df_train)
            self.val_id_ds = make_ds(df_val)
            self.test_id_ds = make_ds(df_test)
            self.val_ood_ds = make_ds(df_val_ood)
            self.test_ood_ds = make_ds(df_test_ood)

        elif self.data_source == "local":
            raise NotImplementedError("Local data loading not implemented.")

    def collate_fn(self, batch):
        src_batch, tgt_batch, task_batch = zip(*batch)

        src_stacked = torch.stack(src_batch)
        tgt_stacked = torch.stack(tgt_batch)
        task_stacked = torch.stack(task_batch)

        return src_stacked, tgt_stacked, task_stacked

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batch_size, shuffle=self.cfg.data.get("shuffle_train", True),
                          num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        loaders = []
        if self.val_id_ds:
            loaders.append(DataLoader(self.val_id_ds, batch_size=self.inference_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn))
        if self.use_ood_val and self.val_ood_ds:
            loaders.append(DataLoader(self.val_ood_ds, batch_size=self.inference_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn))
        return loaders

    def test_dataloader(self):
        loaders = []
        if self.test_id_ds:
            loaders.append(DataLoader(self.test_id_ds, batch_size=self.inference_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn))
        if self.use_ood_test and self.test_ood_ds:
            loaders.append(DataLoader(self.test_ood_ds, batch_size=self.inference_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn))
        return loaders
