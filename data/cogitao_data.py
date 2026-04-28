"""
data/cogitao_data.py

Defines GridDataModule, a PTL DataModule for COGITAO grid data.

- Loads data from HuggingFace or local storage based on the config.
- Pads grids to a max size (with PAD token), optionally prepends task tokens and
  concatenates other special tokens (e.g., BOS/EOS), and encodes sequences.
- Applies optional data augmentation (e.g., color swap, grid flip, rotation) on
  the training set after the training (sub)set is created.
- Provides train, ID/OOD validation, and ID/OOD test dataloaders.
"""

import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Tuple, Set
from omegaconf import OmegaConf

## Personal imports
# Utilities
from utility.logging_utils import logger

# Generic data helpers (grid utilities, augmentations, tokenizer)
from data.data_helpers import (
    GridTokenizer,
    to_grid_tensor,
    pad_2d_grid,
    subset_dataframe,
    apply_augmentation,
)

# Try to import HuggingFace datasets library for loading data from HuggingFace Hub
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


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

    def __getitem__(self, idx):
        """
        Fetches grid samples, preprocesses them (2D padding, flattening, tokenization),
        processes task tokens, and builds the input and target sequences for the model.
        """

        if hasattr(self.data, "iloc"):
            item = self.data.iloc[idx]
        else:
            item = self.data[idx]

        # ------------------------------------------------------------------
        # Process Grids (2D padding + flattening + tokenization)
        # ------------------------------------------------------------------
        src_raw = to_grid_tensor(item['input'])
        tgt_raw = to_grid_tensor(item['output'])

        # 2D Padding
        src_padded_2d = pad_2d_grid(src_raw, self.max_h, self.max_w, self.tokenizer.pad_token_id)
        tgt_padded_2d = pad_2d_grid(tgt_raw, self.max_h, self.max_w, self.tokenizer.pad_token_id)

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

        # ------------------------------------------------------------------
        # Process Task Tokens
        # ------------------------------------------------------------------
        # Get the transformation suite
        # Raw list of strings representing the task, where each string is an atomic transformation (e.g., "translate_up")
        transformation_suite = item.get('transformation_suite', [])
        if isinstance(transformation_suite, np.ndarray):
            transformation_suite = transformation_suite.tolist()

        task_tokens = [
            self.task_map.get(t, self.tokenizer.unk_token_id)
            for t in transformation_suite
        ]

        # Pad task sequence globally
        if len(task_tokens) < self.max_task_seq_len:
            pad_len = self.max_task_seq_len - len(task_tokens)
            task_tokens += [self.task_pad_token_id] * pad_len
        else:
            task_tokens = task_tokens[:self.max_task_seq_len]

        # ------------------------------------------------------------------
        # Build the input sequence for the encoder network and target sequence
        # ------------------------------------------------------------------
        # Structure:
        # - Input: [BOS] (+ task_tokens) + grid_tokens + [EOS]
        # - Target: grid_tokens + [EOS]
        # NOTE: There is no use in having a BOS token in the target sequence for our current setup

        if self.use_task_tokens:
            # If using task tokens, we prepend task tokens to the input grid sequence. We let the model learn embeddings for them and how to use them (e.g., for OOD generalization).
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

        return {
            "src": src_tokens,
            "tgt": tgt_tokens,
            "task_tokens": task_tokens,
            "transformation_suite": transformation_suite
        }

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
        """ Logs detailed structure of the first sample of a split. """

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
        Scan a single COGITAO dataframe split for grid dimensions and task stats.
        
        Returns: (max_h, max_w, atomic_tasks_set, sequence_set, max_seq_len)
        
        """
        split_max_h, split_max_w = 0, 0
        split_atomic_tasks = set()
        split_sequences = set()
        split_max_task_seq_len = 0

        def get_dims(grid):
            if isinstance(grid, np.ndarray):
                if grid.ndim == 2:
                    return grid.shape
                if grid.ndim == 1:
                    if grid.size > 0 and isinstance(grid[0], (list, np.ndarray)):
                        return (grid.shape[0], len(grid[0]))
                    return (1, grid.shape[0])

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
                    if isinstance(d, (tuple, list)) and len(d) == 2:
                        h, w = d
                        split_max_h = max(split_max_h, h)
                        split_max_w = max(split_max_w, w)
                    elif isinstance(d, (tuple, list)) and len(d) == 1:
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

            # ------------------------------------------------------------------
            # Training set size limit (applied before augmentation so that aug acts on the subset)
            # ------------------------------------------------------------------
            seed = self.cfg.get("seed", 42)
            df_train = subset_dataframe(df_train, self.cfg.data.get("num_train_samples", None), seed)

            # ------------------------------------------------------------------
            # Data augmentation (training set only, applied after optional subsetting and before any stats or tokenization)
            # ------------------------------------------------------------------
            aug_cfg = self.cfg.data.get("data_augmentation", None)
            df_train = apply_augmentation(df_train, aug_cfg, seed=self.cfg.get("seed", 42))

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

            # Task padding token ID (last ID in input vocab) defined at run time
            self.task_pad_token_id = vocab_start_id + len(all_atoms)

            # Input vocab size = base + atomic tasks + task_pad
            self.vocab_size = self.tokenizer.base_vocab_size + len(self.task_map) + 1  # +1 for task_pad_token_id

            # ------------------------------------------------------------------
            # Update config at runtime with computed stats (e.g., actual vocab size, max dimensions, etc.) so that downstream components (model, dataloader) can access them
            # ------------------------------------------------------------------
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
            self.cfg.data.task_pad_token_id = int(self.task_pad_token_id)

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
        return {
            "src": torch.stack([item["src"] for item in batch]),
            "tgt": torch.stack([item["tgt"] for item in batch]),
            "task_tokens": torch.stack([item["task_tokens"] for item in batch]),
            "transformation_suite": [item["transformation_suite"] for item in batch],  # list of lists of strings
        }

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.train_batch_size, 
                          shuffle=self.cfg.data.get("shuffle_train", True),
                          num_workers=self.num_workers, 
                          collate_fn=self.collate_fn
                          )

    def val_dataloader(self):
        loaders = []
        if self.val_id_ds:
            loaders.append(DataLoader(self.val_id_ds, 
                                      batch_size=self.inference_batch_size, 
                                      num_workers=self.num_workers, 
                                      collate_fn=self.collate_fn)
                                      )
            
        if self.use_ood_val and self.val_ood_ds:
            loaders.append(DataLoader(self.val_ood_ds, batch_size=self.inference_batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn))
        
        return loaders

    def test_dataloader(self):
        loaders = []
        if self.test_id_ds:
            loaders.append(DataLoader(self.test_id_ds, 
                                      batch_size=self.inference_batch_size, 
                                      num_workers=self.num_workers, 
                                      collate_fn=self.collate_fn)
                                      )
        
        if self.use_ood_test and self.test_ood_ds:
            loaders.append(DataLoader(self.test_ood_ds, 
                                      batch_size=self.inference_batch_size, 
                                      num_workers=self.num_workers, 
                                      collate_fn=self.collate_fn)
                                      )
        
        return loaders
