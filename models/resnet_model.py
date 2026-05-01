"""
models.resnet_model

Defines ResNetModel, a ResNet model (using a ResNet encoder with an MLP decoder),
implemented as a PTL LightningModule.

- Extracts the 2D grid from the flat input token sequence, encodes it with ResNetEncoder, 
  and projects the per-position embeddings to output logits via an MLPDecoder.

- Training, validation, and test steps compute cross-entropy loss (with optional token weighting). 
  Logs base metrics: token accuracy, grid accuracy, no-pad variants, and object accuracy. 
  Also logs scores for meta-metrics.

- On test epoch end, computes and saves to JSON: per-sample predictions, overall stubbornness,
  compositional gap, per-transformation scatter data, etc.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict
import json

from utility.logging_utils import logger

from models.model_helpers import _extract_evolution_samples, _plot_epoch_grids, _plot_metrics, stce_loss, compute_metrics as _compute_metrics_fn, build_network, maybe_compile, compute_per_transform_id_metrics, log_per_transform_id_metrics


class ModelModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return _compute_metrics_fn(preds, targets, pad_id=self.cfg.data.pad_token_id)

    def decode_sample(self, token_ids: torch.Tensor) -> str:
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "tokenizer"):
            ids = token_ids.detach().cpu().tolist()
            return str(self.trainer.datamodule.tokenizer.decode(ids))
        return str(token_ids.detach().cpu().tolist())

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """
        Cross-entropy / StCE / focal loss with optional foreground class weighting and padding ignored.

        Args:
            logits:  [B, S, C] or already reshaped [B*S, C]
            targets: [B, S]    or already reshaped [B*S]
            reduction: "mean" | "none_per_sample"
                - "mean": scalar loss over all non-padding tokens
                - "none_per_sample": [B] tensor, one value per sample

        Returns:
            loss: scalar or [B] depending on `reduction`

        """
        B = targets.shape[0] if targets.dim() == 2 else None
        S = targets.shape[1] if targets.dim() == 2 else None

        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        weight = getattr(self, "loss_weights", None)
        pad_id = getattr(self, "pad_token_id", -100)

        if self.loss_func == "stce":
            ce = stce_loss(logits_flat, targets_flat, pad_id, weight=weight)  # [B*S]; 0.0 at pad positions

        else:
            ce = F.cross_entropy(
                logits_flat, targets_flat,
                weight=weight,
                ignore_index=pad_id,
                reduction="none"
            )  # [B*S]; 0.0 at ignored (pad) positions

        if self.loss_func == "focal":
            # Down-weight easy examples: (1 - p_t)^gamma * CE
            pt = torch.exp(-ce)
            ce = (1 - pt).pow(self.focal_gamma) * ce

        if reduction == "mean":
            non_pad = (targets_flat != pad_id).float()
            return ce.sum() / non_pad.sum().clamp(min=1)

        # "none_per_sample": average over non-padding positions within each sample
        ce = ce.view(B, S)
        non_pad = (targets != pad_id).float()  # [B, S]

        return ce.sum(dim=1) / non_pad.sum(dim=1).clamp(min=1)  # [B]


class ResNetModel(ModelModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.vocab_size = cfg.model.get("output_vocab_size", None)
        self.d_model = cfg.model.get("d_model", 128)

        # Sync config so sub-modules that read cfg.model.d_model see the correct value
        OmegaConf.set_struct(cfg, False)
        cfg.model.d_model = self.d_model
        OmegaConf.set_struct(cfg, True)

        # Logging Flag
        self.log_samples_flag = cfg.logging.get("log_samples_for_inspection", False)

        # Evolution Tracking Flags
        self.visualize_predictions = cfg.logging.get("visualize_model_predictions", False)
        self.evol_batch_idx = cfg.logging.get("evolution_batch_idx", 0)
        self.evol_indices = cfg.logging.get("evolution_sample_indices", [0, 1, 2])
        self.evolution_records = {"train": [], "val_id": [], "val_ood": []}

        # Metric Tracking
        self.epoch_metrics = []

        logger.info(f"Initializing ResNet (ResNetEncoder + MLP Decoder) with d_model={self.d_model}")

        self.encoder = maybe_compile(build_network(cfg.model.encoder_network, cfg), cfg)
        self.decoder = maybe_compile(build_network(cfg.model.decoder_network, cfg), cfg)

        # --- Loss Function ---
        pad_id = cfg.data.pad_token_id
        output_vocab_size = cfg.model.get("output_vocab_size")
        foreground_weight = cfg.model.get("foreground_weight", 1.0)

        # Class weights: upweight non-background cells (values 1-9) vs background (0)
        if foreground_weight != 1.0:
            loss_weights = torch.ones(output_vocab_size)
            loss_weights[1:] = foreground_weight
            self.register_buffer("loss_weights", loss_weights)
        else:
            self.loss_weights = None

        self.pad_token_id = pad_id
        self.loss_func = cfg.model.get("loss_func", "cross_entropy")
        self.focal_gamma = cfg.model.get("focal_gamma", 2.0)

        # --- Task Conditioning ---
        # Unlike the Transformer (which can attend to task tokens), the ResNet processes only
        # the 2D grid, so task conditioning must be injected explicitly.
        # We embed the task tokens, average them (masking out padding), concatenate the resulting
        # [B, D] task vector to every grid position, then project back to D before the MLP decoder.
        # Concatenation (vs. addition) keeps grid and task features more clearly separated until the
        # decoder, thus letting it learn arbitrary non-linear combinations of both (maybe more expressive than a simple additive bias).
        self.use_task_tokens = cfg.model.get("use_task_tokens", False)
        if self.use_task_tokens:
            self.task_embedding = nn.Embedding(cfg.model.input_vocab_size, self.d_model)
            # Project concatenated [grid_feat || task_feat] back to d_model so the decoder is unchanged
            self.task_proj = nn.Linear(2 * self.d_model, self.d_model)

        # --- Logging ---
        self.epoch_samples = {}
        self.test_outputs = []

    def _extract_grid_2d(self, src: torch.Tensor) -> torch.Tensor:
        """
        Extract the 2D grid from the flat input token sequence and reshape to [B, H, W].

        The src sequence has structure: [BOS] (+ task_tokens) + grid_tokens + [EOS]
        We skip the prefix tokens and take exactly H*W grid tokens.

        """
        h = self.cfg.model.max_h
        w = self.cfg.model.max_w
        grid_len = h * w

        use_tasks = self.cfg.model.get("use_task_tokens", False)
        max_task_len = self.cfg.model.get("max_task_seq_len", 0) if use_tasks else 0

        start_idx = 1 + max_task_len    # skip the <BOS> token and the task tokens (if using task tokens)
        grid_tokens = src[:, start_idx:start_idx + grid_len]  # [B, H*W]

        return grid_tokens.view(-1, h, w)  # [B, H, W]

    def forward(self,
                src: torch.Tensor,
                tgt: Optional[torch.Tensor],
                task_tokens: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass: extract 2D grid → ResNetEncoder → task conditioning → MLP Decoder → Logits

        Args:
            src: input sequence of token IDs [B, S] (includes BOS, optional task tokens, grid tokens, EOS)
            tgt: target sequence of token IDs [B, S_grid]; unused here, kept for signature compatibility
            task_tokens: [B, max_task_seq_len] task token IDs including identity/pad tokens;
                         mean-pooled and concatenated to grid features so the model knows what task to
                         perform and at what depth (ResNet cannot attend to them directly)

        Returns:
            logits: [B, S_grid, output_vocab_size]
            
        """
        grid_2d = self._extract_grid_2d(src)    # [B, H, W]
        x = self.encoder(grid_2d)   # [B, H*W, D]

        # Task conditioning: embed all task tokens (including identity/pad tokens) and mean-pool.
        # Task padding tokens represent "no transformation" (identity), so including
        # them in the mean encodes task depth.
        if self.use_task_tokens and task_tokens is not None:
            task_vec = self.task_embedding(task_tokens).mean(dim=1) # [B, D]
            task_vec = task_vec.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, H*W, D]
            x = self.task_proj(torch.cat([x, task_vec], dim=-1))    # [B, H*W, D]

        logits = self.decoder(x)    # [B, H*W, output_vocab_size]
        
        return logits


    # ------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------
    def on_fit_start(self):
        try:
            from torchinfo import summary
            max_seq_len = self.cfg.model.max_seq_len
            grid_len = self.cfg.model.max_h * self.cfg.model.max_w
            device = self.device
            src_dummy = torch.zeros(1, max_seq_len, dtype=torch.long, device=device)
            tgt_dummy = torch.zeros(1, grid_len, dtype=torch.long, device=device)
            # torch.compile wraps modules as OptimizedModule which torchinfo cannot trace;
            # temporarily swap in the original uncompiled modules for the FLOPs pass.
            orig_encoder, orig_decoder = self.encoder, self.decoder
            try:
                self.encoder = getattr(self.encoder, "_orig_mod", self.encoder)
                self.decoder = getattr(self.decoder, "_orig_mod", self.decoder)
                stats = summary(self, input_data=[src_dummy, tgt_dummy], verbose=0, depth=0)
            finally:
                self.encoder, self.decoder = orig_encoder, orig_decoder
            import wandb
            if wandb.run:
                wandb.summary["model/flops"] = stats.total_mult_adds
                wandb.summary["model/params"] = stats.total_params
            logger.info(f"Model FLOPs: {stats.total_mult_adds:,} | Params: {stats.total_params:,}")
        except Exception as e:
            logger.warning(f"FLOPs computation failed: {e}")

    def on_train_epoch_start(self):
        self.epoch_samples["train"] = {"first": None, "last": None}

    def on_validation_epoch_start(self):
        self.epoch_samples["val_id"] = {"first": None, "last": None}
        self.epoch_samples["val_ood"] = {"first": None, "last": None}

    def on_test_epoch_start(self):
        self.test_outputs = []


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        src = batch["src"]  # [B, S]
        tgt = batch["tgt"]  # [B, S_grid + 1]
        task_tokens = batch["task_tokens"]  # [B, max_task_seq_len]
        # transformation_suite not needed during training

        # Remove EOS from target
        tgt_grid = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        logits = self(src, tgt_grid, task_tokens=task_tokens)   # [B, S_grid, output_vocab_size]

        preds = torch.argmax(logits, dim=-1)    # [B, S_grid]; get predicted discrete tokens

        loss = self._compute_loss(logits, tgt_grid, reduction="mean")

        # Compute metrics
        metrics = self.compute_metrics(preds, tgt_grid)

        # Log metrics (on_step=True allows seeing fluctuations during epoch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", metrics["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc", metrics["grid_acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_no_pad", metrics["acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/obj_acc", metrics["obj_acc"], on_step=True, on_epoch=True, prog_bar=True)

        if "train" not in self.epoch_samples:
            self.epoch_samples["train"] = {"first": None, "last": None}

        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples["train"]["first"] = (src[0], tgt_grid[0], preds[0])
            self.epoch_samples["train"]["last"] = (src[-1], tgt_grid[-1], preds[-1])

        # Track sample prediction evolution
        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, tgt_grid, preds)
            if record:
                self.evolution_records["train"].append({
                    "epoch": self.current_epoch,
                    "samples": record
                })

        return loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src         = batch["src"]
        tgt         = batch["tgt"]
        task_tokens = batch["task_tokens"]
        # transformation_suite not needed during validation

        tgt_grid = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        logits = self(src, tgt_grid, task_tokens=task_tokens)
        loss   = self._compute_loss(logits, tgt_grid, reduction="mean")
        preds  = torch.argmax(logits, dim=-1)

        metrics = self.compute_metrics(preds, tgt_grid)

        # Determine prefix based on dataloader index
        # 0 -> ID, 1 -> OOD (assuming DataModule returns [id_loader, ood_loader])
        prefix = "id" if dataloader_idx == 0 else "ood"

        # Log with prog_bar=True so they show up in console
        self.log(f"val/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_obj_acc", metrics["obj_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)

        key = f"val_{prefix}"
        if key not in self.epoch_samples:
            self.epoch_samples[key] = {"first": None, "last": None}

        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples[key]["first"] = (src[0], tgt_grid[0], preds[0])
            self.epoch_samples[key]["last"] = (src[-1], tgt_grid[-1], preds[-1])

        # Track Evolution
        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, tgt_grid, preds)
            if record:
                self.evolution_records[key].append({
                    "epoch": self.current_epoch,
                    "samples": record
                })

        return loss

    def _record_epoch_metrics(self):
        if self.trainer.sanity_checking:
            return

        # Grab current scalar metrics from PTL callback_metrics
        current_metrics = {
            k: v.item() for k, v in self.trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        current_metrics["epoch"] = self.current_epoch

        # Update current epoch entry if exists, else append
        if len(self.epoch_metrics) > 0 and self.epoch_metrics[-1]["epoch"] == self.current_epoch:
            self.epoch_metrics[-1].update(current_metrics)
        else:
            self.epoch_metrics.append(current_metrics)

    def on_train_epoch_end(self):
        if "train" in self.epoch_samples:
            self._log_samples("Train", self.epoch_samples.get("train"))

        # Draw and Save Grids
        if self.evolution_records["train"] and self.evolution_records["train"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "train", self.current_epoch, self.evolution_records["train"][-1]["samples"])

        self._record_epoch_metrics()

    def on_validation_epoch_end(self):
        if "val_id" in self.epoch_samples:
            self._log_samples("Validation (ID)", self.epoch_samples.get("val_id"))

        if "val_ood" in self.epoch_samples and self.epoch_samples["val_ood"].get("last") is not None:
            self._log_samples("Validation (OOD)", self.epoch_samples.get("val_ood"))

        # Draw and Save Grids
        if self.evolution_records["val_id"] and self.evolution_records["val_id"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "val_id", self.current_epoch, self.evolution_records["val_id"][-1]["samples"])
        if self.evolution_records["val_ood"] and self.evolution_records["val_ood"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "val_ood", self.current_epoch, self.evolution_records["val_ood"][-1]["samples"])

        self._record_epoch_metrics()

    # ------------------------------------------------------------------
    # End of Training - Artifact Handling
    # ------------------------------------------------------------------
    def on_fit_end(self):
        cwd = os.getcwd()

        # Save Evolution JSON
        evol_file = os.path.join(cwd, "prediction_evolution.json")
        try:
            with open(evol_file, "w") as f:
                json.dump(self.evolution_records, f, indent=2)
            logger.info(f"Saved prediction evolution to {evol_file}")
        except Exception as e:
            logger.error(f"Failed to save prediction evolution: {e}")

        # Save Metrics JSON
        metrics_file = os.path.join(cwd, "metrics_history.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(self.epoch_metrics, f, indent=2)
            logger.info(f"Saved metrics history to {metrics_file}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {e}")

        # Generate and save plots
        _plot_metrics(self, cwd)

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        src = batch["src"]
        tgt = batch["tgt"]
        task_tokens = batch["task_tokens"]
        transformation_suites = batch["transformation_suite"]  # list of lists of strings, length B

        tgt_grid = tgt[:, :-1]  # [B, S_grid]
        B = src.size(0)
        prefix = "id" if dataloader_idx == 0 else "ood"

        logits = self(src, tgt_grid, task_tokens=task_tokens)
        preds = torch.argmax(logits, dim=-1)

        # Per-sample values for the JSON output
        ps_losses = self._compute_loss(logits, tgt_grid, reduction="none_per_sample")                         # [B]
        ps_metrics = _compute_metrics_fn(preds, tgt_grid, pad_id=self.cfg.data.pad_token_id, per_sample=True)  # dict of [B]

        # Batch aggregates for PTL logging
        loss = ps_losses.mean()
        metrics = {k: v.mean() for k, v in ps_metrics.items()}

        self.log(f"test/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_obj_acc", metrics["obj_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)

        src_cpu = src.detach().cpu()
        tgt_cpu = tgt.detach().cpu()
        preds_cpu = preds.detach().cpu()
        ps_losses_cpu = ps_losses.detach().cpu()
        ps_metrics_cpu = {k: v.detach().cpu() for k, v in ps_metrics.items()}

        for i in range(B):
            self.test_outputs.append({
                "domain_type": prefix,
                "input_raw": src_cpu[i].tolist(),
                "target_raw": tgt_cpu[i].tolist(),
                "prediction_raw": preds_cpu[i].tolist(),
                "input_decoded": self.decode_sample(src_cpu[i]),
                "target_decoded": self.decode_sample(tgt_cpu[i]),
                "prediction_decoded": self.decode_sample(preds_cpu[i]),
                "transformation_suite": transformation_suites[i],
                f"{prefix}_loss": ps_losses_cpu[i].item(),
                f"{prefix}_acc": ps_metrics_cpu["acc"][i].item(),
                f"{prefix}_grid_acc": ps_metrics_cpu["grid_acc"][i].item(),
                f"{prefix}_acc_no_pad": ps_metrics_cpu["acc_no_pad"][i].item(),
                f"{prefix}_grid_acc_no_pad": ps_metrics_cpu["grid_acc_no_pad"][i].item(),
                f"{prefix}_obj_acc": ps_metrics_cpu["obj_acc"][i].item(),
            })

    def on_test_epoch_end(self):
        id_outputs = [o for o in self.test_outputs if o["domain_type"] == "id"]
        ood_outputs = [o for o in self.test_outputs if o["domain_type"] == "ood"]
        matched = id_outputs and ood_outputs and len(id_outputs) == len(ood_outputs)
        metric_keys = ["acc", "grid_acc", "acc_no_pad", "grid_acc_no_pad", "obj_acc"]

        # --- 1. Overall Stubbornness ---
        stubbornness_metrics = {}
        if matched:
            ood_preds = torch.tensor([o["prediction_raw"] for o in ood_outputs], dtype=torch.long)
            id_targets = torch.tensor([o["target_raw"][:-1] for o in id_outputs], dtype=torch.long)
            stub = _compute_metrics_fn(ood_preds, id_targets, pad_id=self.cfg.data.pad_token_id)
            stubbornness_metrics = {f"stubbornness_{k}": v.item() for k, v in stub.items()}
            logger.info("Stubbornness: " + ", ".join(f"{k}={v:.4f}" for k, v in stubbornness_metrics.items()))
        elif id_outputs and ood_outputs:
            logger.warning(f"Stubbornness skipped: ID ({len(id_outputs)}) and OOD ({len(ood_outputs)}) counts mismatch.")

        # --- 2. Compositional Gap (ID metric − OOD metric for each metric key) ---
        comp_gap = {}
        if id_outputs and ood_outputs:
            id_avg = {k: sum(o[f"id_{k}"] for o in id_outputs) / len(id_outputs) for k in metric_keys}
            ood_avg = {k: sum(o[f"ood_{k}"] for o in ood_outputs) / len(ood_outputs) for k in metric_keys}
            comp_gap = {f"gap_{k}": id_avg[k] - ood_avg[k] for k in metric_keys}
            logger.info("Compositional gap: " + ", ".join(f"{k}={v:.4f}" for k, v in comp_gap.items()))

        # --- 3. Per-transformation Stubbornness + OOD accuracy (scatter plot data) ---
        # For each transformation type: stubbornness (OOD preds vs ID targets) and OOD accuracy
        # (OOD preds vs OOD targets). Together they give the scatter: x=OOD_acc, y=stubbornness.
        per_transform_stub = {}
        if ood_outputs:
            # Group OOD samples by transformation suite key
            ood_groups: Dict = {}
            for ood_o in ood_outputs:
                key = "|".join(ood_o.get("transformation_suite", []))
                ood_groups.setdefault(key, {"ood_preds": [], "ood_targets": []})
                ood_groups[key]["ood_preds"].append(ood_o["prediction_raw"])
                ood_groups[key]["ood_targets"].append(ood_o["target_raw"][:-1])  # strip EOS

            # Group matched ID samples by transformation suite key (same order as OOD)
            id_groups: Dict = {}
            if matched:
                for id_o, ood_o in zip(id_outputs, ood_outputs):
                    key = "|".join(ood_o.get("transformation_suite", []))
                    id_groups.setdefault(key, {"id_targets": []})
                    id_groups[key]["id_targets"].append(id_o["target_raw"][:-1])

            for key, ood_g in ood_groups.items():
                ood_p = torch.tensor(ood_g["ood_preds"], dtype=torch.long)
                ood_t = torch.tensor(ood_g["ood_targets"], dtype=torch.long)

                # OOD accuracy for this transformation type
                ood_metrics = _compute_metrics_fn(ood_p, ood_t, pad_id=self.cfg.data.pad_token_id)
                entry = {f"ood_{k}": v.item() for k, v in ood_metrics.items()}

                # Stubbornness for this transformation type (requires matched ID targets)
                if key in id_groups:
                    id_t = torch.tensor(id_groups[key]["id_targets"], dtype=torch.long)
                    stub = _compute_metrics_fn(ood_p, id_t, pad_id=self.cfg.data.pad_token_id)
                    entry.update({f"stubbornness_{k}": v.item() for k, v in stub.items()})

                per_transform_stub[key] = entry

            logger.info(f"Per-transformation scatter data computed for {len(per_transform_stub)} transformation type(s).")

        # --- 4. Per-transformation ID metrics ---
        per_transform_id = compute_per_transform_id_metrics(id_outputs)
        per_transform_id_scalars = log_per_transform_id_metrics(per_transform_id, self.log)

        # --- Log to WandB ---
        for k, v in stubbornness_metrics.items():
            self.log(f"test/{k}", v)
        for k, v in comp_gap.items():
            self.log(f"test/{k}", v)

        # --- Save JSON ---
        output_data = {
            "samples": self.test_outputs,
            "stubbornness": stubbornness_metrics,
            "compositional_gap": comp_gap,
            "per_transformation_stubbornness": per_transform_stub,
            "per_transformation_id": per_transform_id,
            "per_transformation_id_scalars": per_transform_id_scalars,
        }

        output_file = os.path.join(os.getcwd(), "test_predictions.json")
        try:
            with open(output_file, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Predictions saved to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save prediction JSON: {e}")

    def _log_samples(self, phase_name: str, samples_dict: Dict):
        if not self.log_samples_flag:
            return
        if not samples_dict:
            return

        if samples_dict.get("first") is not None:
            logger.info(f"--- {phase_name} Visualization (Epoch {self.current_epoch}) ---")
            for position in ["first", "last"]:
                data = samples_dict.get(position)
                if data:
                    src, tgt, pred = data
                    logger.info(f"[{position.upper()} Sample]")
                    logger.info(f"  Input:  {self.decode_sample(src)}")
                    logger.info(f"  Target: {self.decode_sample(tgt)}")
                    logger.info(f"  Pred:   {self.decode_sample(pred)}")


    def configure_optimizers(self):
        """
        Initializes the optimizer and the learning rate scheduler.

        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        Returns:
            optimizer_config (dict): A dictionary containing the optimizer and the learning rate scheduler.
        
        """

        base_lr = self.cfg.model.lr

        # Single param group. ResNetEncoder has no separate embedding layer
        param_groups = [
            {"params": list(self.parameters()), "lr": base_lr, "initial_lr": base_lr},
        ]

        # Define the optimizer
        if self.cfg.model.optimizer == 'adam':
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.cfg.model.weight_decay)

        elif self.cfg.model.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))

        elif self.cfg.model.optimizer == 'sgd':
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.cfg.model.weight_decay)

        else:
            raise ValueError(f"Unknown optimizer given: {self.cfg.model.optimizer}")

        # Define the learning rate scheduler
        if self.cfg.model.lr_scheduler.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        elif self.cfg.model.lr_scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs, eta_min=1e-6)

        elif self.cfg.model.lr_scheduler.type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        else:
            raise ValueError(f"Unknown scheduler given: {self.cfg.model.lr_scheduler.type}")

        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": self.cfg.model.lr_scheduler.interval,    # 'epoch' or 'step'
            "frequency": self.cfg.model.lr_scheduler.frequency,  # how often to call the scheduler w.r.t. the interval
            "monitor": self.cfg.model.lr_scheduler.monitored_metric,  # metric to track for lr scheduling
        }

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

        return optimizer_config

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override optimizer_step to apply linear LR warmup under PTL automatic optimization.
        """
        if self.cfg.model.get("lr_warmup", {}).get("enabled", False):
            if self.cfg.model.lr_warmup.type == "linear":
                num_warmup = self.cfg.model.lr_warmup.num_steps
                if self.trainer.global_step < num_warmup:
                    lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_warmup)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_scale * pg["initial_lr"]
            else:
                raise ValueError(f"Unknown LR warmup type given: {self.cfg.model.lr_warmup.type}")

        optimizer.step(closure=optimizer_closure)
