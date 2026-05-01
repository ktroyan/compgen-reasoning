"""
models.trm_model

- Defines TRMModel, a TRM model (TRM encoder based on the TRM paper, and MLP decoder),
  implemented as a PTL LightningModule.

- The model is composed of a TRMEncoder (as per the TRM paper) and a TRMDecoder (MLP that
  projects embeddings to output logits).

- As per TRM, a Q-head (linear layer on the EOS token embedding) decides per-sample when to halt recursion.

- Uses manual optimization so that a backward pass and optimizer step are performed once per
  supervision step inside the supervision loop; matching the TRM algorithm.

- Training, validation, and test steps compute cross-entropy loss (with optional token weighting) plus Q-loss. 
  Logs base metrics: token accuracy, grid accuracy, no-pad variants, and object accuracy. 
  Also logs scores for meta-metrics.

- On test epoch end, computes and saves to JSON: per-sample predictions, overall stubbornness,
  compositional gap, halting step distribution, Q-head calibration, iterative accuracy curve, per-transformation scatter data, etc.

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict
import json

## Personal imports
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

class TRMModel(ModelModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self.vocab_size = cfg.model.get("output_vocab_size", None)
        input_dim = cfg.model.get("d_model", 128)
        self.d_model = input_dim
        
        # Sync config
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

        logger.info(f"Initializing TRM (Encoder + MLP Decoder) with d_model={self.d_model}")
        
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
        self.q_loss_weight = cfg.model.get("q_loss_weight", 1.0)

        # --- Logging ---
        self.epoch_samples = {}
        self.test_outputs = []

        # --- TRM specifics ---
        # Instantiate and initialize the Q-head used to decide when halting recursion
        self.q_head = nn.Linear(self.encoder.d_model, 1)

        # Manual optimization: we call backward + opt.step() inside the supervision (N_sup iterations) loop,
        # once per supervision step, as per the TRM paper's training procedure.
        self.automatic_optimization = False
    

    def forward_features_for_MLP_decoder(self, x: torch.Tensor) -> torch.Tensor:
        """
        Output relevant feature embeddings of the encoded sequence x.
        Essentially, the grid token embeddings.

        For TRM, this is in fact the proposed answer y by the encoder with iterative refinement.

        """
        # The encoder's output contains embeddings for grid tokens, but also for special tokens such as <BOS>, <EOS>, and task tokens.
        # Since an MLP decodes in parallel for each position, we only want to keep the embeddings corresponding to the grid tokens.
        # This will allow the outputted/encoded sequence to be the same length as the target sequence
        # (which is [<grid_tokens>, <EOS>] and for which we will discard the <EOS> token when computing the loss).
        
        h = self.cfg.model.max_h
        w = self.cfg.model.max_w
        grid_len = h * w
        
        # Check if task tokens actually exist in the sequence
        use_tasks = self.cfg.model.get("use_task_tokens", False)
        max_task_len = self.cfg.model.get("max_task_seq_len", 0) if use_tasks else 0

        start_idx = 1 + max_task_len    # skip the <BOS> token and the task tokens (if using task tokens)
        end_idx = start_idx + grid_len  # keep the grid token embeddings and discard the token embeddings that come after such as the <EOS> token embedding
        
        return x[:, start_idx:end_idx, :]
    
    def forward(self, 
                x: torch.Tensor, 
                tgt: Optional[torch.Tensor],
                y: Optional[torch.Tensor] = None,
                z: Optional[torch.Tensor] = None,
                task_tokens: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        """
        Forward pass: Encoder -> Decoder -> Logits

        Args:
            x: input sequence of token IDs; [B, S], where is S_grid + num_special_tokens (e.g., <BOS>, <EOS>, task tokens, etc.)
            tgt: target sequence of token IDs [B, S_grid + 1] (including <EOS> token at the end); may be needed for teacher forcing in the decoder
            y: carry state for the answer (initialized as a copy of x for the first outer step)
            z: carry state for the latent state (initialized as zeros for the first outer step)
            task_tokens: optional task tokens to condition on (if using task tokens, they are included in the input sequence x from when the data is loaded in cogitao_data.py)

        Returns:
            logits: output logits from the decoder [B, S_grid, output_vocab_size]
            q_logits: output logits from the Q-head for halting decision [B, 1]
            y_next: next carry state for the answer (detached from the computation graph for the next outer step)
            z_next: next carry state for the latent state (detached from the computation graph for the next outer step)
        
        """

        # Compute the encoder output y and the next carry states for recursion (which are detached from the computation graph to prevent gradients from flowing back through them in the next outer step)
        y_grad, y_next, z_next = self.encoder(
            x,  # [B, S]
            y=y,    # [B, S]
            z=z # None or [B, S, d_model]
        )

        # TODO: see if should use full grid OR full grid + input sequence OR just one token (e.g., <EOS> token)
        # NOTE: we pad the 2D grids, so the there is no padding needed for the sequence and thus the <EOS> token is always at the end
        q_logits = self.q_head(y_grad[:, -1])  # use EOS token embedding to decide when to halt recursion; q_logits is of shape [B, 1]

        # For MLP decoder, we only want to use the encoded grid tokens' embeddings, and not the embeddings of the special tokens such as BOS and EOS
        # This is because the MLP decoder decodes in parallel for each position and thus the input sequence to the MLP decoder should be the same length as the target sequence (which is [<grid_tokens>, <EOS>] and for which we will discard the <EOS> token when computing the loss)
        y_grad = self.forward_features_for_MLP_decoder(y_grad) # [B, S_grid, d_model] <- [B, S, d_model]

        logits = self.decoder(y_grad)   # logits should be of shape [B, S_grid, output_dim] where S_grid is the number of grid tokens in the target sequence

        return logits, q_logits, y_next, z_next


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
            # torch.compile wraps modules as OptimizedModule which torchinfo can't trace;
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

        # {prefix: {step_idx (0-based): [grid_acc values, one per batch]}}
        # so each outer key is a domain ("id" or "ood"), each inner key is a supervision step index (0-based),
        # and the value is a list of grid accuracy values (one per batch) that get averaged in on_test_epoch_end to produce the iterative accuracy curve
        self.iter_acc_records = {"id": {}, "ood": {}}


    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        """
        Using manual optimization to match the TRM paper's training procedure of doing a backward pass and optimizer step once per supervision step within the N_sup loop, rather than after the loop as is standard in PTL automatic optimization.
        """

        opt = self.optimizers()

        src = batch["src"]  # [B, S]
        tgt = batch["tgt"]  # [B, S_grid + 1]
        task_tokens = batch["task_tokens"] # [B, max_task_seq_len]
        # transformation_suite not needed during training

        # Remove EOS from target
        tgt_grid = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        B = src.size(0)

        y = None  # we let the encoder initialize the carry state for the answer as a copy of the input sequence embeddings
        z = None  # we let encoder initialize the carry state for the latent state

        halted = torch.zeros(B, dtype=torch.bool, device=src.device)

        total_loss = 0.0
        final_preds = None

        for _ in range(self.cfg.model.get("N_sup", 1)):

            logits, q_logits, y_next, z_next = self(src, tgt_grid, y, z, task_tokens=task_tokens)

            preds = torch.argmax(logits, dim=-1)    # [B, S_grid]; get predicted discrete tokens

            with torch.no_grad():
                seq_correct = (preds == tgt_grid).all(dim=-1)    # [B]; check if the entire sequence is correct for each sample in the batch

            # Loss per sample (e.g., cross-entropy with class weights and padding ignored)
            ce_loss = self._compute_loss(logits, tgt_grid, reduction="none_per_sample")  # [B]

            # Q-loss per sample
            q_loss = F.binary_cross_entropy_with_logits(
                q_logits.squeeze(-1),
                seq_correct.float(),
                reduction="none"
            )  # [B]

            loss_per_sample = ce_loss + (self.q_loss_weight * q_loss)

            # Only accumulate loss for active samples
            active_mask = ~halted
            supervision_step_loss = (loss_per_sample * active_mask.float()).mean()

            # Backward + optimizer step per supervision step, as per the TRM paper
            opt.zero_grad()
            self.manual_backward(supervision_step_loss)

            # LR warmup (inlined here since optimizer_step is not called with manual optimization)
            if self.cfg.model.get("lr_warmup", {}).get("enabled", False):
                if self.cfg.model.lr_warmup.type == "linear":
                    num_warmup = self.cfg.model.lr_warmup.num_steps
                    if self.trainer.global_step < num_warmup:
                        lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_warmup)
                        for pg in opt.param_groups:
                            pg["lr"] = lr_scale * pg["initial_lr"]

            self.clip_gradients(opt, gradient_clip_val=self.cfg.training.get("gradient_clip_val", 1.0), gradient_clip_algorithm="norm") # manual gradient clipping since using manual optimization
            opt.step()  # NOTE: the trainer counter for steps (self.trainer.global_step) is only incremented after the training_step method call, so the effective global step when using manual optimization is the same but times N_sup

            total_loss = total_loss + supervision_step_loss.detach()

            # Update halting state
            newly_halted = (q_logits.squeeze(-1) > 0)
            halted = halted | newly_halted
            halted_mask = halted.view(B, 1, 1)  # [B, 1, 1] to broadcast over S and D

            # Update states only for active samples
            if y is None:
                y = y_next

            else:
                y = torch.where(
                    halted_mask,  # broadcast along seq_len and embedding dims
                    y,
                    y_next
                )

            if z is None:
                z = z_next

            else:
                z = torch.where(
                    halted_mask,  # broadcast along seq_len and embedding dims
                    z,
                    z_next
                )

            final_preds = preds

            # Halt the computation for the batch if all the samples in the batch have halted
            if halted.all():
                break


        # Compute metrics
        metrics = self.compute_metrics(final_preds, tgt_grid)

        # Log metrics (on_step=True allows seeing fluctuations during epoch)
        self.log("train/loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", metrics["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc", metrics["grid_acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_no_pad", metrics["acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/obj_acc", metrics["obj_acc"], on_step=True, on_epoch=True, prog_bar=True)

        if "train" not in self.epoch_samples:
            self.epoch_samples["train"] = {"first": None, "last": None}
        
        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples["train"]["first"] = (src[0], tgt_grid[0], final_preds[0])
            self.epoch_samples["train"]["last"] = (src[-1], tgt_grid[-1], final_preds[-1])
        
        # Track sample prediction evolution
        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, tgt_grid, final_preds)
            if record:
                self.evolution_records["train"].append({
                    "epoch": self.current_epoch,
                    "samples": record
                })

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src         = batch["src"]
        tgt         = batch["tgt"]
        task_tokens = batch["task_tokens"]
        # transformation_suite not needed during validation

        tgt_grid = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        B = src.size(0)

        y = None  # we let the encoder initialize the carry state for the answer as a copy of the input sequence embeddings
        z = None  # we let encoder initialize the carry state for the latent state

        halted = torch.zeros(B, dtype=torch.bool, device=src.device)

        with torch.no_grad():

            for _ in range(self.cfg.model.N_sup):

                logits, q_logits, y_next, z_next = self(
                    src, tgt_grid, y, z, task_tokens=task_tokens
                )

                newly_halted = (q_logits.squeeze(-1) > 0)
                halted = halted | newly_halted
                halted_mask = halted.view(B, 1, 1)  # [B, 1, 1] to broadcast over S and D

                # Update states only for active samples
                if y is None:
                    y = y_next

                else:
                    y = torch.where(
                        halted_mask,  # broadcast along seq_len and embedding dims
                        y,
                        y_next
                    )

                if z is None:
                    z = z_next

                else:
                    z = torch.where(
                        halted_mask,  # broadcast along seq_len and embedding dims
                        z,
                        z_next
                    )

                if halted.all():
                    break

            loss = self._compute_loss(logits, tgt_grid, reduction="mean")

            preds = torch.argmax(logits, dim=-1)

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
        # With manual optimization, PTL does not auto-step the LR scheduler, so we do it here.
        # With automatic optimization, PTL steps it automatically, so we must not step it again
        if self.cfg.training.get("use_manual_optimization", False):
            sch = self.lr_schedulers()
            if sch is not None and not isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch.step()

        if "train" in self.epoch_samples:
            self._log_samples("Train", self.epoch_samples.get("train"))

        # Draw and Save Grids
        if self.evolution_records["train"] and self.evolution_records["train"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "train", self.current_epoch, self.evolution_records["train"][-1]["samples"])
            
        self._record_epoch_metrics()
    
    def on_validation_epoch_end(self):
        # ReduceLROnPlateau requires a metric value — step it here after validation metrics are available.
        if self.cfg.training.get("use_manual_optimization", False):
            sch = self.lr_schedulers()
            if sch is not None and isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
                monitored = self.cfg.model.lr_scheduler.monitored_metric
                metric_val = self.trainer.callback_metrics.get(monitored)
                if metric_val is not None:
                    sch.step(metric_val)

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
        N_sup = self.cfg.model.N_sup
        prefix = "id" if dataloader_idx == 0 else "ood"

        y = None
        z = None
        halted = torch.zeros(B, dtype=torch.bool, device=src.device)
        # Default halt_step = N_sup (sample used all steps, or never explicitly halted)
        halt_step = torch.full((B,), N_sup, dtype=torch.long, device=src.device)
        # Whether each sample's prediction was correct at the step it halted
        halt_correct = torch.zeros(B, dtype=torch.bool, device=src.device)

        iter_grid_accs = []  # grid accuracy at each supervision step for this batch

        with torch.no_grad():
            for step_idx in range(N_sup):
                logits, q_logits, y_next, z_next = self(src, tgt_grid, y, z, task_tokens=task_tokens)

                preds = torch.argmax(logits, dim=-1)

                # Iterative accuracy: record grid accuracy at this step for all samples
                iter_grid_accs.append(self.compute_metrics(preds, tgt_grid)["grid_acc"].item())

                # Per-sample correctness at this step (full-grid match)
                correct_now = (preds == tgt_grid).all(dim=-1)  # [B]

                # Halting: record the first step at which each sample halts + its correctness
                newly_halted = (q_logits.squeeze(-1) > 0)
                first_halt = newly_halted & ~halted           # samples halting for the first time
                halt_step[first_halt] = step_idx + 1         # 1-indexed
                halt_correct[first_halt] = correct_now[first_halt]
                halted = halted | newly_halted
                halted_mask = halted.view(B, 1, 1)

                if y is None:
                    y = y_next
                else:
                    y = torch.where(halted_mask, y, y_next)

                if z is None:
                    z = z_next
                else:
                    z = torch.where(halted_mask, z, z_next)

                if halted.all():
                    break

            # For samples that never explicitly halted, record correctness of the final prediction
            never_halted = (halt_step == N_sup)
            halt_correct[never_halted] = correct_now[never_halted]

            ps_losses = self._compute_loss(logits, tgt_grid, reduction="none_per_sample")   # [B]
            ps_metrics = _compute_metrics_fn(preds, tgt_grid, pad_id=self.cfg.data.pad_token_id, per_sample=True)  # dict of [B]

        # ps_losses and ps_metrics are from the last executed step
        loss = ps_losses.mean()
        metrics = {k: v.mean() for k, v in ps_metrics.items()}

        # Accumulate per-step iterative accuracy records for on_test_epoch_end
        for s, acc in enumerate(iter_grid_accs):
            self.iter_acc_records[prefix].setdefault(s, []).append(acc)

        self.log(f"test/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_obj_acc", metrics["obj_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)

        halt_step_list = halt_step.cpu().tolist()
        halt_correct_list = halt_correct.cpu().tolist()
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
                "halt_step": halt_step_list[i],
                "halt_correct": halt_correct_list[i],  # was prediction correct at halt step?
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
        N_sup = self.cfg.model.N_sup

        # ------------------------------------------------------------------
        # Meta-metric: Stubbornness (for each base metric)
        # ------------------------------------------------------------------
        stubbornness_metrics = {}
        if matched:
            ood_preds = torch.tensor([o["prediction_raw"] for o in ood_outputs], dtype=torch.long)
            id_targets = torch.tensor([o["target_raw"][:-1] for o in id_outputs], dtype=torch.long)
            stub = _compute_metrics_fn(ood_preds, id_targets, pad_id=self.cfg.data.pad_token_id)
            stubbornness_metrics = {f"stubbornness_{k}": v.item() for k, v in stub.items()}
            logger.info("Stubbornness: " + ", ".join(f"{k}={v:.4f}" for k, v in stubbornness_metrics.items()))
        elif id_outputs and ood_outputs:
            logger.warning(f"Stubbornness skipped: ID ({len(id_outputs)}) and OOD ({len(ood_outputs)}) counts mismatch.")

        # --- Scatter plot data: Per-transformation Stubbornness + OOD accuracy ---
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

        # ------------------------------------------------------------------
        # Per-transformation ID metrics
        # ------------------------------------------------------------------
        per_transform_id = compute_per_transform_id_metrics(id_outputs)
        per_transform_id_scalars = log_per_transform_id_metrics(per_transform_id, self.log)

        # ------------------------------------------------------------------
        # Meta-metric: Compositional Gap (ID metric - OOD metric for each basemetric)
        # ------------------------------------------------------------------
        comp_gap = {}
        if id_outputs and ood_outputs:
            id_avg = {k: sum(o[f"id_{k}"] for o in id_outputs) / len(id_outputs) for k in metric_keys}
            ood_avg = {k: sum(o[f"ood_{k}"] for o in ood_outputs) / len(ood_outputs) for k in metric_keys}
            comp_gap = {f"gap_{k}": id_avg[k] - ood_avg[k] for k in metric_keys}
            logger.info("Compositional gap: " + ", ".join(f"{k}={v:.4f}" for k, v in comp_gap.items()))

        # ------------------------------------------------------------------
        # TRM-specific metric: Halting Step Distribution (mean, std, per-step histogram)
        # ------------------------------------------------------------------
        halting_distribution = {}
        for pfx, outputs in [("id", id_outputs), ("ood", ood_outputs)]:
            steps = [o["halt_step"] for o in outputs if "halt_step" in o]
            if steps:
                n = len(steps)
                mean = sum(steps) / n
                std = (sum((x - mean) ** 2 for x in steps) / n) ** 0.5
                halting_distribution[pfx] = {
                    "mean": mean,
                    "std": std,
                    "histogram": {str(k): steps.count(k) for k in range(1, N_sup + 1)},
                }
                logger.info(f"Halting distribution ({pfx}): mean={mean:.2f}, std={std:.2f}")

        # ------------------------------------------------------------------
        # TRM-specific metric: Q-head Calibration (precision of the halting signal)
        # ------------------------------------------------------------------
        # Among samples that halted at step k, what fraction were actually correct?
        # - overall_precision: n_correct_at_halt / n_total
        # - early_precision: among those that halted before the last step, n_correct / n_early_halted
        # A well-calibrated Q-head halts only when the answer is correct.
        q_head_calibration = {}
        for pfx, outputs in [("id", id_outputs), ("ood", ood_outputs)]:
            if outputs:
                n_total = len(outputs)
                n_halt_correct = sum(1 for o in outputs if o.get("halt_correct", False))
                early_halters = [o for o in outputs if o.get("halt_step", N_sup) < N_sup]
                n_early = len(early_halters)
                n_early_correct = sum(1 for o in early_halters if o.get("halt_correct", False))

                q_head_calibration[pfx] = {
                    "overall_precision": n_halt_correct / n_total,
                    "early_halt_rate": n_early / n_total,
                    "early_precision": n_early_correct / n_early if n_early > 0 else None,
                }
                logger.info(
                    f"Q-head calibration ({pfx}): "
                    f"overall_precision={n_halt_correct / n_total:.4f}, "
                    f"early_halt_rate={n_early / n_total:.4f}, "
                    f"early_precision={n_early_correct / n_early:.4f}" if n_early > 0
                    else f"Q-head calibration ({pfx}): no early halts observed."
                )

        # ------------------------------------------------------------------
        # TRM-specific metric: Iterative Accuracy Curve (grid_acc at each supervision step)
        # ------------------------------------------------------------------
        iter_acc = {}
        for pfx, records in self.iter_acc_records.items():
            if records:
                iter_acc[pfx] = {
                    str(s + 1): sum(vals) / len(vals)   # 1-indexed step → average grid_acc across batches
                    for s, vals in sorted(records.items())
                }

        # ------------------------------------------------------------------
        # Log to WandB
        # ------------------------------------------------------------------
        for k, v in stubbornness_metrics.items():
            self.log(f"test/{k}", v)
        for k, v in comp_gap.items():
            self.log(f"test/{k}", v)
        for pfx, dist in halting_distribution.items():
            self.log(f"test/{pfx}_halt_step_mean", dist["mean"])
            self.log(f"test/{pfx}_halt_step_std", dist["std"])
        for pfx, cal in q_head_calibration.items():
            self.log(f"test/{pfx}_qhead_overall_precision", cal["overall_precision"])
            self.log(f"test/{pfx}_qhead_early_halt_rate", cal["early_halt_rate"])
            if cal["early_precision"] is not None:
                self.log(f"test/{pfx}_qhead_early_precision", cal["early_precision"])

        # ------------------------------------------------------------------
        # Save JSON
        # ------------------------------------------------------------------
        output_data = {
            "samples": self.test_outputs,
            "stubbornness": stubbornness_metrics,
            "compositional_gap": comp_gap,
            "halting_distribution": halting_distribution,
            "q_head_calibration": q_head_calibration,
            "iterative_accuracy": iter_acc,
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
        The optimizer is initialized with the parameters of the model and the learning rate scheduler is initialized with the optimizer.
        
        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        Returns:
            optimizer_config (dict): A dictionary containing the optimizer and the learning rate scheduler to be used during training.
        
        """

        # Split parameters: embedding layer gets its own (higher) LR, everything else uses the base LR
        base_lr = self.cfg.model.lr
        emb_lr = self.cfg.model.get("embeddings_lr", base_lr)

        # Get input embedding layer parameters from the encoder
        emb_params = list(self.encoder.input_embedding.parameters())
        emb_param_ids = {id(p) for p in emb_params}

        # Get all other parameters that are not part of the input embedding layer
        other_params = [p for p in self.parameters() if id(p) not in emb_param_ids]

        # Define parameter groups with their respective learning rates
        # The "initial_lr" key is stored per group and used for learning rate warmup scaling
        param_groups = [
            {"params": other_params,  "lr": base_lr, "initial_lr": base_lr},
            {"params": emb_params,    "lr": emb_lr,  "initial_lr": emb_lr,  "name": "embeddings"},
        ]

        # Define the optimizer
        if self.cfg.model.optimizer == 'adam':
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.cfg.model.weight_decay)

        elif self.cfg.model.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))  # betas as per the TRM paper

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

        if self.cfg.training.get("use_manual_optimization", False):
            # With manual optimization, PTL ignores interval/frequency/monitor — scheduler is stepped manually
            # in on_train_epoch_end (epoch-based) or on_validation_epoch_end (ReduceLROnPlateau).
            lr_scheduler_config = {"scheduler": scheduler}
        else:
            lr_scheduler_config = {
                "scheduler": scheduler,
                "interval": self.cfg.model.lr_scheduler.interval,  # 'epoch' or 'step'
                "frequency": self.cfg.model.lr_scheduler.frequency,
                "monitor": self.cfg.model.lr_scheduler.monitored_metric,  # metric to track for lr scheduling. E.g., val_loss or val_acc
            }

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }

        return optimizer_config
