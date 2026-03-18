"""
models.transformer_model

TODO:
1) Create a ModelModule class that inherits from pl.LightningModule and a TransformerModel class that inherits from ModelModule in order to have a model module that
   can be used with PTL Trainer in training.py and inference.py.

2) The TransformerModel should consist of an encoder and decoder, which are implemented as separate classes
   in the /networks folder (e.g., transformer_encoder.py and transformer_decoder.py) and which together form the full Transformer model.
   That is, input data from the dataloader are fed into the encoder, and the output of the encoder is fed into
   the decoder to produce the final predictions for the problem (e.g., if the data module is "GridDataModule" then it is for grid prediction).

3) The forward method of TransformerModel should define the forward pass through the encoder and decoder.

4) The training_step, validation_step, and test_step methods should compute the relevant losses and metrics and log them.

5) The model should also include any necessary methods for visualization of predictions and logging to WandB, as well as any helper methods needed for decoding predictions or calculating metrics.

"""

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Dict
import json

from utility.logging_utils import logger
from networks.transformer_encoder import TransformerEncoder
from networks.mlp_decoder import MLPDecoder

from models.model_helpers import _extract_evolution_samples, _plot_epoch_grids, _plot_metrics


class ModelModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
    
    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # Element-wise correctness
        correct = (preds == targets)
        
        # Token accuracy (mean over all tokens in batch)
        acc = correct.float().mean()
        
        # Grid accuracy (1.0 only if ALL tokens in a sequence are correct)
        grid_correct = correct.all(dim=1)   # check along sequence dimension (dim=1)
        grid_acc = grid_correct.float().mean()

        # --- Without padding tokens considered ---
        pad_id = self.cfg.data.pad_token_id
        mask = targets != pad_id
        correct = (preds == targets) & mask

        # Token accuracy without padding tokens
        acc_no_pad = correct.sum().float() / mask.sum().float()

        # Grid accuracy without padding tokens: a sequence is correct if all non-pad tokens are correct
        grid_correct = ((preds == targets) | ~mask).all(dim=1)
        grid_acc_no_pad = grid_correct.float().mean()
        
        return {"acc": acc, "grid_acc": grid_acc, "acc_no_pad": acc_no_pad, "grid_acc_no_pad": grid_acc_no_pad}

    def decode_sample(self, token_ids: torch.Tensor) -> str:
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "tokenizer"):
            ids = token_ids.detach().cpu().tolist()
            return str(self.trainer.datamodule.tokenizer.decode(ids))
        return str(token_ids.detach().cpu().tolist())

class TransformerModel(ModelModule):
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
        self.evolution_records = {"train": [], "val_id": [], "val_ood":[]}
        
        # Metric Tracking
        self.epoch_metrics =[]

        logger.info(f"Initializing Transformer (Encoder + MLP Decoder) with d_model={self.d_model}")
        self.encoder = TransformerEncoder(cfg)
        self.decoder = MLPDecoder(cfg)

        # --- Loss Function ---

        # Weights for class token imbalance
        num_classes = self.vocab_size
        loss_weights = torch.ones(num_classes)
        loss_weights[1:10] = 5.0 
        
        self.register_buffer("loss_weights", loss_weights)  # register to handle device transfer

        # TODO: should we ignore padding tokens? should we balance tokens?
        self.loss_fn = nn.CrossEntropyLoss()
        # self.loss_fn = nn.CrossEntropyLoss(weight=self.loss_weights, ignore_index=self.cfg.data.pad_token_id)
        # self.loss_fn = nn.CrossEntropyLoss(weight=self.loss_weights, ignore_index=0)
        # self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.cfg.data.pad_token_id)

        # --- Logging ---
        self.epoch_samples = {}
        self.test_outputs = []

    def forward_features_for_MLP_decoder(self, x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Output relevant feature embeddings of the encoded sequence x
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
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None, task_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Encoder -> Decoder -> Logits
        """

        encoder_output = self.encoder(src)

        # TODO: we can do something with task_tokens to build a task-aware model for OOD generalization
        # Note that the task_tokens are already included in the src sequence fed to the encoder if the config parameter use_task_tokens is set to true
        # Then, src is [BOS, task_tokens..., grid_tokens..., EOS]

        # --- MLP Decoder ---
        if self.cfg.network.decoder.get("name", None) in ["mlp_decoder"]:
            encoder_output = self.forward_features_for_MLP_decoder(encoder_output, tgt)
            logits = self.decoder(encoder_output)   # logits should be of shape [B, S_grid, output_dim] where S_grid is the number of grid tokens in the target sequence

        # --- AR Decoder ---
        elif self.cfg.network.decoder.get("name", None) == "ar_decoder":
            # No special handling needed for AR decoder (it happens in the AR decoder directly since the encoded sequence is used as memory)
            logits = self.decoder(encoder_output, tgt)  # logits should be of shape [B, S_grid, output_dim] where S_grid is the number of grid tokens in the target sequence
            
        else:
            raise ValueError(f"Unknown decoder name: {self.cfg.network.decoder.get('name', None)}")


        return logits   # [B, S_grid, output_dim=output_vocab_size]
            

    # ------------------------------------------------------------------
    # Lifecycle Hooks
    # ------------------------------------------------------------------
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
        src, tgt, task_tokens = batch
        
        logits = self(src, tgt, task_tokens=task_tokens)

        # Get predictions from logits: [B, S_grid] <-- [B, S_grid, output_vocab_size]
        preds = torch.argmax(logits, dim=-1)    # [B, S_grid]; get predicted discrete tokens

        # Remove the <EOS> token from the target sequence for loss computation since we do not want to predict anything for that position in the target sequence (we will ignore it when computing the loss)
        # NOTE: the <EOS> is useful and included in the target sequence for the decoder's forward pass since it allows an AR decoder to know where the end of the grid tokens is in the target sequence
        # For AR decoder, we should consider the <EOS> token when computing the loss
        if self.cfg.network.decoder.get("name", None) in ["mlp_decoder"]:
            # Remove EOS from target
            tgt = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        # Compute loss
        # NOTE: CrossEntropyLoss expects unnormalized logits since it computes the log-softmax and use the NLLLoss.
        #       So input of shape [B, C, S] where C is the size of the logits (i.e., number of classes (output vocab size)) and target of shape [B, S]
        # loss = self.loss_fn(logits.transpose(1, 2), tgt)
        loss = self.loss_fn(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))  # alternative equivalent reshaping for CrossEntropyLoss

        # Compute metrics
        metrics = self.compute_metrics(preds, tgt)

        # Log metrics (on_step=True allows seeing fluctuations during epoch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", metrics["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc", metrics["grid_acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_no_pad", metrics["acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)

        if "train" not in self.epoch_samples:
            self.epoch_samples["train"] = {"first": None, "last": None}
        
        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples["train"]["first"] = (src[0], tgt[0], preds[0])
            self.epoch_samples["train"]["last"] = (src[-1], tgt[-1], preds[-1])
        # Track sample prediction evolution
        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, tgt, preds)
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
        src, tgt, task_tokens = batch
        
        logits = self(src, tgt, task_tokens=task_tokens)

        preds = torch.argmax(logits, dim=-1)

        # Remove the <EOS> token from the target sequence for loss computation since we do not want to predict anything for that position in the target sequence (we will ignore it when computing the loss)
        # NOTE: the <EOS> is useful and included in the target sequence for the decoder's forward pass since it allows an AR decoder to know where the end of the grid tokens is in the target sequence
        # For AR decoder, we should consider the <EOS> token when computing the loss
        if self.cfg.network.decoder.get("name", None) in ["mlp_decoder"]:
            # Remove EOS from target
            tgt = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        loss = self.loss_fn(logits.transpose(1, 2), tgt)
        
        metrics = self.compute_metrics(preds, tgt)
        
        # Determine prefix based on dataloader index
        # 0 -> ID, 1 -> OOD (assuming DataModule returns [id_loader, ood_loader])
        prefix = "id" if dataloader_idx == 0 else "ood"
        
        # Log with prog_bar=True so they show up in console
        self.log(f"val/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)

        key = f"val_{prefix}"
        if key not in self.epoch_samples:
            self.epoch_samples[key] = {"first": None, "last": None}

        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples[key]["first"] = (src[0], tgt[0], preds[0])
            self.epoch_samples[key]["last"] = (src[-1], tgt[-1], preds[-1])

        # Track Evolution
        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, tgt, preds)
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
    # End of Training Artifact Handling
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
        src, tgt, task_tokens = batch

        logits = self(src, tgt, task_tokens=task_tokens)

        preds = torch.argmax(logits, dim=-1)

        # Remove the <EOS> token from the target sequence for loss computation since we do not want to predict anything for that position in the target sequence (we will ignore it when computing the loss)
        # NOTE: the <EOS> is useful and included in the target sequence for the decoder's forward pass since it allows an AR decoder to know where the end of the grid tokens is in the target sequence
        # For AR decoder, we should consider the <EOS> token when computing the loss
        if self.cfg.network.decoder.get("name", None) in ["mlp_decoder"]:
            # Remove EOS from target
            tgt = tgt[:, :-1]  # [B, S_grid] <-- [B, S_grid + 1]

        loss = self.loss_fn(logits.transpose(1, 2), tgt)

        metrics = self.compute_metrics(preds, tgt)
        
        prefix = "id" if dataloader_idx == 0 else "ood"
        
        self.log(f"test/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"test/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)

        preds = torch.argmax(logits, dim=-1)
        
        src_cpu = src.detach().cpu()
        tgt_cpu = tgt.detach().cpu()
        preds_cpu = preds.detach().cpu()
        
        for i in range(len(src)):
            self.test_outputs.append({
                "domain_type": prefix,
                "input_raw": src_cpu[i].tolist(),
                "target_raw": tgt_cpu[i].tolist(),
                "prediction_raw": preds_cpu[i].tolist(),
                "input_decoded": self.decode_sample(src_cpu[i]),
                "target_decoded": self.decode_sample(tgt_cpu[i]),
                "prediction_decoded": self.decode_sample(preds_cpu[i]),
                f"{prefix}_loss": loss.item(),
                f"{prefix}_acc": metrics["acc"].item(),
                f"{prefix}_grid_acc": metrics["grid_acc"].item(),
                f"{prefix}_acc_no_pad": metrics["acc_no_pad"].item(),
                f"{prefix}_grid_acc_no_pad": metrics["grid_acc_no_pad"].item()
            })

    def on_test_epoch_end(self):
        output_file = os.path.join(os.getcwd(), "test_predictions.json")

        try:
            with open(output_file, "w") as f:
                json.dump(self.test_outputs, f, indent=2)
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
    

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Override the PyTorch Lightning optimizer_step method to add custom logic before the optimizer.step() call.
        
        NOTE: We overwrite it for learning rate warm-up.
        """

        if self.cfg.model.get("lr_warmup", {}).get("enabled", False):
            if self.cfg.model.lr_warmup.type == "linear":
                # Linear LR warm up
                num_lr_warmup_steps = self.cfg.model.lr_warmup.num_steps
                if self.trainer.global_step < num_lr_warmup_steps:
                    lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_lr_warmup_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_scale * self.cfg.model.lr
            else:
                raise ValueError(f"Unknown LR warmup type given: {self.cfg.model.lr_warmup.type}")

        # This is the content of the original optimizer_step method from PyTorch Lightning
        optimizer.step(closure=optimizer_closure)   # update params

    def configure_optimizers(self):
        """ 
        Initializes the optimizer and the learning rate scheduler. 
        The optimizer is initialized with the parameters of the model and the learning rate scheduler is initialized with the optimizer.
        
        See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers

        Returns:
            optimizer_config (dict): A dictionary containing the optimizer and the learning rate scheduler to be used during training.
        """

        # Define the optimizer
        if self.cfg.model.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.model.lr, weight_decay=self.cfg.model.weight_decay)
        
        elif self.cfg.model.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.model.lr, weight_decay=self.cfg.model.weight_decay)
        
        elif self.cfg.model.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.model.lr, momentum=0.9, weight_decay=self.cfg.model.weight_decay)
        
        else:
            raise ValueError(f"Unknown optimizer given: {self.cfg.model.optimizer}")

        # Define the learning rate scheduler
        if self.cfg.model.lr_scheduler.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        elif self.cfg.model.lr_scheduler.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        elif self.cfg.model.lr_scheduler.type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        else:
            raise ValueError(f"Unknown scheduler given: {self.cfg.model.lr_scheduler.type}")

        optimizer_config = {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.cfg.model.lr_scheduler.interval,  # 'epoch' or 'step'
                "frequency": self.cfg.model.lr_scheduler.frequency,  # 'epoch' or 'step'; how often to call the scheduler w.r.t. the interval
                "monitor": self.cfg.model.lr_scheduler.monitored_metric,  # metric to track for lr scheduling. E.g., val_loss or val_acc
            },
        }

        return optimizer_config



    # def configure_optimizers(self):
    #     lr = self.cfg.model.get("lr", 1e-4)
    #     weight_decay = self.cfg.model.get("weight_decay", 1e-5)
        
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
    #     total_steps = self.trainer.estimated_stepping_batches
    #     warmup_steps = 200 
        
    #     pct_start = 0.1
    #     if total_steps > 0:
    #         pct_start = min(0.5, warmup_steps / total_steps)

    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #         optimizer, 
    #         max_lr=lr, 
    #         total_steps=total_steps,
    #         pct_start=pct_start, 
    #         anneal_strategy='cos'
    #     )
        
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "step"
    #         }
    #     }