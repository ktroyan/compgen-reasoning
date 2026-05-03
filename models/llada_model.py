"""
models.llada_model

Standalone PyTorch Lightning wrapper for LLaDA masked diffusion on COGITAO.
Training uses target-only stochastic masking; validation and test metrics use
iterative denoising generation.
"""

import json
import os
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from models.model_helpers import (
    _extract_evolution_samples,
    _plot_epoch_grids,
    _plot_metrics,
    build_network,
    compute_metrics as _compute_metrics_fn,
    compute_per_transform_id_metrics,
    log_per_transform_id_metrics,
    maybe_compile,
    stce_loss,
)
from utility.logging_utils import logger


class ModelModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

    def compute_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        return _compute_metrics_fn(preds, targets, pad_id=self.cfg.data.pad_token_id)

    def decode_sample(self, token_ids: torch.Tensor) -> str:
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "tokenizer"):
            return str(self.trainer.datamodule.tokenizer.decode(token_ids.detach().cpu().tolist()))
        return str(token_ids.detach().cpu().tolist())


class LLaDAModel(ModelModule):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        input_dim = cfg.model.get("d_model", 128)
        self.d_model = input_dim

        OmegaConf.set_struct(cfg, False)
        cfg.model.d_model = self.d_model

        base_input_vocab_size = int(cfg.model.input_vocab_size)
        sage_thinking = bool(cfg.network.encoder.diffusion.get("sage_thinking", False))
        cfg.model.mask_token_id = base_input_vocab_size
        cfg.model.thinking_token_id = base_input_vocab_size + 1 if sage_thinking else None
        cfg.model.input_vocab_size = base_input_vocab_size + 1 + int(sage_thinking)

        prompt_len = int(cfg.model.max_seq_len)
        target_len = int(cfg.model.max_h) * int(cfg.model.max_w)
        if cfg.model.get("predict_eos", False):
            target_len += 1
        cfg.model.llada_max_seq_len = prompt_len + target_len
        cfg.network.encoder.max_sequence_length = cfg.model.llada_max_seq_len
        cfg.network.encoder.mask_token_id = cfg.model.mask_token_id

        if cfg.model.get("predict_eos", False):
            cfg.model.output_vocab_size = max(int(cfg.model.output_vocab_size), int(cfg.model.eos_token_id) + 1)
        if sage_thinking:
            cfg.model.output_vocab_size = max(int(cfg.model.output_vocab_size), int(cfg.model.thinking_token_id) + 1)
        if cfg.model.output_dim != cfg.model.output_vocab_size:
            cfg.model.output_dim = cfg.model.output_vocab_size

        OmegaConf.set_struct(cfg, True)

        self.vocab_size = cfg.model.output_vocab_size
        self.target_len = target_len
        self.predict_eos = cfg.model.get("predict_eos", False)
        self.ignore_pad_in_diffusion_loss = cfg.model.get("ignore_pad_in_diffusion_loss", True)

        self.log_samples_flag = cfg.logging.get("log_samples_for_inspection", False)
        self.visualize_predictions = cfg.logging.get("visualize_model_predictions", False)
        self.evol_batch_idx = cfg.logging.get("evolution_batch_idx", 0)
        self.evol_indices = cfg.logging.get("evolution_sample_indices", [0, 1, 2])
        self.evolution_records = {"train": [], "val_id": [], "val_ood": []}
        self.epoch_metrics = []

        logger.info(
            "Initializing LLaDA masked diffusion model "
            f"with d_model={self.d_model}, prompt_len={prompt_len}, target_len={target_len}"
        )

        self.encoder = maybe_compile(build_network(cfg.model.encoder_network, cfg), cfg)

        pad_id = cfg.data.pad_token_id
        output_vocab_size = cfg.model.get("output_vocab_size")
        foreground_weight = cfg.model.get("foreground_weight", 1.0)
        if foreground_weight != 1.0:
            loss_weights = torch.ones(output_vocab_size)
            loss_weights[1:] = foreground_weight
            self.register_buffer("loss_weights", loss_weights)
        else:
            self.loss_weights = None

        self.pad_token_id = pad_id
        self.loss_func = cfg.model.get("loss_func", "cross_entropy")
        self.focal_gamma = cfg.model.get("focal_gamma", 2.0)
        self.epoch_samples = {}
        self.test_outputs = []

    def _target_from_batch(self, tgt: torch.Tensor) -> torch.Tensor:
        if self.predict_eos:
            return tgt
        return tgt[:, :-1]

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        prediction_mask: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        bsz = targets.shape[0] if targets.dim() == 2 else None
        seq_len = targets.shape[1] if targets.dim() == 2 else None
        logits_flat = logits.reshape(-1, logits.size(-1))
        targets_flat = targets.reshape(-1)

        weight = getattr(self, "loss_weights", None)
        if self.loss_func == "stce":
            ce = stce_loss(logits_flat, targets_flat, self.pad_token_id, weight=weight)
        else:
            ce = F.cross_entropy(
                logits_flat,
                targets_flat,
                weight=weight,
                ignore_index=self.pad_token_id,
                reduction="none",
            )

        if self.loss_func == "focal":
            pt = torch.exp(-ce)
            ce = (1 - pt).pow(self.focal_gamma) * ce

        valid = targets_flat != self.pad_token_id
        if prediction_mask is not None:
            valid = valid & prediction_mask.reshape(-1).bool()

        if reduction == "mean":
            return (ce * valid.float()).sum() / valid.float().sum().clamp(min=1)

        ce = ce.view(bsz, seq_len)
        valid = valid.view(bsz, seq_len).float()
        return (ce * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

    def _mask_target(self, target: torch.Tensor):
        target_masked, mask_target, target_for_loss = self.encoder.mask_input_sequence(target)
        if self.ignore_pad_in_diffusion_loss:
            mask_target = mask_target & (target_for_loss != self.pad_token_id)
        return target_masked, mask_target, target_for_loss

    def forward_sample(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        del y
        return self.encoder(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt_grid: Optional[torch.Tensor],
        task_tokens: Optional[torch.Tensor] = None,
        generate: bool = False,
    ) -> torch.Tensor:
        del task_tokens
        if generate:
            if tgt_grid is None:
                raise ValueError("tgt_grid is required for LLaDA generation length")
            generated = self.encoder.generate_masked_sequence(self.forward_sample, src, tgt_grid)
            return generated[:, src.size(1) :]
        if tgt_grid is None:
            return self.forward_sample(src)
        model_input = torch.cat([src, tgt_grid], dim=1)
        logits_full = self.forward_sample(model_input)
        return logits_full[:, src.size(1) :]

    def on_fit_start(self):
        try:
            from torchinfo import summary

            device = self.device
            src_dummy = torch.zeros(1, self.cfg.model.max_seq_len, dtype=torch.long, device=device)
            tgt_dummy = torch.zeros(1, self.target_len, dtype=torch.long, device=device)
            orig_encoder = self.encoder
            try:
                self.encoder = getattr(self.encoder, "_orig_mod", self.encoder)
                stats = summary(self, input_data=[src_dummy, tgt_dummy], verbose=0, depth=0)
            finally:
                self.encoder = orig_encoder
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

    def training_step(self, batch, batch_idx):
        src = batch["src"]
        target = self._target_from_batch(batch["tgt"])
        target_masked, mask_target, target_for_loss = self._mask_target(target)

        logits = self(src, target_masked, task_tokens=batch.get("task_tokens"))
        preds = torch.argmax(logits, dim=-1)
        loss = self._compute_loss(logits, target_for_loss, prediction_mask=mask_target, reduction="mean")
        metrics = self.compute_metrics(preds, target)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", metrics["acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc", metrics["grid_acc"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc_no_pad", metrics["acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/obj_acc", metrics["obj_acc"], on_step=True, on_epoch=True, prog_bar=True)

        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples["train"]["first"] = (src[0], target[0], preds[0])
            self.epoch_samples["train"]["last"] = (src[-1], target[-1], preds[-1])

        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, target, preds)
            if record:
                self.evolution_records["train"].append({"epoch": self.current_epoch, "samples": record})

        return loss

    def _teacher_forced_loss(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_masked, mask_target, target_for_loss = self._mask_target(target)
        logits = self(src, target_masked)
        return self._compute_loss(logits, target_for_loss, prediction_mask=mask_target, reduction="mean")

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        src = batch["src"]
        target = self._target_from_batch(batch["tgt"])
        prefix = "id" if dataloader_idx == 0 else "ood"

        loss = self._teacher_forced_loss(src, target)
        preds = self(src, target, task_tokens=batch.get("task_tokens"), generate=True)
        metrics = self.compute_metrics(preds, target)

        self.log(f"val/{prefix}_loss", loss, on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc", metrics["acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc", metrics["grid_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_acc_no_pad", metrics["acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_obj_acc", metrics["obj_acc"], on_step=False, on_epoch=True, add_dataloader_idx=False, prog_bar=True)
        self.log(f"val/{prefix}_gen_grid_acc_no_pad", metrics["grid_acc_no_pad"], on_step=False, on_epoch=True, add_dataloader_idx=False)

        key = f"val_{prefix}"
        if self.log_samples_flag and batch_idx == 0:
            self.epoch_samples[key]["first"] = (src[0], target[0], preds[0])
            self.epoch_samples[key]["last"] = (src[-1], target[-1], preds[-1])

        if batch_idx == self.evol_batch_idx:
            record = _extract_evolution_samples(self, src, target, preds)
            if record:
                self.evolution_records[key].append({"epoch": self.current_epoch, "samples": record})

        return loss

    def _record_epoch_metrics(self):
        if self.trainer.sanity_checking:
            return
        current_metrics = {
            k: v.item()
            for k, v in self.trainer.callback_metrics.items()
            if isinstance(v, torch.Tensor) and v.numel() == 1
        }
        current_metrics["epoch"] = self.current_epoch
        if len(self.epoch_metrics) > 0 and self.epoch_metrics[-1]["epoch"] == self.current_epoch:
            self.epoch_metrics[-1].update(current_metrics)
        else:
            self.epoch_metrics.append(current_metrics)

    def on_train_epoch_end(self):
        if "train" in self.epoch_samples:
            self._log_samples("Train", self.epoch_samples.get("train"))
        if self.evolution_records["train"] and self.evolution_records["train"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "train", self.current_epoch, self.evolution_records["train"][-1]["samples"])
        self._record_epoch_metrics()

    def on_validation_epoch_end(self):
        if "val_id" in self.epoch_samples:
            self._log_samples("Validation (ID)", self.epoch_samples.get("val_id"))
        if "val_ood" in self.epoch_samples and self.epoch_samples["val_ood"].get("last") is not None:
            self._log_samples("Validation (OOD)", self.epoch_samples.get("val_ood"))
        if self.evolution_records["val_id"] and self.evolution_records["val_id"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "val_id", self.current_epoch, self.evolution_records["val_id"][-1]["samples"])
        if self.evolution_records["val_ood"] and self.evolution_records["val_ood"][-1]["epoch"] == self.current_epoch:
            _plot_epoch_grids(self, "val_ood", self.current_epoch, self.evolution_records["val_ood"][-1]["samples"])
        self._record_epoch_metrics()

    def on_fit_end(self):
        cwd = os.getcwd()
        try:
            with open(os.path.join(cwd, "prediction_evolution.json"), "w") as f:
                json.dump(self.evolution_records, f, indent=2)
            logger.info(f"Saved prediction evolution to {os.path.join(cwd, 'prediction_evolution.json')}")
        except Exception as e:
            logger.error(f"Failed to save prediction evolution: {e}")
        try:
            with open(os.path.join(cwd, "metrics_history.json"), "w") as f:
                json.dump(self.epoch_metrics, f, indent=2)
            logger.info(f"Saved metrics history to {os.path.join(cwd, 'metrics_history.json')}")
        except Exception as e:
            logger.error(f"Failed to save metrics history: {e}")
        _plot_metrics(self, cwd)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        src = batch["src"]
        tgt = batch["tgt"]
        target = self._target_from_batch(tgt)
        transformation_suites = batch["transformation_suite"]
        prefix = "id" if dataloader_idx == 0 else "ood"

        preds = self(src, target, task_tokens=batch.get("task_tokens"), generate=True)
        target_masked, mask_target, target_for_loss = self._mask_target(target)
        logits = self(src, target_masked)
        ps_losses = self._compute_loss(logits, target_for_loss, prediction_mask=mask_target, reduction="none_per_sample")
        ps_metrics = _compute_metrics_fn(preds, target, pad_id=self.cfg.data.pad_token_id, per_sample=True)

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

        for i in range(src.size(0)):
            self.test_outputs.append(
                {
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
                }
            )

    def on_test_epoch_end(self):
        id_outputs = [o for o in self.test_outputs if o["domain_type"] == "id"]
        ood_outputs = [o for o in self.test_outputs if o["domain_type"] == "ood"]
        matched = id_outputs and ood_outputs and len(id_outputs) == len(ood_outputs)
        metric_keys = ["acc", "grid_acc", "acc_no_pad", "grid_acc_no_pad", "obj_acc"]

        stubbornness_metrics = {}
        if matched:
            ood_preds = torch.tensor([o["prediction_raw"] for o in ood_outputs], dtype=torch.long)
            id_targets = torch.tensor([o["target_raw"][:-1] for o in id_outputs], dtype=torch.long)
            stub = _compute_metrics_fn(ood_preds, id_targets, pad_id=self.cfg.data.pad_token_id)
            stubbornness_metrics = {f"stubbornness_{k}": v.item() for k, v in stub.items()}
            logger.info("Stubbornness: " + ", ".join(f"{k}={v:.4f}" for k, v in stubbornness_metrics.items()))
        elif id_outputs and ood_outputs:
            logger.warning(f"Stubbornness skipped: ID ({len(id_outputs)}) and OOD ({len(ood_outputs)}) counts mismatch.")

        comp_gap = {}
        if id_outputs and ood_outputs:
            id_avg = {k: sum(o[f"id_{k}"] for o in id_outputs) / len(id_outputs) for k in metric_keys}
            ood_avg = {k: sum(o[f"ood_{k}"] for o in ood_outputs) / len(ood_outputs) for k in metric_keys}
            comp_gap = {f"gap_{k}": id_avg[k] - ood_avg[k] for k in metric_keys}
            logger.info("Compositional gap: " + ", ".join(f"{k}={v:.4f}" for k, v in comp_gap.items()))

        per_transform_stub = {}
        if ood_outputs:
            ood_groups: Dict = {}
            for ood_o in ood_outputs:
                key = "|".join(ood_o.get("transformation_suite", []))
                ood_groups.setdefault(key, {"ood_preds": [], "ood_targets": []})
                ood_groups[key]["ood_preds"].append(ood_o["prediction_raw"])
                ood_groups[key]["ood_targets"].append(ood_o["target_raw"][:-1])
            id_groups: Dict = {}
            if matched:
                for id_o, ood_o in zip(id_outputs, ood_outputs):
                    key = "|".join(ood_o.get("transformation_suite", []))
                    id_groups.setdefault(key, {"id_targets": []})
                    id_groups[key]["id_targets"].append(id_o["target_raw"][:-1])
            for key, ood_g in ood_groups.items():
                ood_p = torch.tensor(ood_g["ood_preds"], dtype=torch.long)
                ood_t = torch.tensor(ood_g["ood_targets"], dtype=torch.long)
                ood_metrics = _compute_metrics_fn(ood_p, ood_t, pad_id=self.cfg.data.pad_token_id)
                entry = {f"ood_{k}": v.item() for k, v in ood_metrics.items()}
                if key in id_groups:
                    id_t = torch.tensor(id_groups[key]["id_targets"], dtype=torch.long)
                    stub = _compute_metrics_fn(ood_p, id_t, pad_id=self.cfg.data.pad_token_id)
                    entry.update({f"stubbornness_{k}": v.item() for k, v in stub.items()})
                per_transform_stub[key] = entry
            logger.info(f"Per-transformation scatter data computed for {len(per_transform_stub)} transformation type(s).")

        per_transform_id = compute_per_transform_id_metrics(id_outputs)
        per_transform_id_scalars = log_per_transform_id_metrics(per_transform_id, self.log)
        for k, v in stubbornness_metrics.items():
            self.log(f"test/{k}", v)
        for k, v in comp_gap.items():
            self.log(f"test/{k}", v)

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
        if not self.log_samples_flag or not samples_dict:
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
        base_lr = self.cfg.model.lr
        emb_lr = self.cfg.model.get("embeddings_lr", base_lr)
        encoder = getattr(self.encoder, "_orig_mod", self.encoder)
        emb_params = list(encoder.input_embedding.parameters())
        emb_param_ids = {id(p) for p in emb_params}
        other_params = [p for p in self.parameters() if id(p) not in emb_param_ids]
        param_groups = [
            {"params": other_params, "lr": base_lr, "initial_lr": base_lr},
            {"params": emb_params, "lr": emb_lr, "initial_lr": emb_lr, "name": "embeddings"},
        ]

        if self.cfg.model.optimizer == "adam":
            optimizer = torch.optim.Adam(param_groups, weight_decay=self.cfg.model.weight_decay)
        elif self.cfg.model.optimizer == "adamw":
            optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.model.weight_decay, betas=(0.9, 0.95))
        elif self.cfg.model.optimizer == "sgd":
            optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=self.cfg.model.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer given: {self.cfg.model.optimizer}")

        if self.cfg.model.lr_scheduler.type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        elif self.cfg.model.lr_scheduler.type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.training.max_epochs, eta_min=1e-6)
        elif self.cfg.model.lr_scheduler.type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError(f"Unknown scheduler given: {self.cfg.model.lr_scheduler.type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": self.cfg.model.lr_scheduler.interval,
                "frequency": self.cfg.model.lr_scheduler.frequency,
                "monitor": self.cfg.model.lr_scheduler.monitored_metric,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
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
