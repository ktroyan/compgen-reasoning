"""
callbacks.ema_callback

Exponential Moving Average (EMA) of model weights, as used in the TRM paper.

After each optimizer step, shadow weights are updated:
    ema_param = decay * ema_param + (1 - decay) * current_param

Validation and test steps run with the EMA weights; training uses the live weights.

"""

import pytorch_lightning as pl
import torch


class EMACallback(pl.Callback):
    """
    Maintains an EMA copy of all trainable parameters.
    - Training: live weights are updated by the optimizer as normal.
    - Validation / Test: weights are swapped to EMA before the epoch and
      restored to live weights afterwards, so checkpointing and logging
      still reflect the EMA model's performance.

    Args:
        decay: EMA decay factor (e.g. 0.999). Higher = slower-moving average.
        cpu_offload: Whether to store shadow weights on CPU to avoid increasing GPU memory and thus chance of OOM errors.
    """

    def __init__(self, decay: float = 0.999, cpu_offload: bool = True):
        self.decay = decay
        self.cpu_offload = cpu_offload
        self._shadow: dict[str, torch.Tensor] = {}
        self._backup: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Initialise shadow weights at the start of training
    # ------------------------------------------------------------------
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                shadow = param.data.clone().detach()
                self._shadow[name] = shadow.cpu() if self.cpu_offload else shadow

    # ------------------------------------------------------------------
    # Update shadow weights after every optimizer step
    # ------------------------------------------------------------------
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self._shadow:
                live = param.data.cpu() if self.cpu_offload else param.data
                self._shadow[name].mul_(self.decay).add_(live, alpha=1.0 - self.decay)

    # ------------------------------------------------------------------
    # Swap to EMA weights before validation / test
    # ------------------------------------------------------------------
    def _apply_ema(self, pl_module: pl.LightningModule):
        for name, param in pl_module.named_parameters():
            if param.requires_grad and name in self._shadow:
                # Back up live weights to CPU (free GPU slot)
                self._backup[name] = param.data.cpu() if self.cpu_offload else param.data.clone()
                # Load EMA weights onto GPU
                param.data.copy_(self._shadow[name].to(param.device))

    def _restore(self, pl_module: pl.LightningModule):
        for name, param in pl_module.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name].to(param.device))
        self._backup.clear()

    def on_validation_epoch_start(self, trainer, pl_module):
        if self._shadow:  # skip sanity check before training begins
            self._apply_ema(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._backup:
            self._restore(pl_module)

    def on_test_epoch_start(self, trainer, pl_module):
        if self._shadow:
            self._apply_ema(pl_module)

    def on_test_epoch_end(self, trainer, pl_module):
        if self._backup:
            self._restore(pl_module)
