"""
utility/callbacks/epoch_summary_callback.py

Logs a clean, readable epoch summary at the end of each validation pass.

Especially useful in non-TTY environments (e.g. SLURM log files) where the
PTL progress bar is disabled and no visual epoch summary is printed.

"""

import pytorch_lightning as pl

from utility.logging_utils import logger


class EpochSummaryCallback(pl.Callback):

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        train_m = {k: v for k, v in metrics.items() if k.startswith("train/")}
        val_m   = {k: v for k, v in metrics.items() if k.startswith("val/")}

        sep = "-" * 62
        lines = [sep, f"  Epoch {trainer.current_epoch:>4d} Summary", sep]

        def _fmt(v):
            try:
                return f"{float(v):.5f}"
            except (TypeError, ValueError):
                return str(v)

        if train_m:
            lines.append("  TRAIN")
            for k in sorted(train_m):
                lines.append(f"    {k:<38}  {_fmt(train_m[k])}")

        if val_m:
            lines.append("  VAL")
            for k in sorted(val_m):
                lines.append(f"    {k:<38}  {_fmt(val_m[k])}")

        lines.append(sep)
        logger.info("\n" + "\n".join(lines))
