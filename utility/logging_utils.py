"""
/utility/logging_utils.py

This module sets up the logging infrastructure for the project using Loguru and defines custom PyTorch Lightning callbacks for logging gradient norms and parameter counts.
"""

import os
import sys
import torch
import pytorch_lightning as pl
from datetime import datetime
import yaml
import logging
from loguru import logger as loguru_logger

# -----------------------
# Main local logger
# -----------------------

class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level
        try:
            level = loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        loguru_logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())


def setup_loguru(loguru_logger):
    # Generate a timestamp for the log file name
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%m-%d-%Hh%M")

    # Make sure the log folder exists
    log_folder = "experiments/logs"
    os.makedirs(log_folder, exist_ok=True)

    # Log file path
    log_filename = (f"{log_folder}/experiment_{formatted_datetime}.log")

    # Detect whether we are running interactively.
    # isatty() is not reliable under SLURM (srun can allocate a pseudo-terminal),
    # so we also check for the SLURM_JOB_ID environment variable which is always
    # set for batch jobs.
    is_slurm = "SLURM_JOB_ID" in os.environ
    is_tty = sys.stdout.isatty() and not is_slurm

    # Plain format for file sinks and non-TTY stdout (SLURM logs): single line, no ANSI codes
    fmt_plain = (
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{file}:{function}:{line} | "
        "{message}"
    )

    # Rich format for interactive terminals: colors + two-line layout
    fmt_rich = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</> | "
        "<lvl>{level: <8}</> | "
        "<cyan>{file}:{function}:{line}</>\n"
        "<lvl>{message}</>"
    )

    # Remove the default logger with its associated sink
    loguru_logger.remove()

    # File sink: always plain, no colors (files do not render ANSI)
    loguru_logger.add(
        log_filename,
        level="TRACE",
        format=fmt_plain,
        colorize=False,
        enqueue=True,
    )

    # Stdout sink: rich + colors when interactive, plain otherwise (SLURM / piped)
    loguru_logger.add(
        sys.stdout,
        level="INFO",
        format=fmt_rich if is_tty else fmt_plain,
        colorize=is_tty,
        enqueue=True,
    )

    # Redirect standard logging (coming from libraries used in the codebase) to Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)

    return loguru_logger


# Configure the logger and make the object accessible to other modules
logger = setup_loguru(loguru_logger)

