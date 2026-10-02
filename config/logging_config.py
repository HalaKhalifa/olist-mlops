"""Structured logging configuration for console and file logging."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from config.settings import settings


def setup_logger(
    name: str = "olist_mlops",
    log_file: Optional[Path] = None,
    level: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a logger configured with console and rotating file handlers."""
    log_level_str = level or settings.logging.level
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt=settings.logging.format,
        datefmt=settings.logging.date_format,
    )

    # 1. Console Handler (stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. File Handler (Rotating)
    target_log_file = log_file or settings.logging.log_file
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(target_log_file),
        maxBytes=settings.logging.max_bytes,
        backupCount=settings.logging.backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# Default application logger
logger = setup_logger()
