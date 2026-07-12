"""Centralized logging setup."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "oran_fed_robust", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
