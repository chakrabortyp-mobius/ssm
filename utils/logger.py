"""
utils/logger.py
===============
Lightweight structured logger used across all modules.
No external dependencies — stdlib only.

Usage
-----
    from utils.logger import get_logger
    log = get_logger("DataLoader")
    log.info("Loaded 5000 rows")
    log.warning("Column 'url' dropped — too sparse")
    log.debug("X.shape = (5000, 32)")   # only shown if level=DEBUG
"""

import logging
import sys


_FMT = "[%(levelname)s] %(name)s | %(message)s"


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Return a named logger with consistent formatting.

    Parameters
    ----------
    name  : module/component name shown in log prefix
    level : 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR'
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger
