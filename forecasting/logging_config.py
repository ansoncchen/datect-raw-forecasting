"""
Logging Configuration Module
============================

Provides centralized logging configuration for the DATect forecasting system.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level='INFO', enable_file_logging=False, log_dir='./logs/'):
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    if enable_file_logging:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_path / f'datect_{timestamp}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)
        logger.info(f"File logging enabled: {log_file}")


def get_logger(name):
    return logging.getLogger(name)
