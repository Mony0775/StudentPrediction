# src/utils/logger.py
import logging
import sys
from datetime import datetime

def setup_logger(name='student_performance', log_level=logging.INFO):
    """Set up logger with console and file handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_filename = f'logs/app_{datetime.now().strftime("%Y%m%d")}.log'
    try:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    except Exception:
        pass
    
    return logger

logger = setup_logger()