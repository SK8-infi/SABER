import logging
import os
import sys
from typing import Optional

def setup_logger(name: str = "saber", log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger that outputs to both stdout and a file.
    
    Args:
        name: Name of the logger.
        log_dir: Directory where logs should be stored.
        
    Returns:
        A configured logging.Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers if the logger is configured multiple times
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
