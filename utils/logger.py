import logging
import os
from datetime import datetime
from config.settings import config

def setup_logger(name, log_path=None):
    """Configure and return a logger instance"""
    if log_path is None:
        log_path = config.LOG_PATH
        
    os.makedirs(log_path, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_path, f"{name}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger