import os
import sys
import logging
from datetime import datetime

# Define log format
LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

# Ensure log directory exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Define log file path with timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH)
    ]
)

def get_logger(name):
    return logging.getLogger(name)
