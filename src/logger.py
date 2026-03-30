import logging
import os
import sys
from datetime import datetime

# Logging directory
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Log file path
log_file = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(LOGS_DIR, log_file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("DeliveryDelayLogger")

def get_logger(module_name):
    return logging.getLogger(module_name)
