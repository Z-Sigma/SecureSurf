import pandas as pd
import logging
from src.config import RAW_DATA_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path=RAW_DATA_PATH):
    """
    Load the dataset from a CSV file.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e

if __name__ == "__main__":
    data = load_data()
    print(data.head())
