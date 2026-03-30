import pandas as pd
from src.config import LABEL_MAP, TARGET
from src.logger import get_logger

logger = get_logger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic data cleaning: remove www, handle nulls, encode target.
    """
    logger.info("Cleaning data...")
    
    # Remove 'www.' from URLs (matching notebook procedure)
    df['url'] = df['url'].replace('www.', '', regex=True)
    
    # Encode target variable
    df[TARGET] = df['type'].map(LABEL_MAP)
    
    # Handle missing values if any
    df = df.dropna()
    
    logger.info("Data cleaning complete.")
    return df

if __name__ == "__main__":
    from src.data_ingestion import load_data
    raw_df = load_data()
    clean_df = clean_data(raw_df)
    print(clean_df.head())
