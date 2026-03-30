import mlflow
from src.data_ingestion import load_data
from src.preprocessing import clean_data
from src.feature_engineering import extract_features
from src.model_trainer import train_and_log_models
from src.config import MLFLOW_TRACKING_URI
from src.logger import get_logger

# Enforce tracking URI at the start
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

logger = get_logger(__name__)

def run_pipeline():
    """
    Run the complete MLOps pipeline for all models.
    """
    try:
        # Ensure tracking URI is set
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info("Initializing MLflow tracking...")
        logger.info("Starting Malicious URL Detection Pipeline")
        
        # 1. Ingestion
        df = load_data()
        
        # 2. Preprocessing
        df = clean_data(df)
        
        # 3. Feature Engineering
        # (Using a sample for testing if needed, or full data)
        # For full dataset (600k+ rows), feature engineering might take some time.
        # Let's use a sample of 100,000 to keep it manageable during demonstration.
        df_sample = df.sample(min(100000, len(df)), random_state=42)
        df_feats = extract_features(df_sample)
        
        # 4. Model Training and Logging
        results = train_and_log_models(df_feats)
        
        logger.info(f"Pipeline finished. Trained {len(results)} models.")
        
        for res in results:
            logger.info(f"Model: {res['model']}, Accuracy: {res['accuracy']}")
            
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise e

if __name__ == "__main__":
    run_pipeline()
