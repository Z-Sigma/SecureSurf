import argparse
import pandas as pd
import mlflow.pyfunc
import mlflow.artifacts
import os
import joblib
from src.feature_engineering import extract_features
from src.config import REVERSE_LABEL_MAP, MLFLOW_TRACKING_URI, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
from src.logger import get_logger

# Set environment variables for MLflow/S3 access
from dotenv import load_dotenv
load_dotenv()

os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

logger = get_logger(__name__)

def predict_url(url, run_id):
    """
    Predict the type of a given URL using a specific MLflow run.
    """
    logger.info(f"Predicting for URL: {url}")
    
    # 1. Prepare data
    df = pd.DataFrame([url], columns=['url'])
    
    # 2. Extract features
    try:
        df_features = extract_features(df)
        # Drop the original 'url' column as it's not a feature
        X = df_features.drop(columns=['url'])
    except Exception as e:
        logger.error(f"Error during feature extraction: {e}")
        return None

    # 3. Load model from MLflow
    logger.info(f"Connecting to MLflow Tracking Server at: {MLFLOW_TRACKING_URI}")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    
    try:
        # Automatically find the first artifact that looks like a model
        artifacts = client.list_artifacts(run_id)
        logger.info(f"Discovered artifacts in run {run_id}: {[a.path for a in artifacts]}")
        model_name = next((a.path for a in artifacts if "model" in a.path.lower()), None)
        
        if not model_name:
            logger.error(f"No model artifacts found in run {run_id}. Found: {[a.path for a in artifacts]}")
            return None
            
        # Download and load the joblib file
        logger.info(f"Downloading model artifact: {model_name}...")
        local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=model_name)
        model = joblib.load(local_path)
        logger.info(f"Model loaded successfully from {local_path}")
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("Tip: Ensure the MLflow server is running and the Run ID is correct.")
        return None

    # 4. Make prediction
    try:
        prediction = model.predict(X)
        label_idx = int(prediction[0])
        label = REVERSE_LABEL_MAP.get(label_idx, "Unknown")
        return label
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecureSurf URL Prediction Tool")
    parser.add_argument("--url", type=str, required=True, help="The URL to analyze")
    parser.add_argument("--run_id", type=str, required=True, help="The MLflow Run ID to use for inference")
    
    args = parser.parse_args()
    
    result = predict_url(args.url, args.run_id)
    
    if result:
        print("\n" + "="*30)
        print(f"URL: {args.url}")
        print(f"Prediction: {result.upper()}")
        print("="*30 + "\n")
    else:
        print("\nPrediction failed. Check logs for details.\n")
