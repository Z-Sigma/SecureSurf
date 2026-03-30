import pandas as pd
from src.logger import get_logger
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from src.config import FEATURES, TARGET, MLFLOW_TRACKING_URI, MODELS_CONFIG, S3_ENDPOINT_URL, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

# Set environment variables for MLflow/S3
os.environ["MLFLOW_S3_ENDPOINT_URL"] = S3_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = AWS_ACCESS_KEY_ID
os.environ["AWS_SECRET_ACCESS_KEY"] = AWS_SECRET_ACCESS_KEY
os.environ["AWS_DEFAULT_REGION"] = AWS_DEFAULT_REGION

logger = get_logger(__name__)

def train_and_log_models(df: pd.DataFrame):
    """
    Train and log multiple models as separate MLflow experiments.
    """
    logger.info("Initializing MLflow tracking...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Split data once for consistency
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []

    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg["name"]
        model_class = model_cfg["class"]
        model_params = model_cfg["params"]

        logger.info(f"Setting up MLflow experiment for model: {model_name}")
        mlflow.set_experiment(f"Malicious_URL_{model_name}")

        with mlflow.start_run(run_name=f"{model_name}_Default"):
            logger.info(f"Training {model_name}...")
            
            # Initialize and fit
            model = model_class(**model_params)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }

            logger.info(f"{model_name} Metrics: {metrics}")

            # Log params and metrics
            mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, f"{model_name.lower()}_model")
            
            results.append({"model": model_name, "accuracy": metrics["accuracy"]})

    logger.info("All models trained and logged.")
    return results

if __name__ == "__main__":
    pass
