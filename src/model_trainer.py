import pandas as pd
from src.logger import get_logger
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
import joblib
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
            
            # Log model (Robust method: joblib + log_artifact)
            model_filename = f"{model_name.lower()}_model.joblib"
            joblib.dump(model, model_filename)
            mlflow.log_artifact(model_filename)
            os.remove(model_filename)
            logger.info(f"Successfully logged model artifact: {model_filename}")
            
            # Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                logger.info(f"Logging feature importance for {model_name}...")
                importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
                # Log top 5 as tags for quick view
                for feature, val in importances.head(5).items():
                    mlflow.set_tag(f"top_feature_{feature}", f"{val:.4f}")
                
                # Log full list as a CSV artifact
                importance_df = importances.reset_index()
                importance_df.columns = ['feature', 'importance']
                importance_df.to_csv("feature_importance.csv", index=False)
                mlflow.log_artifact("feature_importance.csv")
                os.remove("feature_importance.csv")
            
            results.append({"model": model_name, "accuracy": metrics["accuracy"]})

    logger.info("All models trained and logged.")
    return results

if __name__ == "__main__":
    pass
