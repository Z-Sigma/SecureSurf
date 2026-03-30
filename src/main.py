import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from src import data_ingestion, preprocessing, feature_engineering, model_training, config
from src.logger import get_logger

logger = get_logger(__name__)

def main():
    # 1. Setup
    logger.info("Setting up MLflow tracking...")
    model_training.setup_mlflow()
    
    # 2. Data Ingestion
    logger.info("Starting data ingestion...")
    data_dict = data_ingestion.load_all_data()
    
    # 3. Preprocessing
    logger.info("Preprocessing product and geolocation data...")
    df = preprocessing.clean_products(data_dict['products'])
    data_dict['products'] = df
    
    merged_df = preprocessing.merge_datasets(data_dict)
    
    # 4. Feature Engineering
    logger.info("Engineering features and distance calculations (geopy)...")
    merged_df = feature_engineering.calculate_distances(merged_df)
    merged_df = feature_engineering.extract_product_features(merged_df)
    
    # Aggregate to order level
    logger.info("Aggregating features to order level...")
    order_df = feature_engineering.aggregate_by_order(merged_df)
    
    # Time features and target variable
    logger.info("Calculating time features and target 'delivery_delay'...")
    final_df = feature_engineering.process_time_features(order_df, data_dict['orders'])
    
    # One-hot encoding of categories
    logger.info("One-hot encoding category features...")
    all_cats = data_dict['products']['product_category_name'].dropna().unique().tolist()
    final_df = feature_engineering.encode_categories(final_df, all_cats)
    
    # 5. Model Training
    logger.info("Preparing training sets (80/20 split)...")
    drop_cols = ['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']
    X = final_df.drop(drop_cols + ['delivery_delay'], axis=1)
    y = final_df['delivery_delay']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {}
    
    logger.info("Benchmarking XGBoost...")
    _, rmse_xgb = model_training.train_xgboost(X_train, X_test, y_train, y_test)
    results['XGBoost'] = rmse_xgb
    
    logger.info("Benchmarking LightGBM...")
    _, rmse_lgbm = model_training.train_lightgbm(X_train, X_test, y_train, y_test)
    results['LightGBM'] = rmse_lgbm

    logger.info("Benchmarking Random Forest...")
    _, rmse_rf = model_training.train_random_forest(X_train, X_test, y_train, y_test)
    results['RandomForest'] = rmse_rf

    logger.info("Benchmarking CatBoost...")
    _, rmse_cb = model_training.train_catboost(X_train, X_test, y_train, y_test)
    results['CatBoost'] = rmse_cb
    
    # Find best model
    best_model_name = min(results, key=results.get)
    logger.info(f"🏆 BEST MODEL IDENTIFIED: {best_model_name} with RMSE: {results[best_model_name]:.4f}")
    
    logger.info("✅ Full MLOps Pipeline completed successfully!")

if __name__ == "__main__":
    main()
