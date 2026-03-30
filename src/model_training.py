import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def setup_mlflow():
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_xgboost(X_train, X_test, y_train, y_test, params=None):
    """Train XGBoost model and log to MLflow."""
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 100
        }
    
    with mlflow.start_run(run_name="XGBoost_Baseline"):
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.xgboost.log_model(model, "model")
        logger.info(f"XGBoost model training complete. Metrics logged to MLflow.")
        return model, rmse

def train_lightgbm(X_train, X_test, y_train, y_test, params=None):
    """Train LightGBM model and log to MLflow."""
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.1,
            'num_leaves': 31,
            'n_estimators': 100
        }
    
    with mlflow.start_run(run_name="LightGBM_Baseline"):
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.lightgbm.log_model(model, "model")
        logger.info("LightGBM model training complete. Metrics logged to MLflow.")
        return model, rmse

def train_random_forest(X_train, X_test, y_train, y_test, params=None):
    """Train Random Forest model and log to MLflow."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        }
    
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.sklearn.log_model(model, "model")
        print(f"Random Forest model trained. RMSE: {rmse:.4f}")
        return model, rmse

def train_catboost(X_train, X_test, y_train, y_test, params=None):
    """Train CatBoost model and log to MLflow."""
    if params is None:
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'verbose': False
        }
    
    with mlflow.start_run(run_name="CatBoost_Baseline"):
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        rmse, mae, r2 = eval_metrics(y_test, predictions)
        
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        mlflow.catboost.log_model(model, "model")
        print(f"CatBoost model trained. RMSE: {rmse:.4f}")
        return model, rmse

def run_optuna_tuning(X_train, X_val, y_train, y_val, n_trials=10):
    """Run Optuna hyperparameter tuning for LightGBM."""
    def objective(trial):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        
    return study.best_params
