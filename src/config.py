import os
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "malicious_phish.csv"
RAW_DATA_PATH = DATA_DIR / "malicious_phish.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# MLflow Config
MLFLOW_TRACKING_URI = "http://localhost:5000"

# Feature lists
FEATURES = [
    'url_len', 'letters_count', 'digits_count', 'special_chars_count', 
    'shortened', 'abnormal_url', 'secure_http', 'have_ip',
    'count_dot', 'count_at', 'count_dir', 'count_embed', 
    'count_percent', 'count_equal', 'count_hyphen', 'count_tld'
]

TARGET = 'url_type'

# Label Mapping
LABEL_MAP = {
    'benign': 0,
    'defacement': 1,
    'phishing': 2,
    'malware': 3
}

# Model Configuration
MODELS_CONFIG = [
    {"name": "DecisionTree", "class": DecisionTreeClassifier, "params": {"random_state": 42}},
    {"name": "RandomForest", "class": RandomForestClassifier, "params": {"n_estimators": 100, "random_state": 42}},
    {"name": "AdaBoost", "class": AdaBoostClassifier, "params": {"random_state": 42}},
    {"name": "KNeighbors", "class": KNeighborsClassifier, "params": {}},
    {"name": "ExtraTrees", "class": ExtraTreesClassifier, "params": {"random_state": 42}},
    {"name": "GaussianNB", "class": GaussianNB, "params": {}}
]
