import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "delivery_data")

# Data File Paths
CUSTOMERS_DATA = os.path.join(DATA_DIR, "olist_customers_dataset.csv")
GEOLOCATION_DATA = os.path.join(DATA_DIR, "olist_geolocation_dataset.csv")
ORDER_ITEMS_DATA = os.path.join(DATA_DIR, "olist_order_items_dataset.csv")
ORDER_PAYMENTS_DATA = os.path.join(DATA_DIR, "olist_order_payments_dataset.csv")
ORDER_REVIEWS_DATA = os.path.join(DATA_DIR, "olist_order_reviews_dataset.csv")
ORDERS_DATA = os.path.join(DATA_DIR, "olist_orders_dataset.csv")
PRODUCTS_DATA = os.path.join(DATA_DIR, "olist_products_dataset.csv")
SELLERS_DATA = os.path.join(DATA_DIR, "olist_sellers_dataset.csv")
CATEGORY_TRANSLATION_DATA = os.path.join(DATA_DIR, "product_category_name_translation.csv")

# MLflow Settings
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Delivery_Delay_Prediction"

# Feature Settings
TARGET_COLUMN = "delivery_delay"  # Defined later in feature engineering
