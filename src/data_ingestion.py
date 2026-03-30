import pandas as pd
from src import config
from src.logger import get_logger

logger = get_logger(__name__)

def load_customers():
    logger.info("Loading customers data")
    return pd.read_csv(config.CUSTOMERS_DATA)

def load_geolocation():
    return pd.read_csv(config.GEOLOCATION_DATA)

def load_order_items():
    return pd.read_csv(config.ORDER_ITEMS_DATA)

def load_order_payments():
    return pd.read_csv(config.ORDER_PAYMENTS_DATA)

def load_order_reviews():
    return pd.read_csv(config.ORDER_REVIEWS_DATA)

def load_orders():
    return pd.read_csv(config.ORDERS_DATA)

def load_products():
    return pd.read_csv(config.PRODUCTS_DATA)

def load_sellers():
    return pd.read_csv(config.SELLERS_DATA)

def load_category_translation():
    return pd.read_csv(config.CATEGORY_TRANSLATION_DATA)

def load_all_data():
    """Load all datasets into a dictionary."""
    data = {
        "customers": load_customers(),
        "geolocation": load_geolocation(),
        "order_items": load_order_items(),
        "order_payments": load_order_payments(),
        "order_reviews": load_order_reviews(),
        "orders": load_orders(),
        "products": load_products(),
        "sellers": load_sellers(),
        "category_translation": load_category_translation(),
    }
    return data
