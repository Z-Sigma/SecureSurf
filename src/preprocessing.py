import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def process_geolocation(geolocation_df):
    """Interpolate lat/long for zip codes."""
    logger.info("Interpolating geolocation zip codes...")
    df = geolocation_df.drop(['geolocation_city', 'geolocation_state'], axis=1)
    df = df.groupby('geolocation_zip_code_prefix').mean()
    # Reindex to fill missing zip codes and interpolate
    df = df.reindex(np.arange(df.index.min(), df.index.max() + 1)).interpolate(method='nearest')
    return df

def clean_products(products_df):
    """Drop unnecessary columns and potentially group categories."""
    keep_cols = [
        'product_id', 'product_category_name', 'product_weight_g',
        'product_length_cm', 'product_height_cm', 'product_width_cm'
    ]
    df = products_df[keep_cols].copy()
    return df

def merge_datasets(data_dict):
    """Merge all relevant datasets into a single master dataframe."""
    orders = data_dict['orders']
    items = data_dict['order_items']
    products = data_dict['products']
    customers = data_dict['customers']
    sellers = data_dict['sellers']
    geolocation = process_geolocation(data_dict['geolocation'])

    # Process customers with geo
    customers_geo = pd.merge(
        customers[['customer_id', 'customer_zip_code_prefix']], 
        geolocation, 
        left_on='customer_zip_code_prefix', 
        right_index=True
    ).drop(['customer_zip_code_prefix'], axis=1)
    customers_geo.columns = ['customer_id', 'customer_lat', 'customer_lng']

    # Process sellers with geo
    sellers_geo = pd.merge(
        sellers[['seller_id', 'seller_zip_code_prefix']], 
        geolocation, 
        left_on='seller_zip_code_prefix', 
        right_index=True
    ).drop(['seller_zip_code_prefix'], axis=1)
    sellers_geo.columns = ['seller_id', 'seller_lat', 'seller_lng']

    # Master merge
    df = pd.merge(items, orders, on='order_id')
    df = pd.merge(df, products, on='product_id')
    df = pd.merge(df, customers_geo, on='customer_id')
    df = pd.merge(df, sellers_geo, on='seller_id')

    return df
