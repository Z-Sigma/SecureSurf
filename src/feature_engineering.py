import pandas as pd
import numpy as np
from geopy import distance
from src.logger import get_logger

logger = get_logger(__name__)

def calculate_distances(df):
    """Calculate geodisic distance between seller and customer."""
    logger.info("Computing geopy distance for all order rows. This can take a moment...")
    df['distance'] = df.apply(
        lambda x: distance.distance(
            (x['seller_lat'], x['seller_lng']),
            (x['customer_lat'], x['customer_lng'])
        ).km, axis=1
    )
    # Drop raw lat/lng after distance is calculated to save space
    df = df.drop(['seller_lat', 'seller_lng', 'customer_lat', 'customer_lng'], axis=1)
    return df

def extract_product_features(df):
    """Calculate product max dimension and volume."""
    df['product_max_cm'] = df[['product_length_cm', 'product_height_cm', 'product_width_cm']].max(axis=1)
    df['product_volume_cm'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']
    df = df.drop(['product_length_cm', 'product_height_cm', 'product_width_cm'], axis=1)
    df['product_category_name'] = df['product_category_name'].fillna('other')
    return df

def aggregate_by_order(df):
    """Aggregate item-level data to order-level features."""
    agg_df = df.groupby('order_id').agg({
        'price': ['count', 'max', 'sum'],
        'freight_value': ['max', 'sum'],
        'product_category_name': lambda x: list(x.unique()),
        'product_weight_g': ['max', 'sum'],
        'distance': ['max', 'sum'],
        'product_max_cm': ['max', 'sum'],
        'product_volume_cm': ['max', 'sum']
    })
    
    agg_df.columns = [
        'item_count', 'price_max', 'price_sum', 
        'freight_value_max', 'freight_value_sum', 
        'product_category_names', 'product_weight_g_max',
        'product_weight_g_sum', 'distance_max', 'distance_sum', 
        'product_max_cm_max', 'product_max_cm_sum', 
        'product_volume_cm_max', 'product_volume_cm_sum'
    ]
    return agg_df

def process_time_features(df, orders_df):
    """Calculate delivery delay and other time features."""
    # We need the original order timestamps
    df = pd.merge(df, orders_df[['order_id', 'order_purchase_timestamp', 'order_delivered_customer_date']], on='order_id')
    
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['order_delivered_customer_date'] = pd.to_datetime(df['order_delivered_customer_date'])
    
    # Target variable: delivery delay in days
    df['delivery_delay'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 86400
    
    # Drop rows with missing delivery dates (cannot train on these)
    df = df.dropna(subset=['delivery_delay'])
    
    # Extract temporal features
    df['purchase_hour'] = df['order_purchase_timestamp'].dt.hour
    df['purchase_dayofweek'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month
    
    return df

def encode_categories(df, all_categories):
    """One-hot encode product categories from the list column."""
    for cat in all_categories:
        df[cat] = df['product_category_names'].apply(lambda x: 1 if cat in x else 0)
    return df.drop(['product_category_names'], axis=1)
