from pathlib import Path
from typing import Dict, List, Union
import pandas as pd

from config.logging_config import logger

# Raw order schema columns expected for feature engineering
EXPECTED_RAW_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_estimated_delivery_date",
    "customer_state",
    "primary_seller_state",
    "primary_product_category",
    "dominant_payment_type",
    "item_count",
    "total_price",
    "avg_item_price",
    "total_freight",
    "avg_item_freight",
    "total_weight_g",
    "total_volume_cm3",
    "num_sellers",
    "total_payment_value",
    "payment_installments_max",
    "payment_transactions_count",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
]

# Columns strictly forbidden at inference time (target leakage firewall)
FORBIDDEN_LEAKAGE_COLUMNS = [
    "order_delivered_customer_date",
    "actual_delivery_days",
    "delivery_delay_days",
    "is_late",
]


def load_parquet_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a Parquet dataset into a pandas DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found at: {path}")
    logger.info(f"Loading dataset from Parquet file: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


def order_dict_to_dataframe(order_data: Union[Dict, List[Dict]]) -> pd.DataFrame:
    """Convert single order dictionary or list of order dictionaries to DataFrame."""
    if isinstance(order_data, dict):
        df = pd.DataFrame([order_data])
    else:
        df = pd.DataFrame(order_data)

    # Convert timestamp string columns to datetime
    timestamp_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date",
    ]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
