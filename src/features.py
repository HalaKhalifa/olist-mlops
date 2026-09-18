"""Feature engineering module for Olist late delivery prediction.

Replicates the exact deterministic transformations defined in Notebook 5:
- Haversine distance from customer to seller geolocations
- Purchase timestamp temporal components (hour, day of week, month)
- Operational lag features (approval lag, carrier dispatch lag)
- Estimated delivery window duration
- Strict leakage elimination
"""

from typing import List
import numpy as np
import pandas as pd

# Columns to drop after feature extraction (IDs, raw coordinates, raw timestamps)
DROP_COLS = [
    "order_id",
    "customer_id",
    "customer_unique_id",
    "primary_seller_id",
    "order_status",
    "order_delivered_customer_date",
    "actual_delivery_days",
    "delivery_delay_days",
    "order_delivered_carrier_date",
    "customer_city",
    "primary_seller_city",
    "customer_lat",
    "customer_lng",
    "seller_lat",
    "seller_lng",
    "customer_zip_code_prefix",
    "primary_seller_zip_code",
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]

NUMERIC_COLS: List[str] = [
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
    "estimated_delivery_days",
    "order_hour",
    "order_dayofweek",
    "order_month",
    "approval_lag_hrs",
    "carrier_lag_days",
    "haversine_distance_km",
]

CATEGORICAL_COLS: List[str] = [
    "customer_state",
    "primary_seller_state",
    "primary_product_category",
    "dominant_payment_type",
]


def haversine_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Calculate great-circle distance between two points on Earth in kilometers."""
    radius = 6371.0
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    )
    # Clip for numerical stability within [-1, 1]
    a = np.clip(a, 0.0, 1.0)
    return 2.0 * radius * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic feature transformations to an order DataFrame.

    Produces the exact feature representations expected by the ColumnTransformer.
    """
    df = df.copy()

    # Ensure datetime format
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_estimated_delivery_date",
    ]:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # 1. Temporal features from purchase timestamp
    if "order_purchase_timestamp" in df.columns:
        df["order_hour"] = df["order_purchase_timestamp"].dt.hour
        df["order_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
        df["order_month"] = df["order_purchase_timestamp"].dt.month
    else:
        for col in ["order_hour", "order_dayofweek", "order_month"]:
            if col not in df.columns:
                df[col] = np.nan

    # 2. Operational lag features
    if "order_approved_at" in df.columns and "order_purchase_timestamp" in df.columns:
        df["approval_lag_hrs"] = (
            df["order_approved_at"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 3600.0
    elif "approval_lag_hrs" not in df.columns:
        df["approval_lag_hrs"] = np.nan

    if (
        "order_delivered_carrier_date" in df.columns
        and "order_purchase_timestamp" in df.columns
    ):
        df["carrier_lag_days"] = (
            df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
        ).dt.total_seconds() / 86400.0
    elif "carrier_lag_days" not in df.columns:
        df["carrier_lag_days"] = np.nan

    # 3. Estimated delivery window
    if "estimated_delivery_days" not in df.columns:
        if (
            "order_estimated_delivery_date" in df.columns
            and "order_purchase_timestamp" in df.columns
        ):
            df["estimated_delivery_days"] = (
                df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
            ).dt.total_seconds() / 86400.0
        else:
            df["estimated_delivery_days"] = np.nan

    # 4. Haversine distance
    geo_cols = ["customer_lat", "customer_lng", "seller_lat", "seller_lng"]
    if all(col in df.columns for col in geo_cols):
        geo_mask = df[geo_cols].notna().all(axis=1)
        df["haversine_distance_km"] = np.nan
        if geo_mask.any():
            df.loc[geo_mask, "haversine_distance_km"] = haversine_km(
                df.loc[geo_mask, "customer_lat"].astype(float).values,
                df.loc[geo_mask, "customer_lng"].astype(float).values,
                df.loc[geo_mask, "seller_lat"].astype(float).values,
                df.loc[geo_mask, "seller_lng"].astype(float).values,
            )
    elif "haversine_distance_km" not in df.columns:
        df["haversine_distance_km"] = np.nan

    # 5. Drop leaky and raw intermediate columns
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    return df
