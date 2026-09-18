"""Data validation service using Great Expectations and schema assertions.

Defines validation expectations:
- Schema column presence and type checks
- Numeric value ranges (positive prices, freights, items, valid geographic coordinates)
- Allowed categorical domains (valid Brazilian states, recognized payment methods)
- Target leakage firewall checks (rejecting post-delivery columns)
- Explicit policy: REJECT bad payloads before inference with HTTP 422 / error details
"""

from typing import Any, List, Tuple
import pandas as pd

from config.logging_config import logger
from config.settings import settings
from src.data import FORBIDDEN_LEAKAGE_COLUMNS, order_dict_to_dataframe

try:
    from great_expectations.dataset import PandasDataset
    HAS_GX = True
except ImportError:
    HAS_GX = False


class DataValidator:
    """Validates incoming order data against domain expectations before model inference."""

    def __init__(self):
        self.validation_config = settings.validation

    def validate(self, order_data: Any) -> Tuple[bool, List[str]]:
        """Validate an order dictionary, list of orders, or DataFrame.

        Returns:
            Tuple of (is_valid: bool, error_messages: List[str])
        """
        errors: List[str] = []

        if isinstance(order_data, pd.DataFrame):
            df = order_data.copy()
        else:
            df = order_dict_to_dataframe(order_data)

        # 1. Target Leakage Firewall Check
        for col in FORBIDDEN_LEAKAGE_COLUMNS:
            if col in df.columns and df[col].notna().any():
                errors.append(
                    f"Target leakage violation: forbidden post-delivery column '{col}' detected in payload."
                )

        # 2. Great Expectations Checks
        if HAS_GX:
            gx_df = PandasDataset(df)

            # Check positive ranges for financial / quantity attributes
            if "total_price" in df.columns:
                res = gx_df.expect_column_values_to_be_between(
                    "total_price", min_value=self.validation_config.min_price
                )
                if not res["success"]:
                    errors.append(
                        f"Validation error on total_price: values must be >= {self.validation_config.min_price}"
                    )

            if "total_freight" in df.columns:
                res = gx_df.expect_column_values_to_be_between(
                    "total_freight", min_value=self.validation_config.min_freight
                )
                if not res["success"]:
                    errors.append(
                        f"Validation error on total_freight: values must be >= {self.validation_config.min_freight}"
                    )

            if "item_count" in df.columns:
                res = gx_df.expect_column_values_to_be_between(
                    "item_count", min_value=self.validation_config.min_items
                )
                if not res["success"]:
                    errors.append(
                        f"Validation error on item_count: values must be >= {self.validation_config.min_items}"
                    )

            # Geographic bounds
            for lat_col in ["customer_lat", "seller_lat"]:
                if lat_col in df.columns:
                    valid_lat = df[lat_col].dropna()
                    if not valid_lat.empty and not valid_lat.between(-90.0, 90.0).all():
                        errors.append(f"Validation error on {lat_col}: latitude must be between -90 and 90.")

            for lng_col in ["customer_lng", "seller_lng"]:
                if lng_col in df.columns:
                    valid_lng = df[lng_col].dropna()
                    if not valid_lng.empty and not valid_lng.between(-180.0, 180.0).all():
                        errors.append(f"Validation error on {lng_col}: longitude must be between -180 and 180.")

            # Categorical domain checks
            if "dominant_payment_type" in df.columns:
                res = gx_df.expect_column_values_to_be_in_set(
                    "dominant_payment_type",
                    self.validation_config.allowed_payment_types,
                )
                if not res["success"]:
                    errors.append(
                        f"Validation error on dominant_payment_type: unrecognized value. "
                        f"Allowed: {self.validation_config.allowed_payment_types}"
                    )

            for state_col in ["customer_state", "primary_seller_state"]:
                if state_col in df.columns:
                    res = gx_df.expect_column_values_to_be_in_set(
                        state_col, self.validation_config.allowed_brazilian_states
                    )
                    if not res["success"]:
                        errors.append(
                            f"Validation error on {state_col}: invalid Brazilian state code."
                        )
        else:
            # Fallback pure pandas validation if GX is unavailable
            if "total_price" in df.columns and (df["total_price"] < 0).any():
                errors.append("Validation error: total_price cannot be negative.")
            if "total_freight" in df.columns and (df["total_freight"] < 0).any():
                errors.append("Validation error: total_freight cannot be negative.")

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Data validation failed with {len(errors)} errors: {errors}")

        return is_valid, errors


# Singleton validator instance
data_validator = DataValidator()
