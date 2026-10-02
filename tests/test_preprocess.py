"""Unit tests for preprocessing service and ColumnTransformer pipeline."""

import numpy as np

from src.data import order_dict_to_dataframe
from src.preprocess import preprocessor_service


def test_preprocessor_shape_and_columns(sample_order_dict):
    """Verify ColumnTransformer outputs exactly 150 features matching the training feature list."""
    df = order_dict_to_dataframe(sample_order_dict)
    transformed = preprocessor_service.transform(df)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (1, 150)
    assert len(preprocessor_service.feature_names) == 150


def test_preprocessor_handles_missing_values(sample_order_dict):
    """Verify that imputers in the pipeline handle missing values without crashing."""
    sparse_order = sample_order_dict.copy()
    sparse_order["customer_lat"] = None
    sparse_order["customer_lng"] = None
    sparse_order["seller_lat"] = None
    sparse_order["seller_lng"] = None
    sparse_order["order_approved_at"] = None
    sparse_order["order_delivered_carrier_date"] = None
    sparse_order["primary_product_category"] = None

    df = order_dict_to_dataframe(sparse_order)
    transformed = preprocessor_service.transform(df)

    assert transformed.shape == (1, 150)
    # Ensure no NaN remains after median/most_frequent imputation and scaling
    assert not np.isnan(transformed).any()
