"""Unit tests for feature engineering and mathematical derivations."""

import numpy as np
import pandas as pd

from src.features import engineer_features, haversine_km


def test_haversine_identical_coordinates():
    """Haversine distance between identical points must be exactly zero."""
    lat = np.array([-23.5505])
    lon = np.array([-46.6333])
    dist = haversine_km(lat, lon, lat, lon)
    assert np.isclose(dist[0], 0.0, atol=1e-5)


def test_haversine_sao_paulo_to_rio():
    """Haversine distance between Sao Paulo and Rio de Janeiro should be approximately 360 km."""
    # SP: -23.5505, -46.6333 | RJ: -22.9068, -43.1729
    sp_lat, sp_lon = np.array([-23.5505]), np.array([-46.6333])
    rj_lat, rj_lon = np.array([-22.9068]), np.array([-43.1729])
    dist = haversine_km(sp_lat, sp_lon, rj_lat, rj_lon)[0]
    assert 340.0 < dist < 380.0


def test_engineer_features_temporal_and_lags():
    """Verify temporal feature extraction and operational lag calculations."""
    data = {
        "order_purchase_timestamp": ["2017-10-18 16:30:00"],
        "order_approved_at": ["2017-10-18 18:30:00"],
        "order_delivered_carrier_date": ["2017-10-20 16:30:00"],
        "order_estimated_delivery_date": ["2017-11-01 00:00:00"],
        "customer_lat": [-22.90],
        "customer_lng": [-43.17],
        "seller_lat": [-23.55],
        "seller_lng": [-46.63],
        "customer_state": ["RJ"],
        "primary_seller_state": ["SP"],
    }
    df = pd.DataFrame(data)
    result = engineer_features(df)

    # Temporal
    assert result["order_hour"].iloc[0] == 16
    assert result["order_dayofweek"].iloc[0] == 2  # Wednesday
    assert result["order_month"].iloc[0] == 10

    # Lags
    assert np.isclose(result["approval_lag_hrs"].iloc[0], 2.0, atol=1e-2)
    assert np.isclose(result["carrier_lag_days"].iloc[0], 2.0, atol=1e-2)
    assert result["haversine_distance_km"].iloc[0] > 300.0

    # Raw / leaky columns dropped
    for dropped in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "customer_lat",
        "seller_lat",
    ]:
        assert dropped not in result.columns
