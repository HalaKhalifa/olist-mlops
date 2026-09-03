# EDA Summary — Training Set

- Training rows: 67,529
- Late orders: 5,478 (8.11%)

## Key Findings
- Strongest signal: carrier_lag_days (r=+0.216), haversine_distance_km (r=+0.071)
- Categorical signals: customer_state (RJ/BA 13-14% late vs SP 5.8%)
- Temporal: Late rate slightly higher at midnight and noon orders
- Missing values: customer_lat/lng (0.3%), seller_lat/lng (0.2%) — impute with median
