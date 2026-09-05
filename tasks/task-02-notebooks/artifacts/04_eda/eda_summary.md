# Detailed EDA Findings Summary — Training Split (N=67,529)

## 1. Class Distribution & Imbalance Diagnosis
- **Training Population**: 67,529 orders.
- **On-Time (0)**: 62,051 (91.89%)
- **Late (1)**: 5,478 (8.11%)
- **Imbalance Ratio**: 11.33 : 1.
- **Decision**: Accuracy is strictly invalid. Models must be optimized for ROC-AUC, PR-AUC, and minority F1/Recall.

## 2. Key Predictive Signals
- **Geographic Distance (`haversine_distance_km`)**: Strong positive correlation with late deliveries (r = +0.071, p < 1e-10).
- **Interstate Logistics Friction**: Cross-state deliveries exhibit a 9.92% delay rate versus 4.09% for same-state deliveries (2.4x higher).
- **State Disparities**: Remote/Northeastern states (RJ: 13.4%, BA: 14.2%, MA: 15.6%) experience substantially higher delays than Sao Paulo (SP: 5.8%).
- **Estimated Window Buffer (`estimated_delivery_days`)**: Longer promised buffers correlate negatively with late deliveries (r = -0.059).
- **Bulky & Expensive Freight**: Higher freight costs and item weight exhibit higher frequency of delivery delays.
- **Payment Mechanism**: Boleto orders have slightly higher late rate (9.0%) vs credit card (7.9%) due to payment verification latency.

## 3. Data Integrity, Outliers & Missingness
- **Missing Coordinates**: 0.3% customer and 0.2% seller lat/lng missing. Orders with missing coordinates experience 12.2% late rate vs 8.1% overall.
- **Skewness**: Freight value, weight, and price exhibit heavy right tails (skewness > 3.0).
- **Nonsensical Ranges**: Zero negative prices, negative freight, or negative delivery estimates were found.

## 4. Feature Engineering Decisions for Notebook 5
1. **Imputation**: Median imputer for numerical features; most frequent imputer for categoricals.
2. **Scaling**: StandardScaler for numerical features to handle scale disparities across price, weight, and distance.
3. **Categorical Encoding**: One-Hot Encoding for state (`customer_state`, `primary_seller_state`), payment type, and top product categories.
4. **Derived Features**: Haversine distance (`haversine_distance_km`), order purchase hour, day of week, month, and payment approval lag in hours.
5. **Prediction Time Isolation**: Post-delivery attributes are dropped. Features must only represent information available at inference time.
