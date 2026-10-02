"""Pydantic v2 schemas for API requests, responses, and model metadata."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class OrderInput(BaseModel):
    """Schema representing an incoming order for late delivery prediction."""

    model_config = ConfigDict(extra="allow")

    order_id: Optional[str] = Field(
        default="sample_order_001",
        description="Unique identifier for the order",
        examples=["9e8835f613d3d61b5c4aeae2550b893a"],
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the customer",
        examples=["1b5db70f5a471093cb88e558bdee77ae"],
    )
    order_purchase_timestamp: str = Field(
        ...,
        description="ISO timestamp when the order was purchased",
        examples=["2017-10-18 16:42:42"],
    )
    order_approved_at: Optional[str] = Field(
        default=None,
        description="ISO timestamp when payment was approved",
        examples=["2017-10-18 16:56:45"],
    )
    order_delivered_carrier_date: Optional[str] = Field(
        default=None,
        description="ISO timestamp when order was dispatched to carrier",
        examples=["2017-10-24 19:22:45"],
    )
    order_estimated_delivery_date: str = Field(
        ...,
        description="Estimated delivery SLA promised to customer at purchase",
        examples=["2017-11-08 00:00:00"],
    )
    customer_state: str = Field(
        ...,
        description="Two-letter Brazilian state code for customer",
        examples=["RJ"],
    )
    customer_city: Optional[str] = Field(
        default=None, description="City of the customer", examples=["rio de janeiro"],
    )
    customer_zip_code_prefix: Optional[str] = Field(
        default=None, description="Customer postal code prefix", examples=["22430"],
    )
    customer_lat: Optional[float] = Field(
        default=None, description="Customer geolocation latitude", examples=[-22.98197],
    )
    customer_lng: Optional[float] = Field(
        default=None,
        description="Customer geolocation longitude",
        examples=[-43.21999],
    )
    primary_seller_state: str = Field(
        ...,
        description="Two-letter Brazilian state code for primary seller",
        examples=["SP"],
    )
    primary_seller_city: Optional[str] = Field(
        default=None, description="City of primary seller", examples=["sao paulo"],
    )
    primary_seller_zip_code: Optional[float] = Field(
        default=None, description="Seller zip code prefix", examples=[8270.0],
    )
    seller_lat: Optional[float] = Field(
        default=None, description="Seller geolocation latitude", examples=[-23.56128],
    )
    seller_lng: Optional[float] = Field(
        default=None, description="Seller geolocation longitude", examples=[-46.46197],
    )
    primary_product_category: Optional[str] = Field(
        default="outros",
        description="Category name of the highest-priced item",
        examples=["telephony"],
    )
    dominant_payment_type: Optional[str] = Field(
        default="credit_card",
        description="Primary payment method",
        examples=["credit_card"],
    )
    item_count: float = Field(
        default=1.0, ge=1.0, description="Number of items in the order", examples=[1.0],
    )
    total_price: float = Field(
        ..., ge=0.0, description="Total items value in BRL", examples=[70.9],
    )
    avg_item_price: Optional[float] = Field(
        default=None, ge=0.0, description="Average item value", examples=[70.9],
    )
    total_freight: float = Field(
        ..., ge=0.0, description="Total freight value in BRL", examples=[14.25],
    )
    avg_item_freight: Optional[float] = Field(
        default=None, ge=0.0, description="Average freight per item", examples=[14.25],
    )
    total_weight_g: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total package weight in grams",
        examples=[250.0],
    )
    total_volume_cm3: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total package volume in cubic centimeters",
        examples=[1280.0],
    )
    num_sellers: Optional[float] = Field(
        default=1.0,
        ge=1.0,
        description="Count of distinct sellers fulfilled",
        examples=[1.0],
    )
    total_payment_value: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Total transaction payment value",
        examples=[85.15],
    )
    payment_installments_max: Optional[float] = Field(
        default=1.0,
        ge=1.0,
        description="Max payment installments chosen",
        examples=[3.0],
    )
    payment_transactions_count: Optional[float] = Field(
        default=1.0,
        ge=1.0,
        description="Count of payment transactions",
        examples=[1.0],
    )


class BatchOrderInput(BaseModel):
    """Schema for batch prediction requests."""

    orders: List[OrderInput] = Field(
        ..., min_length=1, description="List of orders to predict in batch",
    )


class PredictionResponse(BaseModel):
    """Schema for single order prediction response."""

    order_id: str
    prediction: int = Field(description="Binary late indicator: 1 = Late, 0 = On-Time")
    label: str = Field(description="Human readable outcome: 'late' or 'on_time'")
    late_probability: float = Field(
        description="Estimated probability of late delivery in [0, 1]"
    )
    model_name: str
    model_version: str
    latency_ms: float = Field(description="Inference latency in milliseconds")


class BatchPredictionResponse(BaseModel):
    """Schema for batch order prediction response."""

    predictions: List[PredictionResponse]
    total_orders: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    """Schema for service health check."""

    status: str
    service: str
    version: str
    model_loaded: bool
    environment: str


class ModelInfoResponse(BaseModel):
    """Schema for model info and metadata."""

    model_name: str
    model_version: str
    model_stage: str
    model_type: str
    total_features: int
    hyperparameters: Dict[str, Any]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]


class MonitoringSummaryResponse(BaseModel):
    """Live prediction-log summary used for drift and latency alerts."""

    timestamp: str
    total_predictions: int
    late_count: int = 0
    on_time_count: int = 0
    current_late_rate: float = 0.0
    baseline_late_rate: float = 0.0
    rate_difference: float = 0.0
    drift_detected: bool = False
    mean_probability: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    active_alerts: List[str] = []
    message: Optional[str] = None
