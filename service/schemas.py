"""Pydantic schemas for the Crypto Fraud Detection FastAPI Inference Service."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class RiskTier(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class TransactionInput(BaseModel):
    """Input payload for a single transaction scoring request."""

    txId: Optional[Union[str, int]] = Field(
        default=None, description="Unique identifier for the Bitcoin transaction"
    )
    features: Dict[str, float] = Field(
        ...,
        description="Feature mapping: keys should match feat_1..feat_165 or graph_* attributes. Missing features will default to 0.0.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "txId": "230425980",
                "features": {
                    "feat_1": -0.1714,
                    "feat_2": -0.1846,
                    "feat_3": -1.2013,
                    "graph_in_degree": 4.0,
                    "graph_out_degree": 1.0,
                    "graph_pagerank": 0.00014,
                },
            }
        }
    }


class BatchTransactionInput(BaseModel):
    """Input payload for batch transaction scoring."""

    transactions: List[TransactionInput] = Field(
        ..., min_length=1, max_length=5000, description="List of transactions to score"
    )


class PredictionResponse(BaseModel):
    """Risk prediction result for an individual transaction."""

    txId: Optional[Union[str, int]] = None
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    is_illicit: bool
    risk_tier: RiskTier
    action: str = Field(
        ..., description="Recommended operational action: AUTO_APPROVE, MANUAL_REVIEW, or AUTO_FREEZE"
    )
    threshold_applied: float
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    """Aggregated batch prediction response."""

    total_transactions: int
    flagged_illicit: int
    illicit_rate_pct: float
    summary_by_tier: Dict[str, int]
    predictions: List[PredictionResponse]


class FeatureAttribution(BaseModel):
    """Local attribution for a specific feature from SHAP explainability."""

    feature: str
    feature_value: float
    shap_attribution: float
    impact: str


class ExplanationResponse(BaseModel):
    """SHAP-derived compliance explanation for a transaction."""

    txId: Optional[Union[str, int]] = None
    fraud_probability: float
    baseline_expected_value: float
    risk_tier: RiskTier
    top_risk_drivers: List[FeatureAttribution]
    top_mitigating_factors: List[FeatureAttribution]
    compliance_reason_summary: str


class ThresholdUpdateRequest(BaseModel):
    """Payload to dynamically adjust operational decision threshold."""

    threshold: Optional[float] = Field(None, ge=0.01, le=0.99)
    low_risk_threshold: Optional[float] = Field(None, ge=0.01, le=0.99)
    high_risk_threshold: Optional[float] = Field(None, ge=0.01, le=0.99)

    @field_validator("high_risk_threshold")
    @classmethod
    def validate_tiers(cls, v, info):
        low = info.data.get("low_risk_threshold")
        if v is not None and low is not None and v <= low:
            raise ValueError("high_risk_threshold must be strictly greater than low_risk_threshold")
        return v


class ThresholdResponse(BaseModel):
    """Operational threshold configuration."""

    current_threshold: float
    low_risk_threshold: float
    high_risk_threshold: float
    tier_actions: Dict[str, str]
    source: str


class HealthResponse(BaseModel):
    """Service health and metadata diagnostic response."""

    status: str
    model_loaded: bool
    model_name: str
    feature_count: int
    explainer_ready: bool
    current_threshold: float
