"""FastAPI Inference & Explainability Service for Bitcoin Fraud Detection."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from service.predictor import FraudPredictor
from service.schemas import (
    BatchPredictionResponse,
    BatchTransactionInput,
    ExplanationResponse,
    FeatureAttribution,
    HealthResponse,
    PredictionResponse,
    ThresholdResponse,
    ThresholdUpdateRequest,
    TransactionInput,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crypto_fraud_api")

# Global predictor instance
predictor = FraudPredictor(artifacts_dir="artifacts")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: loads ML models and SHAP explainer on startup."""
    logger.info("Starting up Crypto Fraud Inference Service...")
    try:
        predictor.load()
        logger.info("Model and SHAP artifacts successfully loaded.")
    except Exception as e:
        logger.warning(
            f"Startup note: artifacts could not be loaded immediately ({e}). "
            "Endpoints will prompt to run artifact generation if queried."
        )
    yield
    logger.info("Shutting down Crypto Fraud Inference Service.")


app = FastAPI(
    title="Bitcoin Fraud Detection Inference & Explainability API",
    description=(
        "Enterprise-grade machine learning microservice for scoring cryptocurrency transactions, "
        "classifying AML risk tiers, executing decision-theoretic threshold policies, "
        "and providing SHAP regulatory explainability."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_predictor_ready():
    if not predictor.is_loaded:
        # Attempt just-in-time load
        try:
            predictor.load()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    f"Model artifacts not loaded: {str(e)}. "
                    "Run 'python scripts/bootstrap_artifacts.py' or complete pipeline training."
                ),
            )


@app.get("/", tags=["General"])
async def root() -> Dict[str, Any]:
    """Root metadata and API index."""
    return {
        "service": "Bitcoin Fraud Detection Inference & Explainability API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "model_loaded": predictor.is_loaded,
        "model_name": predictor.meta.get("model_name", "Pending initialization"),
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check verifying model status, feature count, and active threshold."""
    return HealthResponse(
        status="healthy" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_name=predictor.meta.get("model_name", "Not loaded"),
        feature_count=len(predictor.feature_order),
        explainer_ready=predictor.explainer is not None,
        current_threshold=predictor.threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_transaction(payload: TransactionInput):
    """Score a single Bitcoin transaction for illicit activity risk."""
    _ensure_predictor_ready()
    start = time.perf_counter()

    prob, is_illicit, risk_tier, action = predictor.predict_single(payload.features)
    latency = round((time.perf_counter() - start) * 1000, 2)

    return PredictionResponse(
        txId=payload.txId,
        fraud_probability=round(prob, 4),
        is_illicit=is_illicit,
        risk_tier=risk_tier,
        action=action,
        threshold_applied=predictor.threshold,
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(payload: BatchTransactionInput):
    """Vectorized scoring of multiple Bitcoin transactions in a single batch."""
    _ensure_predictor_ready()
    start = time.perf_counter()

    features_list = [tx.features for tx in payload.transactions]
    batch_results = predictor.predict_batch(features_list)
    total_latency = (time.perf_counter() - start) * 1000
    avg_latency = round(total_latency / max(1, len(batch_results)), 2)

    predictions = []
    tier_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    illicit_count = 0

    for tx, (prob, is_ill, risk_tier, action) in zip(payload.transactions, batch_results):
        if is_ill:
            illicit_count += 1
        tier_counts[risk_tier.value] = tier_counts.get(risk_tier.value, 0) + 1

        predictions.append(
            PredictionResponse(
                txId=tx.txId,
                fraud_probability=round(prob, 4),
                is_illicit=is_ill,
                risk_tier=risk_tier,
                action=action,
                threshold_applied=predictor.threshold,
                latency_ms=avg_latency,
            )
        )

    return BatchPredictionResponse(
        total_transactions=len(predictions),
        flagged_illicit=illicit_count,
        illicit_rate_pct=round((illicit_count / len(predictions)) * 100, 2),
        summary_by_tier=tier_counts,
        predictions=predictions,
    )


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_transaction(payload: TransactionInput):
    """Generate SHAP local feature attributions and natural language AML reason codes."""
    _ensure_predictor_ready()

    explanation = predictor.explain_single(payload.features, max_reasons=5)

    return ExplanationResponse(
        txId=payload.txId,
        fraud_probability=explanation["fraud_probability"],
        baseline_expected_value=explanation["baseline_expected_value"],
        risk_tier=explanation["risk_tier"],
        top_risk_drivers=[
            FeatureAttribution(**driver) for driver in explanation["top_risk_drivers"]
        ],
        top_mitigating_factors=[
            FeatureAttribution(**factor) for factor in explanation["top_mitigating_factors"]
        ],
        compliance_reason_summary=explanation["compliance_reason_summary"],
    )


@app.get("/threshold", response_model=ThresholdResponse, tags=["Threshold Tuning"])
async def get_threshold():
    """Retrieve active decision threshold and multi-tier routing boundaries."""
    _ensure_predictor_ready()
    return ThresholdResponse(
        current_threshold=predictor.threshold,
        low_risk_threshold=predictor.policy.low_risk_threshold,
        high_risk_threshold=predictor.policy.high_risk_threshold,
        tier_actions=predictor.policy.tier_labels,
        source="active_memory_policy",
    )


@app.post("/threshold", response_model=ThresholdResponse, tags=["Threshold Tuning"])
async def update_threshold(payload: ThresholdUpdateRequest):
    """Dynamically adjust the decision threshold and risk tier boundaries in production."""
    _ensure_predictor_ready()
    try:
        predictor.update_thresholds(
            threshold=payload.threshold,
            low_risk_threshold=payload.low_risk_threshold,
            high_risk_threshold=payload.high_risk_threshold,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return ThresholdResponse(
        current_threshold=predictor.threshold,
        low_risk_threshold=predictor.policy.low_risk_threshold,
        high_risk_threshold=predictor.policy.high_risk_threshold,
        tier_actions=predictor.policy.tier_labels,
        source="dynamic_api_update",
    )
