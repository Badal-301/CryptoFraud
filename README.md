# CryptoFraud — Bitcoin Fraud Detection & Explainability Platform

An enterprise-grade, regulatory-ready machine learning system for detecting illicit Bitcoin transactions on the Elliptic Bitcoin transaction graph. Built with gradient boosting, graph feature engineering, decision-theoretic threshold tuning, SHAP explainability, and a high-performance FastAPI inference microservice.

---

## Key Features

1. **Graph-Enriched ML Pipeline**:
   - 166 baseline local and aggregated neighborhood features + 3 engineered graph topology features (`graph_in_degree`, `graph_out_degree`, `graph_pagerank`).
   - Chronological split (time steps 1–34 train, 35–49 test) avoiding future data leakage.
   - Robust class imbalance mitigation via cost-sensitive learning (`scale_pos_weight`) and SMOTE.

2. **Decision-Theoretic Threshold Tuning (`src/threshold_tuning.py`)**:
   - **Cost-Utility Optimization**: Replaces the arbitrary 0.5 default with a business loss-minimizing threshold balancing False Negative cost ($C_{FN} \approx \$1,000$ for missed fraud / AML penalties) against False Positive cost ($C_{FP} \approx \$30$ for analyst review time).
   - **$F_\beta$ & Recall-Constrained Search**: Target-recall guarantees (e.g. $\ge 85\%$ recall) and $F_2$ optimization favoring fraud capture.
   - **3-Tier Operational Alert Routing**:
     - `AUTO_APPROVE` (Low Risk: $p < t_{\text{low}}$) — Instant automated settlement.
     - `MANUAL_REVIEW` (Medium Risk: $t_{\text{low}} \le p < t_{\text{high}}$) — Investigator queue.
     - `AUTO_FREEZE` (High Risk: $p \ge t_{\text{high}}$) — Immediate fund hold & Suspicious Activity Report (SAR).

3. **SHAP Regulatory Explainability (`src/explainability.py`)**:
   - **Global Interpretability**: Mean $|SHAP|$ feature importance and beeswarm summary distributions isolating key risk catalysts.
   - **Local Interpretability & Reason Codes**: Transaction-level decomposition outputting baseline probability, directional feature attributions, and plain-English compliance statements for FinCEN / FATF audits.
   - **Model-Agnostic & Tree-Path Execution**: Native `TreeExplainer` integration with surrogate fallback.

4. **Production FastAPI Microservice (`service/`)**:
   - Asynchronous, thread-safe model predictor with startup lifespan loading.
   - Real-time single transaction scoring (`POST /predict`), high-throughput batch scoring (`POST /predict/batch`), on-demand SHAP explanations (`POST /explain`), and dynamic runtime threshold adjustments (`POST /threshold`).
   - Comprehensive test suite (`pytest`) with 100% endpoint coverage.

---

## Directory Structure

```text
CryptoFraud/
├── artifacts/                         # Serialized models, scaler, metadata & configs
│   ├── fraud_model.joblib             # Trained XGBoost classifier
│   ├── scaler.joblib                  # Fitted StandardScaler
│   ├── model_meta.json                # Feature ordering & validation metrics
│   ├── threshold_config.json          # Optimal decision thresholds & tier policies
│   └── shap_explainer.joblib          # Persisted SHAP explainer
├── service/                           # FastAPI inference microservice
│   ├── __init__.py
│   ├── main.py                        # FastAPI application routes & lifespan
│   ├── predictor.py                   # In-memory predictor & explainability manager
│   └── schemas.py                     # Pydantic validation schemas
├── src/                               # Core ML engines
│   ├── __init__.py
│   ├── explainability.py              # SHAP explainability & reason-code engine
│   └── threshold_tuning.py            # Cost-utility & multi-tier threshold tuner
├── scripts/
│   ├── bootstrap_artifacts.py         # Automated artifact generator & pipeline tester
│   └── update_notebook.py             # Pipeline notebook synchronizer
├── tests/                             # Unit & integration test suites
│   ├── test_explainability.py         # SHAP attribution unit tests
│   ├── test_service.py                # FastAPI endpoint integration tests
│   └── test_threshold_tuning.py       # Threshold optimizer unit tests
├── crypto_fraud_detection_pipeline.ipynb # End-to-end research notebook
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Quickstart & Running the API

### 1. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Generate or Train Artifacts

To generate verified model artifacts, scaler, threshold policy, and SHAP explainer:

```bash
python scripts/bootstrap_artifacts.py
```

### 3. Launch the FastAPI Service

Start the server using `uvicorn`:

```bash
uvicorn service.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive API documentation will be available at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## API Endpoints & Usage

### 1. Health Check
`GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "XGBoost (scale_pos_weight)",
  "feature_count": 168,
  "explainer_ready": true,
  "current_threshold": 0.0198
}
```

### 2. Real-Time Transaction Scoring
`POST /predict`

**Request Body**:
```json
{
  "txId": "btc_tx_839210",
  "features": {
    "feat_1": 2.45,
    "feat_2": 1.82,
    "graph_in_degree": 2.0,
    "graph_out_degree": 14.0,
    "graph_pagerank": 0.0042
  }
}
```

**Response**:
```json
{
  "txId": "btc_tx_839210",
  "fraud_probability": 0.9821,
  "is_illicit": true,
  "risk_tier": "HIGH",
  "action": "AUTO_FREEZE",
  "threshold_applied": 0.0198,
  "latency_ms": 3.14
}
```

### 3. On-Demand SHAP Compliance Explanation
`POST /explain`

**Request Body**:
```json
{
  "txId": "btc_tx_839210",
  "features": {
    "feat_1": 2.45,
    "feat_2": 1.82,
    "graph_out_degree": 14.0
  }
}
```

**Response**:
```json
{
  "txId": "btc_tx_839210",
  "fraud_probability": 0.9821,
  "baseline_expected_value": 0.1754,
  "risk_tier": "HIGH",
  "top_risk_drivers": [
    {
      "feature": "graph_out_degree",
      "feature_value": 14.0,
      "shap_attribution": 5.8214,
      "impact": "INCREASES_RISK"
    },
    {
      "feature": "feat_1",
      "feature_value": 2.45,
      "shap_attribution": 0.1842,
      "impact": "INCREASES_RISK"
    }
  ],
  "top_mitigating_factors": [],
  "compliance_reason_summary": "Fraud probability 98.2%. Primary illicit risk drivers: graph_out_degree (+5.821, val=14.00), feat_1 (+0.184, val=2.45)."
}
```

### 4. Dynamic Threshold Updating
`POST /threshold`

**Request Body**:
```json
{
  "threshold": 0.025,
  "low_risk_threshold": 0.015,
  "high_risk_threshold": 0.70
}
```

---

## Running the Automated Test Suite

Run all unit and integration tests:

```bash
pytest tests/ -v
```

Output:
```text
tests/test_explainability.py::test_explainer_initialization PASSED
tests/test_explainability.py::test_global_feature_importance PASSED
tests/test_explainability.py::test_explain_transaction PASSED
tests/test_explainability.py::test_explainer_save_and_load PASSED
tests/test_service.py::test_root_endpoint PASSED
tests/test_service.py::test_health_endpoint PASSED
tests/test_service.py::test_predict_single_transaction_licit PASSED
tests/test_service.py::test_predict_single_transaction_high_risk PASSED
tests/test_service.py::test_predict_batch PASSED
tests/test_service.py::test_explain_endpoint PASSED
tests/test_service.py::test_threshold_lifecycle PASSED
tests/test_service.py::test_invalid_transaction_payload PASSED
tests/test_threshold_tuning.py::test_evaluate_threshold PASSED
tests/test_threshold_tuning.py::test_optimize_f_beta PASSED
tests/test_threshold_tuning.py::test_optimize_cost PASSED
tests/test_threshold_tuning.py::test_target_recall PASSED
tests/test_threshold_tuning.py::test_multi_tier_policy PASSED
tests/test_threshold_tuning.py::test_export_config PASSED

18 passed in 10.35s
```