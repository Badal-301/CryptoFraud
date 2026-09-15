"""Unit tests for the SHAP explainability module."""

import sys
from pathlib import Path

# Ensure root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from src.explainability import FraudExplainer


@pytest.fixture
def trained_model_and_data():
    np.random.seed(42)
    n = 200
    features = [f"feat_{i}" for i in range(1, 11)] + ["graph_out_degree"]
    X = pd.DataFrame(np.random.randn(n, len(features)), columns=features)
    y = (X["graph_out_degree"] > 1.0).astype(int).values

    model = XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, features, X


def test_explainer_initialization(trained_model_and_data):
    model, features, X = trained_model_and_data
    explainer = FraudExplainer(model, features, background_sample=X.iloc[:20])
    assert explainer.model is model
    assert len(explainer.feature_names) == len(features)
    assert 0.0 <= explainer.expected_value <= 1.0


def test_global_feature_importance(trained_model_and_data):
    model, features, X = trained_model_and_data
    explainer = FraudExplainer(model, features)
    importance_df = explainer.get_global_feature_importance(X, max_features=5)

    assert len(importance_df) <= 5
    assert "feature" in importance_df.columns
    assert "mean_abs_shap" in importance_df.columns
    assert importance_df["mean_abs_shap"].iloc[0] >= importance_df["mean_abs_shap"].iloc[-1]


def test_explain_transaction(trained_model_and_data):
    model, features, X = trained_model_and_data
    explainer = FraudExplainer(model, features)

    sample_txn = X.iloc[0].to_dict()
    explanation = explainer.explain_transaction(sample_txn, max_reasons=3)

    assert "fraud_probability" in explanation
    assert "baseline_expected_value" in explanation
    assert "top_risk_drivers" in explanation
    assert "top_mitigating_factors" in explanation
    assert "compliance_reason_summary" in explanation
    assert isinstance(explanation["compliance_reason_summary"], str)


def test_explainer_save_and_load(trained_model_and_data, tmp_path):
    model, features, X = trained_model_and_data
    explainer = FraudExplainer(model, features)
    save_file = tmp_path / "test_explainer.joblib"

    explainer.save(save_file)
    assert save_file.exists()

    loaded = FraudExplainer.load(save_file)
    assert len(loaded.feature_names) == len(features)
    sample_txn = X.iloc[0].to_dict()
    exp = loaded.explain_transaction(sample_txn)
    assert "fraud_probability" in exp
