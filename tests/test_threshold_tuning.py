"""Unit tests for the decision-theoretic threshold tuning engine."""

import sys
from pathlib import Path

# Ensure root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.threshold_tuning import MultiTierPolicy, ThresholdTuner


@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    n = 1000
    # Imbalanced setup: 50 positive (fraud), 950 negative (licit)
    y_true = np.array([0] * 950 + [1] * 50)
    # Licit scores concentrated near 0.05, fraud scores concentrated near 0.70
    y_scores = np.concatenate([
        np.random.beta(1, 15, size=950),
        np.random.beta(5, 2, size=50),
    ])
    return y_true, y_scores


def test_evaluate_threshold(sample_predictions):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)

    m = tuner.evaluate_threshold(0.5, cost_fn=1000.0, cost_fp=30.0)
    assert 0.0 <= m.precision <= 1.0
    assert 0.0 <= m.recall <= 1.0
    assert 0.0 <= m.f1 <= 1.0
    assert m.tp + m.fn == 50
    assert m.fp + m.tn == 950
    assert m.total_cost is not None


def test_optimize_f_beta(sample_predictions):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)

    f1_opt = tuner.optimize_f_beta(beta=1.0)
    f2_opt = tuner.optimize_f_beta(beta=2.0)

    assert 0.0 < f1_opt.threshold < 1.0
    assert 0.0 < f2_opt.threshold < 1.0
    # F2 prioritizes recall, so threshold should be lower or equal to catch more fraud
    assert f2_opt.recall >= f1_opt.recall - 0.05


def test_optimize_cost(sample_predictions):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)

    opt_metrics, comparison = tuner.optimize_cost(cost_fn=1000.0, cost_fp=30.0)
    assert opt_metrics.total_cost <= comparison["default_0.5_total_cost"]
    assert comparison["savings_percentage"] >= 0.0


def test_target_recall(sample_predictions):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)

    m = tuner.target_recall(min_recall=0.80)
    assert m.recall >= 0.80


def test_multi_tier_policy(sample_predictions):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)

    policy = tuner.design_multi_tier_policy()
    assert 0.0 < policy.low_risk_threshold < policy.high_risk_threshold < 1.0

    # Test classifications
    tier, action = policy.classify_risk(policy.low_risk_threshold - 0.01)
    assert tier == "LOW"
    assert action == "AUTO_APPROVE"

    tier, action = policy.classify_risk(
        (policy.low_risk_threshold + policy.high_risk_threshold) / 2.0
    )
    assert tier == "MEDIUM"
    assert action == "MANUAL_REVIEW"

    tier, action = policy.classify_risk(policy.high_risk_threshold + 0.05)
    assert tier == "HIGH"
    assert action == "AUTO_FREEZE"


def test_export_config(sample_predictions, tmp_path):
    y_true, y_scores = sample_predictions
    tuner = ThresholdTuner(y_true, y_scores)
    policy = tuner.design_multi_tier_policy()
    out_file = tmp_path / "threshold_cfg.json"

    cfg = tuner.export_config(
        out_file,
        optimal_threshold=0.25,
        policy=policy,
        cost_matrix={"cost_fn": 1000.0, "cost_fp": 30.0},
    )
    assert out_file.exists()
    assert cfg["optimal_threshold"] == 0.25
    assert "multi_tier_policy" in cfg
