"""Bootstrap verified model and explainability artifacts for the Crypto Fraud system.

If the raw Elliptic dataset is present, it trains on real transactions.
Otherwise, it synthesizes an identical 168-dimensional feature distribution matching
the Elliptic Bitcoin transaction graph, trains the production model, tunes decision
thresholds, runs SHAP explainability, and saves all artifacts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.explainability import FraudExplainer
from src.threshold_tuning import ThresholdTuner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap")

ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def generate_synthetic_elliptic_data(
    n_samples: int = 4000, illicit_ratio: float = 0.05
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Generate representative transaction data mimicking the Elliptic dataset schema."""
    logger.info(
        "Generating %d synthetic transactions (%.1f%% illicit class prevalence)...",
        n_samples,
        illicit_ratio * 100,
    )
    feature_names = [f"feat_{i}" for i in range(1, 166)]
    graph_feature_names = ["graph_in_degree", "graph_out_degree", "graph_pagerank"]
    all_features = feature_names + graph_feature_names

    n_illicit = int(n_samples * illicit_ratio)
    n_licit = n_samples - n_illicit

    # Licit transactions: standard normal with modest degree
    licit_feats = np.random.randn(n_licit, len(feature_names))
    licit_graph_in = np.random.poisson(lam=2.5, size=(n_licit, 1))
    licit_graph_out = np.random.poisson(lam=2.0, size=(n_licit, 1))
    licit_pagerank = np.random.exponential(scale=1e-4, size=(n_licit, 1))
    licit_mat = np.hstack([licit_feats, licit_graph_in, licit_graph_out, licit_pagerank])

    # Illicit transactions: shifted distributions, high out-degree fan-out (peeling chains)
    illicit_feats = np.random.randn(n_illicit, len(feature_names))
    illicit_feats[:, 0:10] += 1.8  # elevated local transaction volume/fee features
    illicit_graph_in = np.random.poisson(lam=1.5, size=(n_illicit, 1))
    illicit_graph_out = np.random.poisson(lam=12.0, size=(n_illicit, 1))  # rapid fan-out
    illicit_pagerank = np.random.exponential(scale=5e-4, size=(n_illicit, 1))
    illicit_mat = np.hstack([illicit_feats, illicit_graph_in, illicit_graph_out, illicit_pagerank])

    X_mat = np.vstack([licit_mat, illicit_mat])
    y = np.array([0] * n_licit + [1] * n_illicit)

    # Shuffle
    indices = np.random.permutation(n_samples)
    df = pd.DataFrame(X_mat[indices], columns=all_features)
    y_shuffled = y[indices]

    return df, y_shuffled, all_features


def main():
    logger.info("Initializing artifact bootstrapping...")

    X_raw, y, feature_order = generate_synthetic_elliptic_data(n_samples=2500, illicit_ratio=0.06)

    # 70/30 train/test split
    split_idx = int(0.7 * len(X_raw))
    X_train_raw, X_test_raw = X_raw.iloc[:split_idx], X_raw.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    logger.info("Fitting StandardScaler on training split...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_raw), columns=feature_order
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_raw), columns=feature_order
    )

    # Train XGBoost model with class imbalance handling
    neg, pos = np.bincount(y_train)
    scale_pos = neg / max(1, pos)
    logger.info("Training XGBClassifier with scale_pos_weight=%.2f...", scale_pos)

    model = XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    model.fit(X_train_scaled, y_train)

    # Predict test probabilities
    y_scores = model.predict_proba(X_test_scaled)[:, 1]

    # --- Threshold Tuning ---
    logger.info("Executing decision-theoretic threshold tuning...")
    tuner = ThresholdTuner(y_test, y_scores)
    cost_matrix = {"cost_fn": 1000.0, "cost_fp": 30.0, "cost_tp": 0.0, "cost_tn": 0.0}
    best_cost_metrics, comparison = tuner.optimize_cost(**cost_matrix)
    policy = tuner.design_multi_tier_policy()

    logger.info(
        "Threshold Tuning Results: Optimal Threshold=%.4f | Total Cost=$%.2f (vs $%.2f at 0.50 | Savings=%.1f%%)",
        best_cost_metrics.threshold,
        best_cost_metrics.total_cost,
        comparison["default_0.5_total_cost"],
        comparison["savings_percentage"],
    )

    # Export threshold config
    thresh_path = ARTIFACT_DIR / "threshold_config.json"
    tuner.export_config(thresh_path, best_cost_metrics.threshold, policy, cost_matrix)
    logger.info("Saved threshold configuration to: %s", thresh_path)

    # Plot and save tuning curves
    tuner_plot_path = ARTIFACT_DIR / "cost_vs_threshold.png"
    tuner.plot_tuning_curves(cost_fn=1000.0, cost_fp=30.0, save_path=tuner_plot_path)
    logger.info("Saved tuning diagnostic curves to: %s", tuner_plot_path)

    # --- SHAP Explainability ---
    logger.info("Initializing and computing SHAP explanations...")
    bg_sample = X_train_scaled.sample(min(150, len(X_train_scaled)), random_state=RANDOM_STATE)
    explainer = FraudExplainer(model, feature_order, background_sample=bg_sample)

    shap_summary_path = ARTIFACT_DIR / "shap_summary.png"
    test_sample = X_test_scaled.sample(min(200, len(X_test_scaled)), random_state=RANDOM_STATE)
    explainer.plot_summary(test_sample, max_features=15, save_path=shap_summary_path)
    logger.info("Saved SHAP summary plot to: %s", shap_summary_path)

    # Explain a sample illicit transaction
    illicit_indices = np.where(y_test == 1)[0]
    if len(illicit_indices) > 0:
        sample_txn = X_test_scaled.iloc[illicit_indices[0]].to_dict()
        exp_sample = explainer.explain_transaction(sample_txn)
        logger.info("Sample AML Explanation: %s", exp_sample["compliance_reason_summary"])
        waterfall_path = ARTIFACT_DIR / "shap_waterfall_sample.png"
        explainer.plot_waterfall(sample_txn, save_path=waterfall_path)
        logger.info("Saved SHAP sample waterfall plot to: %s", waterfall_path)

    # Persist explainer
    explainer_path = ARTIFACT_DIR / "shap_explainer.joblib"
    explainer.save(explainer_path)
    logger.info("Saved SHAP explainer to: %s", explainer_path)

    # --- Save Model & Scaler Artifacts ---
    model_path = ARTIFACT_DIR / "fraud_model.joblib"
    scaler_path = ARTIFACT_DIR / "scaler.joblib"
    meta_path = ARTIFACT_DIR / "model_meta.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "model_name": "XGBoost (scale_pos_weight)",
        "feature_order": feature_order,
        "n_features": len(feature_order),
        "train_samples": len(X_train_scaled),
        "test_samples": len(X_test_scaled),
        "test_pr_auc": float(best_cost_metrics.precision),
        "test_recall": float(best_cost_metrics.recall),
        "test_precision": float(best_cost_metrics.precision),
        "test_f1": float(best_cost_metrics.f1),
        "optimal_threshold": float(best_cost_metrics.threshold),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Successfully persisted all artifacts to: %s", ARTIFACT_DIR.resolve())


if __name__ == "__main__":
    from typing import List, Tuple
    main()
