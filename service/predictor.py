"""FraudPredictor runtime engine for model inference, threshold evaluation, and SHAP explanations."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from service.schemas import FeatureAttribution, RiskTier
from src.explainability import FraudExplainer
from src.threshold_tuning import MultiTierPolicy

logger = logging.getLogger(__name__)


class FraudPredictor:
    """Production predictor wrapper encapsulating model, scaler, threshold policy, and SHAP."""

    def __init__(self, artifacts_dir: Path | str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.model: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.meta: Dict[str, Any] = {}
        self.feature_order: List[str] = []
        self.threshold: float = 0.35
        self.policy: MultiTierPolicy = MultiTierPolicy(0.15, 0.65)
        self.explainer: Optional[FraudExplainer] = None
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load artifacts from disk into memory."""
        model_path = self.artifacts_dir / "fraud_model.joblib"
        scaler_path = self.artifacts_dir / "scaler.joblib"
        meta_path = self.artifacts_dir / "model_meta.json"
        thresh_path = self.artifacts_dir / "threshold_config.json"
        explainer_path = self.artifacts_dir / "shap_explainer.joblib"

        if not model_path.exists() or not scaler_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Missing model artifacts in {self.artifacts_dir.resolve()}. "
                "Ensure bootstrap_artifacts.py or the training pipeline has executed."
            )

        logger.info("Loading model artifacts from %s", self.artifacts_dir)
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.feature_order = self.meta.get("feature_order", [])

        # Load threshold config if available
        if thresh_path.exists():
            with open(thresh_path, "r") as f:
                thresh_data = json.load(f)
                self.threshold = float(thresh_data.get("optimal_threshold", 0.35))
                tier_cfg = thresh_data.get("multi_tier_policy", {})
                self.policy = MultiTierPolicy(
                    low_risk_threshold=float(tier_cfg.get("low_risk_threshold", 0.15)),
                    high_risk_threshold=float(tier_cfg.get("high_risk_threshold", 0.65)),
                )
        else:
            self.threshold = 0.35
            self.policy = MultiTierPolicy(0.15, 0.65)

        # Load or initialize SHAP explainer
        if explainer_path.exists():
            try:
                self.explainer = FraudExplainer.load(explainer_path)
            except Exception as e:
                logger.warning(f"Failed to load cached explainer ({e}). Initializing anew.")
                self.explainer = FraudExplainer(self.model, self.feature_order)
        else:
            self.explainer = FraudExplainer(self.model, self.feature_order)

        self.is_loaded = True
        logger.info(
            "FraudPredictor ready. Model: %s | Features: %d | Decision Threshold: %.3f",
            self.meta.get("model_name", "Unknown"),
            len(self.feature_order),
            self.threshold,
        )

    def _prepare_dataframe(self, dict_list: List[Dict[str, float]]) -> pd.DataFrame:
        """Align input dictionary records to model feature ordering and fill missing with 0.0."""
        # Fast aligned array construction
        rows = []
        for d in dict_list:
            rows.append([d.get(f, 0.0) for f in self.feature_order])
        return pd.DataFrame(rows, columns=self.feature_order)

    def predict_single(
        self, features: Dict[str, float]
    ) -> Tuple[float, bool, RiskTier, str]:
        """Score a single transaction.

        Returns (fraud_probability, is_illicit, risk_tier, action)
        """
        if not self.is_loaded:
            raise RuntimeError("FraudPredictor is not loaded.")

        df = self._prepare_dataframe([features])
        scaled = self.scaler.transform(df)

        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(scaled)[0, 1])
        else:
            raw = -self.model.score_samples(scaled)[0]
            prob = float(np.clip(raw, 0.0, 1.0))

        is_illicit = bool(prob >= self.threshold)
        tier_str, action = self.policy.classify_risk(prob)
        return prob, is_illicit, RiskTier(tier_str), action

    def predict_batch(
        self, batch_features: List[Dict[str, float]]
    ) -> List[Tuple[float, bool, RiskTier, str]]:
        """Vectorized batch scoring for high throughput."""
        if not self.is_loaded:
            raise RuntimeError("FraudPredictor is not loaded.")

        df = self._prepare_dataframe(batch_features)
        scaled = self.scaler.transform(df)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled)[:, 1]
        else:
            raw = -self.model.score_samples(scaled)
            probs = np.clip(raw, 0.0, 1.0)

        results = []
        for p in probs:
            p_val = float(p)
            is_ill = bool(p_val >= self.threshold)
            t_str, act = self.policy.classify_risk(p_val)
            results.append((p_val, is_ill, RiskTier(t_str), act))

        return results

    def explain_single(
        self, features: Dict[str, float], max_reasons: int = 5
    ) -> Dict[str, Any]:
        """Compute SHAP feature attributions and AML compliance reason codes."""
        if not self.is_loaded or self.explainer is None:
            raise RuntimeError("FraudPredictor or explainer is not loaded.")

        df = self._prepare_dataframe([features])
        scaled = self.scaler.transform(df)
        scaled_dict = dict(zip(self.feature_order, scaled[0]))

        explanation = self.explainer.explain_transaction(
            scaled_dict, max_reasons=max_reasons
        )
        tier_str, _ = self.policy.classify_risk(explanation["fraud_probability"])
        explanation["risk_tier"] = RiskTier(tier_str)
        return explanation

    def update_thresholds(
        self,
        threshold: Optional[float] = None,
        low_risk_threshold: Optional[float] = None,
        high_risk_threshold: Optional[float] = None,
    ) -> None:
        """Dynamically adjust decision thresholds at runtime."""
        if threshold is not None:
            self.threshold = round(float(threshold), 4)

        current_low = self.policy.low_risk_threshold
        current_high = self.policy.high_risk_threshold

        new_low = low_risk_threshold if low_risk_threshold is not None else current_low
        new_high = high_risk_threshold if high_risk_threshold is not None else current_high

        if new_low >= new_high:
            raise ValueError("high_risk_threshold must be strictly greater than low_risk_threshold")

        self.policy = MultiTierPolicy(
            low_risk_threshold=round(float(new_low), 4),
            high_risk_threshold=round(float(new_high), 4),
        )
