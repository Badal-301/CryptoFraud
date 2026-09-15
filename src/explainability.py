"""SHAP Explainability Module for Crypto Fraud Detection.

Provides global and local transaction-level interpretability, natural language
compliance reason codes, and regulatory-ready visual artifacts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except (ImportError, OSError, Exception) as e:
    HAS_SHAP = False
    logger.warning(
        f"SHAP native DLL not loadable ({e}). Explainer will operate in robust surrogate attribution mode."
    )


class FraudExplainer:
    """Enterprise explainability engine for transaction risk scoring and AML compliance."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_sample: Optional[pd.DataFrame | np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        model : Any
            Trained model artifact (e.g. RandomForestClassifier, XGBClassifier).
        feature_names : List[str]
            Exact feature column names in model input order.
        background_sample : Optional[pd.DataFrame | np.ndarray]
            Reference background dataset for SHAP value computation (50-200 representative rows).
        """
        self.model = model
        self.feature_names = list(feature_names)
        self.background_sample = (
            pd.DataFrame(background_sample, columns=self.feature_names)
            if background_sample is not None and not isinstance(background_sample, pd.DataFrame)
            else background_sample
        )
        self.explainer: Optional[Any] = None
        self.expected_value: float = 0.5
        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialize SHAP TreeExplainer or fallback explainer."""
        if not HAS_SHAP:
            self._init_fallback()
            return

        try:
            # Tree-based fast exact SHAP computation
            if hasattr(self.model, "estimators_") or "XGB" in type(self.model).__name__:
                try:
                    self.explainer = shap.TreeExplainer(
                        self.model, feature_perturbation="tree_path_dependent"
                    )
                except Exception:
                    self.explainer = shap.TreeExplainer(self.model)

                raw_ev = self.explainer.expected_value
                if isinstance(raw_ev, (list, np.ndarray)):
                    raw_ev = float(raw_ev[1] if len(raw_ev) > 1 else raw_ev[0])
                else:
                    raw_ev = float(raw_ev)

                # Convert margin output (log-odds) to probability if outside [0, 1]
                if raw_ev < 0.0 or raw_ev > 1.0:
                    self.expected_value = float(1.0 / (1.0 + np.exp(-raw_ev)))
                else:
                    self.expected_value = raw_ev
            else:
                # Generic Explainer
                bg = (
                    self.background_sample.values
                    if self.background_sample is not None
                    else None
                )
                self.explainer = shap.Explainer(self.model.predict_proba, bg)
                self.expected_value = 0.5
        except Exception as e:
            logger.warning(f"Could not initialize native SHAP explainer ({e}). Falling back.")
            self._init_fallback()

    def _init_fallback(self) -> None:
        """Fallback explainer based on feature importances or surrogate perturbation."""
        self.explainer = None
        self.expected_value = 0.5

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP attributions with safety checks across SHAP versions."""
        if not HAS_SHAP or self.explainer is None:
            return self._compute_fallback_attributions(X)

        try:
            raw_shap = self.explainer.shap_values(X)
        except Exception as e:
            logger.warning(f"TreeExplainer.shap_values failed ({e}). Using robust surrogate attribution.")
            return self._compute_fallback_attributions(X)

        if isinstance(raw_shap, list):
            # Binary classification list of [neg_class_shap, pos_class_shap]
            values = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
        elif hasattr(raw_shap, "values"):
            # shap.Explanation object in newer SHAP releases
            values = raw_shap.values
            if values.ndim == 3:  # (samples, features, classes)
                values = values[:, :, 1]
        else:
            values = raw_shap

        return np.asarray(values)

    def _compute_fallback_attributions(self, X: pd.DataFrame) -> np.ndarray:
        """Heuristic feature attribution fallback when native SHAP is not installed."""
        # Use tree feature importances modulated by standardized feature deviation
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        X_mat = X.values
        # Center values around mean
        mean_v = np.mean(X_mat, axis=0) if len(X_mat) > 1 else np.zeros(X_mat.shape[1])
        std_v = np.std(X_mat, axis=0) + 1e-6 if len(X_mat) > 1 else np.ones(X_mat.shape[1])
        z_scores = (X_mat - mean_v) / std_v

        # Attributions scale with importance * deviation
        attributions = z_scores * importances
        return attributions

    def get_global_feature_importance(
        self, sample_data: pd.DataFrame, max_features: int = 20
    ) -> pd.DataFrame:
        """Calculate global mean absolute SHAP values across a representative data sample."""
        df_in = sample_data[self.feature_names]
        shap_vals = self._compute_shap_values(df_in)
        mean_abs = np.mean(np.abs(shap_vals), axis=0)

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        return importance_df.head(max_features)

    def explain_transaction(
        self,
        transaction_features: Union[Dict[str, float], pd.Series],
        max_reasons: int = 5,
    ) -> Dict[str, Any]:
        """Explain an individual transaction prediction with natural language AML reason codes.

        Parameters
        ----------
        transaction_features : Dict[str, float] or pd.Series
            Feature values for a single transaction. Missing values filled with 0.0.
        max_reasons : int
            Number of top positive/negative drivers to return.

        Returns
        -------
        Dict[str, Any]
            Structured explanation including risk drivers, mitigating factors,
            baseline expected probability, and human-readable compliance explanation.
        """
        row_dict = dict(transaction_features)
        feature_vector = [row_dict.get(f, 0.0) for f in self.feature_names]
        df_row = pd.DataFrame([feature_vector], columns=self.feature_names)

        shap_vals = self._compute_shap_values(df_row)[0]

        # Get model predicted probability
        if hasattr(self.model, "predict_proba"):
            fraud_prob = float(self.model.predict_proba(df_row)[0, 1])
        else:
            fraud_prob = float(self.model.predict(df_row)[0])

        indexed_shap = list(zip(self.feature_names, feature_vector, shap_vals))

        # Sort positive contributors (increasing illicit risk)
        risk_drivers = sorted(
            [item for item in indexed_shap if item[2] > 0],
            key=lambda x: x[2],
            reverse=True,
        )[:max_reasons]

        # Sort negative contributors (lowering illicit risk / evidence of licit activity)
        mitigating_factors = sorted(
            [item for item in indexed_shap if item[2] < 0],
            key=lambda x: x[2],
        )[:max_reasons]

        def _fmt(items: List[Tuple[str, float, float]], direction: str):
            return [
                {
                    "feature": f_name,
                    "feature_value": round(float(f_val), 4),
                    "shap_attribution": round(float(s_val), 4),
                    "impact": direction,
                }
                for f_name, f_val, s_val in items
            ]

        drivers_formatted = _fmt(risk_drivers, "INCREASES_RISK")
        mitigating_formatted = _fmt(mitigating_factors, "DECREASES_RISK")

        # Generate AML compliance natural language reason string
        if risk_drivers:
            top_reasons_str = ", ".join(
                [f"{f} (+{s:.3f}, val={v:.2f})" for f, v, s in risk_drivers[:3]]
            )
            reason_text = (
                f"Fraud probability {fraud_prob:.1%}. Primary illicit risk drivers: {top_reasons_str}."
            )
        else:
            reason_text = f"Fraud probability {fraud_prob:.1%}. Normal transaction activity observed."

        return {
            "fraud_probability": round(fraud_prob, 4),
            "baseline_expected_value": round(self.expected_value, 4),
            "top_risk_drivers": drivers_formatted,
            "top_mitigating_factors": mitigating_formatted,
            "compliance_reason_summary": reason_text,
        }

    def plot_summary(
        self,
        sample_data: pd.DataFrame,
        max_features: int = 15,
        plot_type: str = "dot",
        save_path: Optional[str | Path] = None,
    ) -> Optional[Any]:
        """Generate SHAP beeswarm or bar summary plot."""
        if not HAS_MATPLOTLIB or plt is None:
            logger.warning("Matplotlib is not available. Skipping summary plot generation.")
            return None
        df_in = sample_data[self.feature_names]
        shap_vals = self._compute_shap_values(df_in)

        plt.figure(figsize=(10, 6))

        if HAS_SHAP:
            shap.summary_plot(
                shap_vals,
                df_in,
                max_display=max_features,
                plot_type=plot_type,
                show=False,
            )
        else:
            # Fallback bar plot of top feature importance
            mean_abs = np.mean(np.abs(shap_vals), axis=0)
            top_indices = np.argsort(mean_abs)[::-1][:max_features]
            names = [self.feature_names[i] for i in top_indices]
            scores = mean_abs[top_indices]

            plt.barh(names[::-1], scores[::-1], color="#1f77b4")
            plt.title(f"Top {max_features} Global Feature Attributions", fontsize=12)
            plt.xlabel("Mean |Attribution|")

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300, bbox_inches="tight")

        fig = plt.gcf()
        return fig

    def plot_waterfall(
        self,
        transaction_features: Union[Dict[str, float], pd.Series],
        max_display: int = 10,
        save_path: Optional[str | Path] = None,
    ) -> Optional[Any]:
        """Plot local waterfall plot for a specific transaction."""
        if not HAS_MATPLOTLIB or plt is None:
            logger.warning("Matplotlib is not available. Skipping waterfall plot generation.")
            return None
        row_dict = dict(transaction_features)
        feature_vector = [row_dict.get(f, 0.0) for f in self.feature_names]
        df_row = pd.DataFrame([feature_vector], columns=self.feature_names)

        shap_vals = self._compute_shap_values(df_row)[0]

        plt.figure(figsize=(9, 5))

        if HAS_SHAP and hasattr(shap, "Explanation"):
            try:
                exp = shap.Explanation(
                    values=shap_vals,
                    base_values=self.expected_value,
                    data=df_row.values[0],
                    feature_names=self.feature_names,
                )
                shap.plots.waterfall(exp, max_display=max_display, show=False)
            except Exception:
                self._draw_fallback_waterfall(shap_vals, max_display)
        else:
            self._draw_fallback_waterfall(shap_vals, max_display)

        plt.tight_layout()
        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(p, dpi=300, bbox_inches="tight")

        fig = plt.gcf()
        return fig

    def _draw_fallback_waterfall(self, shap_vals: np.ndarray, max_display: int) -> None:
        """Horizontal bar breakdown for local attribution."""
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:max_display]
        names = [self.feature_names[i] for i in top_idx][::-1]
        vals = shap_vals[top_idx][::-1]
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in vals]

        plt.barh(names, vals, color=colors)
        plt.axvline(0, color="black", linestyle="--", alpha=0.6)
        plt.title("Local Transaction Attribution Breakdown", fontsize=12)
        plt.xlabel("SHAP Value (Red = Risk Driver, Green = Risk Reducer)")

    def save(self, filepath: str | Path) -> None:
        """Persist the explainer artifact to disk."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "background_sample": self.background_sample,
                "expected_value": self.expected_value,
            },
            p,
        )

    @classmethod
    def load(cls, filepath: str | Path) -> FraudExplainer:
        """Load a persisted explainer artifact."""
        data = joblib.load(filepath)
        return cls(
            model=data["model"],
            feature_names=data["feature_names"],
            background_sample=data.get("background_sample"),
        )
