"""Threshold Tuning Engine for Crypto Fraud Detection.

This module provides decision-theoretic and metric-driven threshold optimization
specifically designed for highly imbalanced fraud detection scenarios (e.g. Elliptic
Bitcoin dataset with ~2% illicit rate).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


@dataclass
class ThresholdMetrics:
    """Evaluation metrics at a specific decision threshold."""
    threshold: float
    precision: float
    recall: float
    f1: float
    f2: float
    tp: int
    fp: int
    fn: int
    tn: int
    total_cost: Optional[float] = None
    alert_rate: float = 0.0


@dataclass
class MultiTierPolicy:
    """Three-tier risk decision policy for transaction routing."""
    low_risk_threshold: float  # p < low_risk_threshold -> AUTO_APPROVE
    high_risk_threshold: float  # p >= high_risk_threshold -> AUTO_FREEZE
    # low_risk_threshold <= p < high_risk_threshold -> MANUAL_REVIEW
    tier_labels: Dict[str, str] = None

    def __post_init__(self):
        if self.tier_labels is None:
            self.tier_labels = {
                "LOW": "AUTO_APPROVE",
                "MEDIUM": "MANUAL_REVIEW",
                "HIGH": "AUTO_FREEZE",
            }

    def classify_risk(self, probability: float) -> Tuple[str, str]:
        """Classify fraud probability into risk tier and operational action."""
        if probability < self.low_risk_threshold:
            return "LOW", self.tier_labels["LOW"]
        elif probability < self.high_risk_threshold:
            return "MEDIUM", self.tier_labels["MEDIUM"]
        else:
            return "HIGH", self.tier_labels["HIGH"]


class ThresholdTuner:
    """Systematic threshold tuning across financial cost and classification metrics."""

    def __init__(self, y_true: np.ndarray, y_scores: np.ndarray):
        """
        Parameters
        ----------
        y_true : np.ndarray
            Binary ground truth labels (0 for licit, 1 for illicit).
        y_scores : np.ndarray
            Continuous predicted probabilities for the positive/illicit class in [0, 1].
        """
        self.y_true = np.asarray(y_true).astype(int)
        self.y_scores = np.asarray(y_scores).astype(float)

        if len(self.y_true) != len(self.y_scores):
            raise ValueError("y_true and y_scores must have identical length.")

        self.precisions, self.recalls, self.pr_thresholds = precision_recall_curve(
            self.y_true, self.y_scores
        )
        # precision_recall_curve appends 1.0 to precision and 0.0 to recall with no threshold
        self.candidate_thresholds = np.clip(self.pr_thresholds, 1e-4, 1.0 - 1e-4)

    def evaluate_threshold(
        self,
        threshold: float,
        cost_fn: float = 1000.0,
        cost_fp: float = 30.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
    ) -> ThresholdMetrics:
        """Evaluate detailed metrics and financial cost at an exact threshold."""
        y_pred = (self.y_scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()

        prec = precision_score(self.y_true, y_pred, zero_division=0)
        rec = recall_score(self.y_true, y_pred, zero_division=0)
        f1 = fbeta_score(self.y_true, y_pred, beta=1.0, zero_division=0)
        f2 = fbeta_score(self.y_true, y_pred, beta=2.0, zero_division=0)

        total_cost = (
            (cost_fn * fn) + (cost_fp * fp) + (cost_tp * tp) + (cost_tn * tn)
        )
        alert_rate = float((tp + fp) / len(self.y_true))

        return ThresholdMetrics(
            threshold=float(threshold),
            precision=float(prec),
            recall=float(rec),
            f1=float(f1),
            f2=float(f2),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            tn=int(tn),
            total_cost=float(total_cost),
            alert_rate=alert_rate,
        )

    def optimize_f_beta(self, beta: float = 1.0) -> ThresholdMetrics:
        """Find the decision threshold that maximizes the F-beta score.

        A beta of 1.0 weights precision and recall equally.
        A beta of 2.0 places twice as much weight on recall (catching fraud).
        """
        best_f = -1.0
        best_thresh = 0.5

        # Sample grid for stability
        threshold_grid = np.unique(
            np.concatenate([np.linspace(0.01, 0.99, 200), self.candidate_thresholds])
        )

        for t in threshold_grid:
            y_pred = (self.y_scores >= t).astype(int)
            score = fbeta_score(self.y_true, y_pred, beta=beta, zero_division=0)
            if score > best_f:
                best_f = score
                best_thresh = t

        return self.evaluate_threshold(best_thresh)

    def optimize_cost(
        self,
        cost_fn: float = 1000.0,
        cost_fp: float = 30.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
    ) -> Tuple[ThresholdMetrics, Dict[str, float]]:
        """Find the threshold that minimizes total financial/regulatory expected loss.

        Parameters
        ----------
        cost_fn : float
            Cost of a False Negative (missed fraud, regulatory fine, lost funds). Default: $1000.
        cost_fp : float
            Cost of a False Positive (analyst review time, customer friction). Default: $30.
        cost_tp : float
            Cost of investigating a true positive. Default: $0.
        cost_tn : float
            Cost of true negative. Default: $0.

        Returns
        -------
        best_metrics : ThresholdMetrics
            Metrics at the cost-optimal threshold.
        comparison : Dict[str, float]
            Financial comparison with default threshold 0.5 and naive baselines.
        """
        threshold_grid = np.unique(
            np.concatenate([np.linspace(0.01, 0.99, 200), self.candidate_thresholds])
        )

        best_cost = float("inf")
        best_thresh = 0.5

        for t in threshold_grid:
            metrics = self.evaluate_threshold(
                t, cost_fn=cost_fn, cost_fp=cost_fp, cost_tp=cost_tp, cost_tn=cost_tn
            )
            if metrics.total_cost < best_cost:
                best_cost = metrics.total_cost
                best_thresh = t

        best_metrics = self.evaluate_threshold(
            best_thresh, cost_fn=cost_fn, cost_fp=cost_fp, cost_tp=cost_tp, cost_tn=cost_tn
        )
        default_metrics = self.evaluate_threshold(
            0.5, cost_fn=cost_fn, cost_fp=cost_fp, cost_tp=cost_tp, cost_tn=cost_tn
        )

        cost_savings = default_metrics.total_cost - best_metrics.total_cost
        cost_savings_pct = (
            (cost_savings / default_metrics.total_cost * 100)
            if default_metrics.total_cost > 0
            else 0.0
        )

        comparison = {
            "optimal_threshold": best_thresh,
            "optimal_total_cost": best_metrics.total_cost,
            "default_0.5_total_cost": default_metrics.total_cost,
            "financial_savings": cost_savings,
            "savings_percentage": cost_savings_pct,
        }

        return best_metrics, comparison

    def target_recall(self, min_recall: float = 0.85) -> ThresholdMetrics:
        """Find the highest threshold that maintains at least `min_recall`."""
        valid_indices = np.where(self.recalls[:-1] >= min_recall)[0]
        if len(valid_indices) == 0:
            thresh = float(self.candidate_thresholds[0])
        else:
            thresh = float(np.max(self.candidate_thresholds[valid_indices]))

        return self.evaluate_threshold(thresh)

    def design_multi_tier_policy(
        self,
        low_risk_quantile: float = 0.15,
        target_high_risk_precision: float = 0.85,
    ) -> MultiTierPolicy:
        """Design a 3-tier operational alert routing policy.

        - LOW RISK: Auto-cleared transactions. Threshold chosen where false negative risk is negligible.
        - MEDIUM RISK: Routed to AML investigator queue.
        - HIGH RISK: Auto-freeze transaction or file immediate suspicious activity report.
        """
        t_low = float(np.clip(np.quantile(self.y_scores, low_risk_quantile), 0.05, 0.25))

        high_prec_indices = np.where(self.precisions[:-1] >= target_high_risk_precision)[0]
        if len(high_prec_indices) > 0:
            t_high = float(self.candidate_thresholds[high_prec_indices[0]])
        else:
            t_high = 0.75

        if t_low >= t_high:
            t_low = t_high / 2.0

        return MultiTierPolicy(
            low_risk_threshold=round(t_low, 4),
            high_risk_threshold=round(t_high, 4),
        )

    def plot_tuning_curves(
        self,
        cost_fn: float = 1000.0,
        cost_fp: float = 30.0,
        save_path: Optional[str | Path] = None,
    ) -> Optional[Any]:
        """Plot dual diagnostic curves: Precision/Recall/F1 and Financial Cost vs Threshold."""
        if not HAS_MATPLOTLIB or plt is None:
            return None
        grid = np.linspace(0.01, 0.99, 100)
        precs, recs, f1s, costs = [], [], [], []

        for t in grid:
            m = self.evaluate_threshold(t, cost_fn=cost_fn, cost_fp=cost_fp)
            precs.append(m.precision)
            recs.append(m.recall)
            f1s.append(m.f1)
            costs.append(m.total_cost)

        opt_cost_metrics, _ = self.optimize_cost(cost_fn=cost_fn, cost_fp=cost_fp)
        opt_f1_metrics = self.optimize_f_beta(beta=1.0)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Metric tradeoffs
        ax1.plot(grid, precs, label="Precision", color="#1f77b4", lw=2)
        ax1.plot(grid, recs, label="Recall", color="#ff7f0e", lw=2)
        ax1.plot(grid, f1s, label="F1-Score", color="#2ca02c", lw=2)
        ax1.axvline(
            opt_f1_metrics.threshold,
            color="#2ca02c",
            linestyle="--",
            label=f"Max F1 ({opt_f1_metrics.threshold:.2f})",
        )
        ax1.set_title("Classification Metrics vs Decision Threshold", fontsize=12)
        ax1.set_xlabel("Decision Threshold")
        ax1.set_ylabel("Score")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3)

        # Right: Financial Cost
        ax2.plot(grid, costs, label=f"Total Cost ($FN={cost_fn}, $FP={cost_fp})", color="#d62728", lw=2)
        ax2.axvline(
            opt_cost_metrics.threshold,
            color="#d62728",
            linestyle="--",
            label=f"Min Cost ({opt_cost_metrics.threshold:.2f})",
        )
        ax2.axvline(0.5, color="gray", linestyle=":", label="Default (0.50)")
        ax2.set_title("Expected Financial Cost vs Decision Threshold", fontsize=12)
        ax2.set_xlabel("Decision Threshold")
        ax2.set_ylabel("Total Loss ($)")
        ax2.set_xlim(0, 1)
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300)

        return fig

    def export_config(
        self,
        output_filepath: str | Path,
        optimal_threshold: float,
        policy: MultiTierPolicy,
        cost_matrix: Dict[str, float],
    ) -> Dict[str, Any]:
        """Export threshold and multi-tier configuration to JSON for production inference."""
        metrics = self.evaluate_threshold(
            optimal_threshold,
            cost_fn=cost_matrix.get("cost_fn", 1000.0),
            cost_fp=cost_matrix.get("cost_fp", 30.0),
        )

        config = {
            "optimal_threshold": round(optimal_threshold, 4),
            "multi_tier_policy": {
                "low_risk_threshold": policy.low_risk_threshold,
                "high_risk_threshold": policy.high_risk_threshold,
                "tiers": policy.tier_labels,
            },
            "metrics_at_optimal_threshold": asdict(metrics),
            "cost_matrix": cost_matrix,
        }

        p = Path(output_filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(config, f, indent=2)

        return config
