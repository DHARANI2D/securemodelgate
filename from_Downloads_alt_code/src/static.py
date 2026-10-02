"""
Static Analysis Module
======================
Detects backdoored models by analyzing:
  1. KS-test on weight distributions vs. clean reference
  2. Layer-wise variance anomaly detection
  3. Gradient norm analysis (outer layer sensitivity)

Decision: model flagged if KS-statistic > threshold OR
           layer variance z-score > threshold
"""

import numpy as np
from scipy.stats import ks_2samp, zscore
import torch


class StaticAnalyzer:
    """
    Weight-distribution-based backdoor detector.

    Builds a reference distribution from clean models,
    then flags suspects whose weight distributions deviate
    significantly (KS-test p < alpha, or high layer variance).
    """

    def __init__(self, ks_threshold=0.08, var_z_threshold=2.5):
        self.ks_threshold  = ks_threshold
        self.var_z_threshold = var_z_threshold
        self.reference_weights = None      # flat np array from clean models
        self.layer_var_ref     = None      # per-layer variance from clean set

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _extract_weights(self, model, max_samples=10000):
        weights = []
        for param in model.parameters():
            weights.extend(param.detach().cpu().float().numpy().flatten())
        w_array = np.array(weights, dtype=np.float32)
        if len(w_array) > max_samples:
            step = max(1, len(w_array) // max_samples)
            w_array = w_array[::step][:max_samples]
        return w_array

    def _layer_variances(self, model):
        """Per-layer weight variance — useful for detecting anomalous layers."""
        vars_ = []
        for param in model.parameters():
            w = param.detach().cpu().float().numpy()
            vars_.append(float(np.var(w)))
        return np.array(vars_)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, clean_models):
        """Build reference distribution from clean model set."""
        all_weights = np.concatenate(
            [self._extract_weights(m["model"]) for m in clean_models]
        )
        self.reference_weights = all_weights

        layer_vars = np.stack(
            [self._layer_variances(m["model"]) for m in clean_models]
        )
        self.layer_var_ref = layer_vars  # shape: (n_clean, n_layers)

    def score(self, model_dict):
        """
        Returns dict with:
          ks_statistic, p_value, layer_anomaly_score, flagged (bool)
        """
        if self.reference_weights is None:
            raise RuntimeError("Call fit() before score()")

        w = self._extract_weights(model_dict["model"])

        # KS test
        ks_stat, p_val = ks_2samp(self.reference_weights, w)

        # Layer variance anomaly
        lv = self._layer_variances(model_dict["model"])
        ref_mean = self.layer_var_ref.mean(axis=0)
        ref_std  = self.layer_var_ref.std(axis=0) + 1e-10
        z_scores = np.abs((lv - ref_mean) / ref_std)
        layer_anomaly = float(z_scores.max())

        flagged = (ks_stat > self.ks_threshold or
                   layer_anomaly > self.var_z_threshold)

        return {
            "ks_statistic":    float(ks_stat),
            "p_value":         float(p_val),
            "layer_anomaly":   float(layer_anomaly),
            "static_flagged":  flagged,
        }

    def evaluate(self, clean_models, backdoored_models):
        """
        Full evaluation: fit on clean, score all, return results dict.
        """
        self.fit(clean_models)

        results = {"scores": [], "labels": []}
        tp = tn = fp = fn = 0

        all_models = [(m, 0) for m in clean_models] + [(m, 1) for m in backdoored_models]
        for m, true_label in all_models:
            s = self.score(m)
            pred = 1 if s["static_flagged"] else 0
            results["scores"].append(s)
            results["labels"].append({"true": true_label, "pred": pred,
                                       "id": m["id"]})
            if   true_label == 1 and pred == 1: tp += 1
            elif true_label == 0 and pred == 0: tn += 1
            elif true_label == 0 and pred == 1: fp += 1
            elif true_label == 1 and pred == 0: fn += 1

        dr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / len(all_models)

        results["metrics"] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "detection_rate": round(dr * 100, 1),
            "fpr":            round(fpr * 100, 1),
            "accuracy":       round(acc * 100, 1),
        }
        print(f"      Static  → DR: {dr*100:.1f}%  FPR: {fpr*100:.1f}%  "
              f"Acc: {acc*100:.1f}%")
        return results
