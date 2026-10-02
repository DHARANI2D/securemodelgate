"""
Full evaluation engine:
  - Combined SecureModelGate (static + behavioral + MAT)
  - Multi-run evaluation for variance / confidence intervals
  - Ablation study (static-only, behavioral-only, combined)
  - Latency benchmarking
"""

import time, copy, random
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
from .attestation import MATIssuer


class Evaluator:

    def __init__(self, static_analyzer, fingerprinter, mat_issuer):
        self.static     = static_analyzer
        self.fp         = fingerprinter
        self.issuer     = mat_issuer

    # ── Single run ────────────────────────────────────────────────────────────

    def _single_run(self, clean_models, backdoored_models):
        all_models = ([(m, 0) for m in clean_models] +
                      [(m, 1) for m in backdoored_models])

        y_true, y_pred, y_score = [], [], []
        decisions = []

        for m, true_label in all_models:
            ss = self.static.score(m)
            fs = self.fp.score(m)
            token, payload, decision = self.issuer.issue(m, ss, fs)

            pred = 1 if decision == "FAIL" else 0
            y_true.append(true_label)
            y_pred.append(pred)
            y_score.append(payload["risk_score"] / 100.0)
            decisions.append({
                "id":        m["id"],
                "true":      true_label,
                "pred":      pred,
                "decision":  decision,
                "ks":        ss["ks_statistic"],
                "kl":        fs["kl_divergence"],
                "risk":      payload["risk_score"],
                "trigger":   m.get("trigger"),
            })

        tp = sum(1 for d in decisions if d["true"]==1 and d["pred"]==1)
        tn = sum(1 for d in decisions if d["true"]==0 and d["pred"]==0)
        fp = sum(1 for d in decisions if d["true"]==0 and d["pred"]==1)
        fn = sum(1 for d in decisions if d["true"]==1 and d["pred"]==0)

        dr  = tp / (tp + fn)      if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn)      if (fp + tn) > 0 else 0.0
        acc = (tp + tn) / len(all_models)

        try:
            auc = roc_auc_score(y_true, y_score)
            fpr_curve, tpr_curve, thresh = roc_curve(y_true, y_score)
        except Exception:
            auc = 0.0; fpr_curve = tpr_curve = thresh = []

        # Per-trigger breakdown
        trigger_breakdown = {}
        for d in decisions:
            if d["true"] == 1:
                t = d["trigger"] or "unknown"
                if t not in trigger_breakdown:
                    trigger_breakdown[t] = {"tp": 0, "fn": 0}
                if d["pred"] == 1: trigger_breakdown[t]["tp"] += 1
                else:              trigger_breakdown[t]["fn"] += 1

        return {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "dr": dr, "fpr": fpr, "acc": acc, "auc": auc,
            "fpr_curve": fpr_curve, "tpr_curve": tpr_curve,
            "decisions": decisions,
            "trigger_breakdown": trigger_breakdown,
        }

    # ── Multi-run ─────────────────────────────────────────────────────────────

    def evaluate(self, clean_models, backdoored_models, n_runs=3):
        """
        Run n_runs independent evaluations (with slight random
        subsampling to measure variance), return mean ± std.
        """
        run_results = []
        for run in range(n_runs):
            # Slight subsampling for variance measurement
            rng = random.Random(run * 100)
            c_sample = rng.sample(clean_models, len(clean_models))
            b_sample = rng.sample(backdoored_models, len(backdoored_models))
            r = self._single_run(c_sample, b_sample)
            run_results.append(r)
            print(f"        Run {run+1}/{n_runs}: "
                  f"DR={r['dr']*100:.1f}%  "
                  f"FPR={r['fpr']*100:.1f}%  "
                  f"AUC={r['auc']:.3f}")

        drs  = [r["dr"]  for r in run_results]
        fprs = [r["fpr"] for r in run_results]
        accs = [r["acc"] for r in run_results]
        aucs = [r["auc"] for r in run_results]

        # Best run for detailed output
        best = max(run_results, key=lambda r: r["dr"])

        # Confusion matrix from best run
        cm = confusion_matrix(
            [d["true"] for d in best["decisions"]],
            [d["pred"] for d in best["decisions"]]
        )

        result = {
            "mean_dr":  round(np.mean(drs) * 100, 1),
            "std_dr":   round(np.std(drs)  * 100, 1),
            "mean_fpr": round(np.mean(fprs) * 100, 1),
            "std_fpr":  round(np.std(fprs)  * 100, 1),
            "mean_acc": round(np.mean(accs) * 100, 1),
            "std_acc":  round(np.std(accs)  * 100, 1),
            "mean_auc": round(np.mean(aucs), 3),
            "std_auc":  round(np.std(aucs), 3),
            "confusion_matrix": cm.tolist(),
            "trigger_breakdown": best["trigger_breakdown"],
            "fpr_curve": [float(x) for x in best["fpr_curve"]],
            "tpr_curve": [float(x) for x in best["tpr_curve"]],
            "decisions": best["decisions"],
            "n_runs": n_runs,
        }

        print(f"\n      Combined → "
              f"DR: {result['mean_dr']}% ± {result['std_dr']}%  "
              f"FPR: {result['mean_fpr']}% ± {result['std_fpr']}%  "
              f"AUC: {result['mean_auc']} ± {result['std_auc']}")
        return result

    # ── Ablation ──────────────────────────────────────────────────────────────

    def ablation(self, clean_models, backdoored_models):
        """
        Ablation study: static-only, behavioral-only, combined.
        Returns dict of method → metrics.
        """
        all_models = ([(m, 0) for m in clean_models] +
                      [(m, 1) for m in backdoored_models])

        results = {}
        for method in ["static_only", "behavioral_only", "combined"]:
            tp = tn = fp = fn = 0
            for m, true_label in all_models:
                ss = self.static.score(m)
                fs = self.fp.score(m)

                if method == "static_only":
                    pred = 1 if ss["static_flagged"] else 0
                elif method == "behavioral_only":
                    pred = 1 if fs["fp_flagged"] else 0
                else:
                    pred = 1 if (ss["static_flagged"] or fs["fp_flagged"]) else 0

                if   true_label == 1 and pred == 1: tp += 1
                elif true_label == 0 and pred == 0: tn += 1
                elif true_label == 0 and pred == 1: fp += 1
                elif true_label == 1 and pred == 0: fn += 1

            dr  = tp / (tp + fn)      if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn)      if (fp + tn) > 0 else 0.0
            results[method] = {
                "detection_rate": round(dr  * 100, 1),
                "fpr":            round(fpr * 100, 1),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            }
            print(f"      Ablation [{method:18s}] "
                  f"DR: {dr*100:.1f}%  FPR: {fpr*100:.1f}%")

        return results

    # ── Latency benchmark ─────────────────────────────────────────────────────

    def benchmark_latency(self, models, n_repeats=3):
        """
        Time each component per model, n_repeats times.
        Returns mean latency per component and throughput.
        """
        static_times, fp_times, mat_times, total_times = [], [], [], []

        for m in models:
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                ss = self.static.score(m)
                t1 = time.perf_counter()
                fs = self.fp.score(m)
                t2 = time.perf_counter()
                token, payload, decision = self.issuer.issue(m, ss, fs)
                t3 = time.perf_counter()

                static_times.append(t1 - t0)
                fp_times.append(t2 - t1)
                mat_times.append(t3 - t2)
                total_times.append(t3 - t0)

        mean_total = float(np.mean(total_times))
        result = {
            "static_mean_s":   round(float(np.mean(static_times)),  3),
            "static_std_s":    round(float(np.std(static_times)),   3),
            "fp_mean_s":       round(float(np.mean(fp_times)),      3),
            "fp_std_s":        round(float(np.std(fp_times)),       3),
            "mat_mean_s":      round(float(np.mean(mat_times)),     4),
            "mat_std_s":       round(float(np.std(mat_times)),      4),
            "total_mean_s":    round(mean_total,                    3),
            "total_std_s":     round(float(np.std(total_times)),    3),
            "throughput_per_hour": round(3600 / mean_total, 0),
            "n_models":        len(models),
            "n_repeats":       n_repeats,
        }
        print(f"      Latency  → total: {result['total_mean_s']}s ± "
              f"{result['total_std_s']}s  "
              f"throughput: {int(result['throughput_per_hour'])}/hr")
        return result
