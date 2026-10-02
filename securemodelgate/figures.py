"""Matplotlib figures for an `EvaluationReport` (securemodelgate.evaluation).

Kept separate from evaluation.py so `run_evaluation` itself has no
matplotlib/plotting concerns — plotting is presentation, not evaluation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless: this module never opens a window
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve

from .adaptive_evaluation import AdaptiveAttackReport
from .evaluation import EvaluationReport


def plot_roc_curve(report: EvaluationReport, out_path: Path) -> None:
    fpr, tpr, _ = roc_curve(report.envelope_score_labels, report.envelope_scores)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, marker=".", label=f"envelope score (AUROC={report.envelope_auroc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax.axvline(report.config.target_fpr, linestyle=":", color="red",
               label=f"target FPR={report.config.target_fpr}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Envelope-score ROC (engineering validation pilot)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_distributions(report: EvaluationReport, out_path: Path) -> None:
    labels = report.envelope_score_labels
    scores = report.envelope_scores
    benign = [s for s, y in zip(scores, labels) if y == 0]
    backdoored = [s for s, y in zip(scores, labels) if y == 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(benign, bins=15, alpha=0.6, label="benign derivatives (held-out)")
    ax.hist(backdoored, bins=15, alpha=0.6, label="backdoored derivatives")
    conformal_threshold = max(report.envelope_calibration_scores) if report.envelope_calibration_scores else None
    if conformal_threshold is not None:
        ax.axvline(conformal_threshold, linestyle="--", color="red", label="conformal threshold")
    ax.set_xlabel("envelope anomaly score  max(1 - min_CKA, JS)")
    ax.set_ylabel("count")
    ax.set_title("Envelope score distribution (engineering validation pilot)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_trigger_breakdown(report: EvaluationReport, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    names = [tb.trigger_type for tb in report.trigger_breakdown]
    tprs = [tb.tp / tb.n for tb in report.trigger_breakdown]
    asrs = [tb.mean_asr for tb in report.trigger_breakdown]

    x = range(len(names))
    width = 0.35
    ax.bar([i - width / 2 for i in x], tprs, width, label="detection TPR")
    ax.bar([i + width / 2 for i in x], asrs, width, label="mean attack success rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-trigger-type detection vs. attack success")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_adaptive_attack(report: AdaptiveAttackReport, out_path: Path) -> None:
    lambdas = [r.lam for r in report.results]
    detection = [r.detection_rate.point_estimate for r in report.results]
    asr = [r.mean_attack_success_rate for r in report.results]
    scores = [r.mean_envelope_score for r in report.results]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(lambdas, detection, marker="o", color="tab:red", label="detection rate")
    ax1.plot(lambdas, asr, marker="s", color="tab:orange", label="mean ASR retained")
    ax1.axhline(1.0, linestyle=":", color="gray", linewidth=0.8)
    ax1.set_xlabel("lambda (adaptive penalty weight)")
    ax1.set_ylabel("rate")
    ax1.set_ylim(-0.05, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(lambdas, scores, marker="^", color="tab:blue", label="mean envelope score")
    ax2.axhline(report.envelope_threshold, linestyle="--", color="tab:blue", linewidth=0.8,
                label="conformal threshold")
    ax2.set_ylabel("envelope score", color="tab:blue")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")
    ax1.set_title("Adaptive attacker (A4): detection vs. penalty weight")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_all_figures(report: EvaluationReport, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        out_dir / "roc_curve.png",
        out_dir / "score_distributions.png",
        out_dir / "trigger_breakdown.png",
    ]
    plot_roc_curve(report, paths[0])
    plot_score_distributions(report, paths[1])
    plot_trigger_breakdown(report, paths[2])
    return paths
