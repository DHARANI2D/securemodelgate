"""Statistical reporting helpers shared by the evaluation harness.

The paper (`paper/techcon2027_revision/02_paper.md`, Section 4) reports
Clopper-Pearson 95% CIs on detection/false-block rates and an AUROC/TPR-
at-fixed-FPR for the envelope score — this module is the one place those
are computed, so `securemodelgate/evaluation.py` and its tests share a
single, checked implementation instead of each re-deriving the formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class Proportion:
    """A count out of n, with its exact (Clopper-Pearson) 95%-style CI."""
    successes: int
    n: int
    alpha: float
    point_estimate: float
    ci_low: float
    ci_high: float

    def __str__(self) -> str:
        return (f"{self.successes}/{self.n} = {self.point_estimate:.1%} "
                f"({(1 - self.alpha):.0%} CI {self.ci_low:.1%}-{self.ci_high:.1%})")


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> Proportion:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Matches the CIs quoted throughout the paper (e.g. "3/20 = 15% (95%
    CI 3-38%)") rather than a normal-approximation interval, which is
    unreliable at these small sample sizes and can misbehave at 0/n or
    n/n.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not (0 <= successes <= n):
        raise ValueError("successes must be in [0, n]")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")

    if successes == 0:
        low = 0.0
    else:
        low = scipy_stats.beta.ppf(alpha / 2, successes, n - successes + 1)
    if successes == n:
        high = 1.0
    else:
        high = scipy_stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)

    return Proportion(
        successes=successes, n=n, alpha=alpha,
        point_estimate=successes / n, ci_low=float(low), ci_high=float(high),
    )


def auroc(labels, scores) -> float:
    """Area under the ROC curve. `labels`: 1 = positive class (e.g.
    "backdoored"/"anomalous"), 0 = negative. `scores`: higher = more
    likely positive."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        raise ValueError("auroc needs both classes represented in `labels`")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels, scores, target_fpr: float = 0.05) -> float:
    """TPR at the operating point whose FPR is closest to (but not
    exceeding, where possible) `target_fpr`, linearly interpolated
    between the two bracketing ROC points sklearn returns.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if len(np.unique(labels)) < 2:
        raise ValueError("tpr_at_fpr needs both classes represented in `labels`")

    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.interp(target_fpr, fpr, tpr))
