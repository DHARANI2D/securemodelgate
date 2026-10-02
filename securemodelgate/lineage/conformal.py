"""
Split-Conformal Thresholds (Section 3.3)
==========================================
Replaces the hand-picked τ = 0.02 / 0.05 the reviewers flagged with two
split-conformal tests, each calibrated on its own population:

  - Envelope test (benign-derivative null): calibrate the D_b/S_w
    envelope score on n benign derivatives (fine-tuned/pruned/INT8).
    τ = the ceil((n+1)(1-α))-th smallest calibration score. Escalate to
    Tier 2 if a candidate's score exceeds τ. Under exchangeability of
    the candidate with the calibration set,
        P(benign derivative escalated) <= α.
    With n=19, α=0.05: ceil(20 * 0.95) = 19 -> τ is the MAXIMUM
    calibration score (matches the "calibrate against the max clean
    score" advice from the earlier review, now with a stated bound).

  - Lineage test (independent-model null): calibrate S_w on models with
    NO true lineage relationship to the base, then verify lineage if a
    candidate's S_w exceeds their (1-α_L) conformal quantile.

Both are instances of the same order-statistic rule
(`conformal_quantile`), just calibrated on different populations and
applied with the comparison flipped depending on which direction is
"suspicious" for that score.

A 1% false-block claim needs n >= 99 calibration models (see
`min_calibration_size`), which is why the abstract states α next to the
n it was calibrated on rather than quoting a bare percentage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def min_calibration_size(alpha: float) -> int:
    """Smallest n such that a split-conformal test at level alpha can, in
    principle, express an error bound of alpha (needs (n+1)*alpha >= 1)."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    return max(1, math.ceil(1.0 / alpha) - 1)


def conformal_quantile(calibration_scores, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest score in `calibration_scores`.

    This is the standard split-conformal threshold: for a test point
    exchangeable with the calibration set, P(test score > threshold) <= alpha.
    Saturates at the maximum calibration score when
    ceil((n+1)(1-alpha)) >= n (e.g. n=19, alpha=0.05 -> index 19 of 19,
    i.e. the max).
    """
    scores = np.asarray(calibration_scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        raise ValueError("conformal_quantile: empty calibration set")
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if not np.all(np.isfinite(scores)):
        # np.sort places NaN at the END of the array regardless of its
        # true value, so a single NaN calibration score can silently
        # become the reported threshold at a high rank (e.g. n=5,
        # alpha=0.05 picks the very last sorted position). Every
        # subsequent `candidate_score > tau` comparison against a NaN
        # threshold is then False in both numpy and plain Python,
        # whichever direction that fails in (never escalating a benign-
        # envelope test is a silent security gap, not just a stats bug).
        # Fail loudly instead of shipping a threshold nothing can exceed.
        raise ValueError(
            f"conformal_quantile: calibration_scores contains {np.sum(~np.isfinite(scores))} "
            f"non-finite value(s) (NaN/inf); a corrupted calibration set must not silently "
            f"produce an unreachable threshold"
        )
    scores = np.sort(scores)

    rank = math.ceil((n + 1) * (1 - alpha))
    idx = min(rank, n) - 1   # 1-indexed rank -> 0-indexed array position, clamped to n
    idx = max(idx, 0)
    return float(scores[idx])


@dataclass
class ConformalVerdict:
    threshold: float
    alpha: float
    n_calibration: int
    score: float
    exceeds_threshold: bool
    warning: str = ""
    failure_reason: str = ""   # set when the verdict fails for a reason other than score <= threshold


def envelope_test(candidate_score: float, benign_calibration_scores, alpha: float = 0.05) -> ConformalVerdict:
    """Escalate to Tier 2 if candidate_score exceeds the benign-derivative
    conformal threshold. `candidate_score` should combine D_b/S_w such
    that HIGHER means more anomalous (e.g. 1 - min_cka, or js_divergence).
    """
    if not math.isfinite(candidate_score):
        # A NaN candidate_score silently reads as "not anomalous" (every
        # comparison against NaN is False) -- exactly the fail-open
        # direction a broken/degenerate behavioral score must NOT get.
        raise ValueError(f"envelope_test: candidate_score is non-finite ({candidate_score!r})")
    n = len(benign_calibration_scores)
    tau = conformal_quantile(benign_calibration_scores, alpha)
    warning = ""
    needed = min_calibration_size(alpha)
    if n < needed:
        warning = (f"n={n} calibration models cannot support alpha={alpha} "
                    f"(needs n>={needed} for a meaningful bound); treat the "
                    f"false-block rate as illustrative, not a guarantee.")
    return ConformalVerdict(
        threshold=tau, alpha=alpha, n_calibration=n,
        score=candidate_score, exceeds_threshold=candidate_score > tau,
        warning=warning,
    )


def lineage_test(candidate_score: float, independent_model_scores, alpha_l: float = 0.05) -> ConformalVerdict:
    """Verify lineage if candidate S_w exceeds the (1-alpha_l) conformal
    quantile of S_w computed on models with NO true lineage relationship
    to the declared base (the null: "candidate is unrelated to base").
    """
    if not math.isfinite(candidate_score):
        raise ValueError(f"lineage_test: candidate_score is non-finite ({candidate_score!r})")
    n = len(independent_model_scores)
    tau = conformal_quantile(independent_model_scores, alpha_l)
    warning = ""
    needed = min_calibration_size(alpha_l)
    if n < needed:
        warning = (f"n={n} independent models cannot support alpha_L={alpha_l} "
                    f"(needs n>={needed}); treat the lineage false-accept "
                    f"rate as illustrative, not a guarantee.")
    return ConformalVerdict(
        threshold=tau, alpha=alpha_l, n_calibration=n,
        score=candidate_score, exceeds_threshold=candidate_score > tau,
        warning=warning,
    )
