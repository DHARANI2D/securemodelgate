"""
Risk-Tiered Admission (Section 3.5)
=====================================
    Tier 0 — trusted signed publisher, unmodified bytes -> admit
             (signature check only, no lineage/behavioral scoring needed).
    Tier 1 — OMS-signed declared base, derivative -> lineage + envelope
             test.
               pass both            -> admit, with attestation
               fail envelope only   -> escalate to Tier 2
               fail lineage         -> block (candidate is not a
                                        descendant of the base it claims)
    Tier 2 — unsigned, lineage unverified, or escalated from Tier 1 ->
             deep scan (pickle/code scanners, then MM-BD/BAIT-class
             detector), human review on a positive result.

This module only encodes the routing/verdict logic; it takes already-
computed conformal verdicts (conformal.py) and a trust-tier flag as
input, so it has no torch/model dependency and is fully unit-testable.

WEBHOOK INTEGRATION NOTE (a real gap this repo cannot close on its own):
a Kubernetes ValidatingAdmissionWebhook call is synchronous with a bounded
timeout (a handful of seconds by default), but deep scan is explicitly
"too slow ... to run on every import" (Section 1) -- it will not finish
inside that window. That means VERDICT_ESCALATE (deep_scan_flagged=None,
i.e. no deep-scan result exists yet) cannot mean "wait and see" at the
webhook layer; the integration MUST map it to `allowed: false` (deny) at
the synchronous admission decision, with deep scan running out-of-band
and a later, separate re-submission or explicit approval carrying the
resolved VERDICT_ADMIT/VERDICT_BLOCK once it completes. A naive
integration that treats ESCALATE as "not BLOCK, so let it through" would
silently defeat Tier 2 for every candidate whose deep scan hasn't run
yet -- which, given the timeout mismatch above, is every candidate at
the moment of the webhook call. This module deliberately keeps
VERDICT_ESCALATE distinct from both VERDICT_ADMIT and VERDICT_BLOCK so
that mapping decision precisely is the integration's responsibility, not
silently defaulted here. `securemodelgate.webhook.admission_response`
implements the required mapping (only ADMIT -> allowed: true); any other
integration (Kyverno, policy-controller) must reproduce it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .conformal import ConformalVerdict

TIER_0 = "0"
TIER_1 = "1"
TIER_2 = "2"

VERDICT_ADMIT = "ADMIT"
VERDICT_ESCALATE = "ESCALATE"
VERDICT_BLOCK = "BLOCK"


@dataclass
class AdmissionDecision:
    tier: str
    verdict: str
    reason: str


def route(
    *,
    trusted_publisher_signed: bool,
    reference_verified: bool,
    lineage_verdict: Optional[ConformalVerdict] = None,
    envelope_verdict: Optional[ConformalVerdict] = None,
    deep_scan_flagged: Optional[bool] = None,
) -> AdmissionDecision:
    """Decide tier + verdict for one candidate model.

    `lineage_verdict.exceeds_threshold=True` means S_w exceeded the
    independent-model quantile, i.e. lineage IS verified (higher S_w is
    good here, unlike the envelope test below).
    `envelope_verdict.exceeds_threshold=True` means the candidate's
    anomaly score exceeded the benign-derivative quantile, i.e. it
    looks LESS like a benign derivative (higher is bad here).
    """
    if trusted_publisher_signed:
        return AdmissionDecision(TIER_0, VERDICT_ADMIT, "trusted publisher signature verified")

    if not reference_verified:
        if deep_scan_flagged is None:
            return AdmissionDecision(TIER_2, VERDICT_ESCALATE,
                                      "no verified OMS-signed base declared; routed to deep scan")
        verdict = VERDICT_BLOCK if deep_scan_flagged else VERDICT_ADMIT
        reason = "deep scan flagged the model" if deep_scan_flagged else "deep scan found nothing; admitted with attestation"
        return AdmissionDecision(TIER_2, verdict, reason)

    if lineage_verdict is None or envelope_verdict is None:
        raise ValueError("reference_verified=True requires both lineage_verdict and envelope_verdict")

    if not lineage_verdict.exceeds_threshold:
        return AdmissionDecision(
            TIER_1, VERDICT_BLOCK,
            lineage_verdict.failure_reason or (
                f"S_w={lineage_verdict.score:.4f} did not exceed the independent-model "
                f"quantile ({lineage_verdict.threshold:.4f} at alpha_L={lineage_verdict.alpha}); "
                f"candidate does not verify as a descendant of the declared base"),
        )

    if envelope_verdict.exceeds_threshold:
        if deep_scan_flagged is None:
            return AdmissionDecision(
                TIER_2, VERDICT_ESCALATE,
                f"envelope score {envelope_verdict.score:.4f} exceeded the benign-derivative "
                f"conformal threshold ({envelope_verdict.threshold:.4f} at alpha={envelope_verdict.alpha}); "
                f"escalated for deep scan",
            )
        verdict = VERDICT_BLOCK if deep_scan_flagged else VERDICT_ADMIT
        reason = ("escalated candidate: deep scan flagged the model" if deep_scan_flagged
                  else "escalated candidate: deep scan found nothing; admitted with attestation")
        return AdmissionDecision(TIER_2, verdict, reason)

    return AdmissionDecision(
        TIER_1, VERDICT_ADMIT,
        f"lineage verified (S_w={lineage_verdict.score:.4f}) and within the benign-derivative "
        f"envelope (score={envelope_verdict.score:.4f} <= {envelope_verdict.threshold:.4f})",
    )
