"""Kubernetes AdmissionReview response for a SecureModelGate decision.

A ValidatingAdmissionWebhook call is synchronous with a bounded timeout,
and deep scan cannot finish inside it. So ESCALATE -- "no deep-scan result
yet" -- must deny at the webhook; deep scan runs out of band and a later
re-submission carries the resolved verdict. Only ADMIT maps to
`allowed: true`; ESCALATE, BLOCK, and any unrecognised verdict deny.

`review_attestation` is the path a webhook takes when it is handed a
previously issued attestation rather than running the gate itself: it
allows only an attestation that verifies against the presented artifacts
AND records ADMIT, and denies on any exception.
"""

from __future__ import annotations

from .lineage.admission_policy import VERDICT_ADMIT, AdmissionDecision


def admission_response(uid: str, decision: AdmissionDecision) -> dict:
    allowed = decision.verdict == VERDICT_ADMIT
    return {
        "apiVersion": "admission.k8s.io/v1",
        "kind": "AdmissionReview",
        "response": {
            "uid": uid,
            "allowed": allowed,
            "status": {
                "code": 200 if allowed else 403,
                "message": f"SecureModelGate tier {decision.tier} {decision.verdict}: {decision.reason}",
            },
        },
    }


def review_attestation(uid: str, envelope: dict, signer, **verify_kwargs) -> dict:
    """AdmissionReview for a presented attestation. `verify_kwargs` are
    `verify.verify_attestation`'s keyword arguments (candidate, mlbom,
    calibration sets, probe_loader, and the optional freshness / registry /
    recompute checks)."""
    from .verify import verify_attestation

    try:
        ok, reason = verify_attestation(envelope, signer, **verify_kwargs)
        if not ok:
            return admission_response(uid, AdmissionDecision("-", "DENY", f"attestation rejected: {reason}"))
        _, payload = signer.verify(envelope)
        pred = payload["predicate"]
        return admission_response(uid, AdmissionDecision(str(pred["tier"]), str(pred["verdict"]),
                                                         f"attestation verified ({reason})"))
    except Exception as e:
        return admission_response(uid, AdmissionDecision("-", "DENY", f"verification error: {type(e).__name__}: {e}"))


def review_candidate(uid: str, **run_admission_kwargs) -> tuple[dict, object]:
    """Run the gate on a candidate and return (AdmissionReview, PipelineResult
    or None). Any exception -- a malformed ML-BOM, non-finite weights, a
    missing calibration set -- is a DENY, never an unanswered request that a
    webhook's failurePolicy might turn into an admit."""
    from .pipeline import run_admission

    try:
        result = run_admission(**run_admission_kwargs)
    except Exception as e:
        return admission_response(uid, AdmissionDecision("-", "DENY", f"gate error: {type(e).__name__}: {e}")), None
    return admission_response(uid, result.decision), result
