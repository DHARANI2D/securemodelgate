"""Admission-time verification of a SecureModelGate attestation.

A valid signature only proves the gate said something. This recomputes
every digest the predicate binds -- candidate weights, ML-BOM, each
calibration set, the probe pool, the seed commitment -- from the artifacts
actually presented at admission, and fails closed on any mismatch, so an
attestation cannot be replayed against a different model, a swapped
ML-BOM, or thresholds calibrated on different data than it claims.

It also re-derives what the predicate asserts from what it records: each
threshold must be the conformal quantile of its (digest-bound) calibration
set at the recorded alpha, and the recorded verdict must be the one the
recorded scores and thresholds imply. With `base_model`, it recomputes the
scores themselves (S_w, per-layer minimum, exemptions, CKA, JS) and
compares them within `score_tolerance`. Optional freshness
(`max_age_seconds`), trusted-base (`trusted_bases`, digest -> signer) and
policy (`expected_policy`: alpha_L, alpha, maxExempt) checks bound replay of
an old, revoked, or superseded-policy attestation.

Any exception while parsing or checking -- a malformed envelope, a missing
predicate field -- is a verification failure, never an uncaught error a
caller might treat as "no answer".
"""

from __future__ import annotations

import calendar
import math
import time
from typing import Mapping, Optional, Sequence

from . import digest
from .lineage.behavioral_delta import commit_seed
from .lineage.conformal import conformal_quantile
from .lineage.intoto_attestation import PREDICATE_TYPE, STATEMENT_TYPE, LocalHmacSigner


def verify_attestation(
    envelope: dict,
    signer: LocalHmacSigner,
    *,
    candidate,
    mlbom: dict,
    lineage_calibration_scores: Optional[Sequence[float]] = None,
    layer_floor_calibration_scores: Optional[Sequence[float]] = None,
    envelope_calibration_scores: Optional[Sequence[float]] = None,
    probe_loader=None,
    expected_gate_version: Optional[str] = None,
    base_model=None,
    score_tolerance: float = 1e-6,
    max_age_seconds: Optional[float] = None,
    now: Optional[float] = None,
    trusted_bases: Optional[Mapping[str, str]] = None,
    expected_policy: Optional[Mapping[str, float]] = None,
) -> tuple[bool, str]:
    """Return (ok, reason). Tier 1 attestations carry lineage/envelope/probe
    evidence, so verifying one REQUIRES the calibration sets and probe
    loader: without them the binding can't be checked, and an unchecked
    binding is treated as a failure, not skipped.

    `ok` means the attestation is authentic and internally consistent for
    THIS candidate -- not that the candidate may be admitted. A genuine
    BLOCK attestation verifies; `webhook.review_attestation` is what
    requires verdict == ADMIT on top.
    """
    try:
        return _verify(envelope, signer, candidate=candidate, mlbom=mlbom,
                       lineage_cal=lineage_calibration_scores, floor_cal=layer_floor_calibration_scores,
                       envelope_cal=envelope_calibration_scores, probe_loader=probe_loader,
                       expected_gate_version=expected_gate_version, base_model=base_model,
                       tol=score_tolerance, max_age_seconds=max_age_seconds,
                       now=time.time() if now is None else now, trusted_bases=trusted_bases,
                       expected_policy=expected_policy)
    except Exception as e:   # malformed input is a failure, not a crash
        return False, f"malformed attestation or inputs ({type(e).__name__}: {e})"


def _close(a: float, b: float, tol: float) -> bool:
    return math.isfinite(a) and math.isfinite(b) and abs(a - b) <= tol


def _verify(envelope, signer, *, candidate, mlbom, lineage_cal, floor_cal, envelope_cal, probe_loader,
            expected_gate_version, base_model, tol, max_age_seconds, now, trusted_bases, expected_policy):
    ok, payload = signer.verify(envelope)
    if not ok:
        return False, f"signature: {payload}"
    if payload.get("_type") != STATEMENT_TYPE or payload.get("predicateType") != PREDICATE_TYPE:
        return False, "not a SecureModelGate in-toto statement (statement or predicate type differs)"

    pred = payload["predicate"]
    subject = payload["subject"][0]["digest"]["sha256"]
    if subject != digest.weight_digest(candidate):
        return False, "subject digest does not match the candidate's weights"
    if pred["mlbomDigest"] != digest.json_digest(mlbom):
        return False, "mlbomDigest does not match the presented ML-BOM"
    if expected_gate_version is not None and pred["gateVersion"] != expected_gate_version:
        return False, f"gateVersion {pred['gateVersion']} != expected {expected_gate_version}"
    if max_age_seconds is not None:
        issued = calendar.timegm(time.strptime(pred["timestamp"], "%Y-%m-%dT%H:%M:%SZ"))
        if now - issued > max_age_seconds:
            return False, f"attestation is {now - issued:.0f}s old (max {max_age_seconds:.0f}s); re-admit"
        if issued - now > 300:
            return False, "attestation timestamp is in the future"

    if not pred["lineage"]:
        return True, "ok (no lineage evidence to bind: Tier 0 or Tier 2 attestation)"

    base = pred["declaredBase"]
    if trusted_bases is not None and trusted_bases.get(base["omsBundleDigest"]) != base["signerIdentity"]:
        return False, "declared base is not in the trusted-base registry with this signer (unknown or revoked)"
    cal_base = pred["lineage"].get("calibrationBaseDigest")
    if cal_base is not None and cal_base != base["omsBundleDigest"]:
        return False, "calibration sets were computed against a different base than the declared one"

    required = {
        "lineage_calibration_scores": lineage_cal,
        "layer_floor_calibration_scores": floor_cal,
        "envelope_calibration_scores": envelope_cal,
        "probe_loader": probe_loader,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        return False, f"cannot verify Tier 1 evidence binding without: {', '.join(missing)}"

    lin, env, probe = pred["lineage"], pred["envelope"], pred["probe"]
    if expected_policy is not None:
        recorded = {"alpha_L": lin["alpha_L"], "alpha": env["alpha"], "maxExempt": lin["maxExempt"]}
        changed = {k: (recorded[k], v) for k, v in expected_policy.items() if recorded.get(k) != v}
        if changed:
            return False, f"attestation was issued under a different policy: {changed}"
    checks = [
        (lin["calibrationSetDigest"], digest.json_digest(list(lineage_cal)), "lineage calibration set"),
        (lin["layerFloorCalibrationSetDigest"], digest.json_digest(list(floor_cal)), "layer-floor calibration set"),
        (env["calibrationSetDigest"], digest.json_digest(list(envelope_cal)), "envelope calibration set"),
        (probe["poolDigest"], digest.probe_pool_digest(probe_loader), "probe pool"),
        (probe["seedCommitment"], commit_seed(probe["seed"]), "probe seed commitment"),
    ]
    for recorded, recomputed, what in checks:
        if recorded != recomputed:
            return False, f"{what} digest does not match the attestation"

    # Thresholds must be what the bound calibration sets imply.
    for recorded, cal, alpha, what in [
        (lin["threshold"], lineage_cal, lin["alpha_L"], "lineage median threshold"),
        (lin["layerFloorThreshold"], floor_cal, lin["alpha_L"], "layer-floor threshold"),
        (env["threshold"], envelope_cal, env["alpha"], "envelope threshold"),
    ]:
        if recorded != conformal_quantile(cal, alpha):
            return False, f"{what} is not the conformal quantile of its calibration set"
    if env["n"] != len(envelope_cal):
        return False, "envelope calibration size does not match"

    # The verdict must follow from the recorded scores and thresholds.
    lineage_ok = (lin["S_w"] > lin["threshold"] and lin["minLayerScore"] > lin["layerFloorThreshold"]
                  and len(lin["exempt"]) <= lin["maxExempt"])
    anomaly = max(1.0 - env["minCKA"], env["JS"])
    if not lineage_ok:
        expected = ("1", "BLOCK")
    elif anomaly > env["threshold"]:
        flagged = (pred["deepScan"] or {}).get("flagged")
        expected = ("2", "ESCALATE" if flagged is None else ("BLOCK" if flagged else "ADMIT"))
    else:
        expected = ("1", "ADMIT")
    if (pred["tier"], pred["verdict"]) != expected:
        return False, f"recorded tier/verdict {pred['tier']}/{pred['verdict']} does not follow from its scores ({expected[0]}/{expected[1]})"

    if base_model is not None:
        ok, reason = _recompute_scores(pred, candidate, mlbom, base_model, probe_loader, tol)
        if not ok:
            return False, reason
        return True, "ok (digests, thresholds, verdict and recomputed scores all match)"
    return True, "ok (digests, thresholds and verdict match; scores not recomputed)"


def _recompute_scores(pred, candidate, mlbom, base_model, probe_loader, tol):
    from .lineage.behavioral_delta import compute_behavioral_delta_from_models
    from .lineage.lineage_score import weight_lineage_score_from_models
    from .lineage.reference_resolver import declared_replaced_parameters

    if digest.weight_digest(base_model) != pred["declaredBase"]["omsBundleDigest"]:
        return False, "supplied base_model is not the declared base"
    lin, env = pred["lineage"], pred["envelope"]
    s = weight_lineage_score_from_models(candidate, base_model,
                                         declared_replaced=declared_replaced_parameters(mlbom))
    if not (_close(s["S_w"], lin["S_w"], tol) and _close(s["min_layer_score"], lin["minLayerScore"], tol)):
        return False, "recomputed S_w / per-layer minimum differ from the attestation"
    if s["exempt"] != lin["exempt"]:
        return False, "recomputed exemptions differ from the attestation"
    b = compute_behavioral_delta_from_models(candidate, base_model, probe_loader)
    if not (_close(b["min_cka"], env["minCKA"], tol) and _close(b["js_divergence"], env["JS"], tol)):
        return False, "recomputed CKA / JS differ from the attestation"
    return True, "ok"
