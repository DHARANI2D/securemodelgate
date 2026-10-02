"""End-to-end admission pipeline.

Wires the six `securemodelgate.lineage` modules into the sequence
Section 3 of the revised paper describes:

    resolve declared base (3.1)
      -> S_w weight-lineage score + D_b behavioral delta (3.2)
      -> split-conformal lineage/envelope verdicts (3.3)
      -> Tier 0/1/2 routing (3.5)
      -> signed in-toto predicate (3.4)

This is the module `securemodelgate.cli`'s `demo` and `admit`
subcommands call, and what `tests/test_pipeline.py` exercises end to
end against real (if small) torch models.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Collection, Mapping, Optional, Sequence

from . import digest
from .lineage import admission_policy, conformal, intoto_attestation
from .lineage.behavioral_delta import commit_seed, compute_behavioral_delta_from_models
from .lineage.lineage_score import weight_lineage_score_from_models
from .lineage.reference_resolver import (
    ReferenceResolution, declared_replaced_parameters, resolve_reference,
)

DEFAULT_MAX_EXEMPT = 1   # one replaced matrix: a re-initialised classification head


@dataclass
class PipelineResult:
    resolution: ReferenceResolution
    decision: admission_policy.AdmissionDecision
    attestation: dict                                        # signed DSSE envelope
    lineage_score: Optional[dict] = None                      # S_w dict, if computed
    behavioral: Optional[dict] = None                         # D_b dict, if computed
    lineage_verdict: Optional[conformal.ConformalVerdict] = None    # combined: median AND per-layer floor
    envelope_verdict: Optional[conformal.ConformalVerdict] = None
    layer_floor_verdict: Optional[conformal.ConformalVerdict] = None    # the per-layer-floor sub-check alone


def envelope_anomaly_score(behavioral: dict) -> float:
    """The single 'named envelope score' the paper's evaluation section
    calls for (Section 4): the max of the two calibrated components, so
    a candidate is flagged if EITHER signal looks anomalous.

    Non-finite components raise. Python's max(x, nan) returns x, so an
    earlier version silently dropped a NaN JS: a candidate whose logits
    overflow (finite fc weights of ~1e37) scored on CKA alone (0.019) and
    would have been admitted. A score the gate cannot compute is an error.
    """
    import math
    cka, js = behavioral["min_cka"], behavioral["js_divergence"]
    if not (math.isfinite(cka) and math.isfinite(js)):
        raise ValueError(f"behavioral delta is non-finite (min_cka={cka}, js={js}); cannot score the candidate")
    return max(1.0 - cka, js)


def combined_lineage_verdict(
    median_verdict: conformal.ConformalVerdict,
    layer_floor_verdict: conformal.ConformalVerdict,
) -> conformal.ConformalVerdict:
    """Lineage verifies only if BOTH the median-aggregated S_w clears its
    threshold AND every matched layer's own score clears a floor
    calibrated on independent models' pooled per-layer scores.

    Guards against splicing: verbatim-copying enough layers (a bit over
    half, for a median) from the real base while swapping the rest for an
    independently-trained — and potentially backdoored — block clears the
    median threshold on its own (empirically S_w ~0.75-0.81 against a
    ~0.59 threshold, see `paper/techcon2027_revision/03_change_log.md`'s
    second-revision-pass entry), even though the swapped layers score in
    the unrelated-model range individually. Requiring every layer to pass
    its own floor closes that gap without breaking genuine derivatives:
    benign fine-tunes / pruned (up to 50%) / quantized (down to 4 levels)
    derivatives all have per-layer MINIMUM scores >= ~0.92 in the pilot
    corpus, comfortably clear of independent models' per-layer MAXIMUM of
    ~0.65.

    When the median passes but the floor doesn't, the returned verdict
    reports the floor check's score/threshold (not the median's), since
    that's the actual reason lineage fails — reporting the median's
    passing numbers alongside exceeds_threshold=False would read as a
    contradiction.
    """
    lineage_passes = median_verdict.exceeds_threshold and layer_floor_verdict.exceeds_threshold
    combined_warning = " ".join(w for w in (median_verdict.warning, layer_floor_verdict.warning) if w)
    if median_verdict.exceeds_threshold and not layer_floor_verdict.exceeds_threshold:
        return conformal.ConformalVerdict(
            threshold=layer_floor_verdict.threshold, alpha=layer_floor_verdict.alpha,
            n_calibration=layer_floor_verdict.n_calibration, score=layer_floor_verdict.score,
            exceeds_threshold=False, warning=combined_warning,
        )
    return conformal.ConformalVerdict(
        threshold=median_verdict.threshold, alpha=median_verdict.alpha,
        n_calibration=median_verdict.n_calibration, score=median_verdict.score,
        exceeds_threshold=lineage_passes, warning=combined_warning,
    )


def lineage_verdict_for(
    lineage_score: dict,
    lineage_calibration_scores: Sequence[float],
    layer_floor_calibration_scores: Sequence[float],
    alpha: float,
    max_exempt: int = DEFAULT_MAX_EXEMPT,
    replaceable: Optional[Collection[str]] = None,
) -> tuple[conformal.ConformalVerdict, conformal.ConformalVerdict]:
    """The full lineage rule, shared by `run_admission` and `evaluation.py`
    so the pilot measures exactly what the gate enforces: median S_w AND
    per-layer floor AND at most `max_exempt` unscored weight matrices.
    Returns (combined verdict, per-layer-floor sub-verdict).

    The cap is what stops "exemption" from becoming the new skip: every
    matrix the scorer could not score (declared replaced, missing, added,
    incomparable) counts, and a candidate with more than `max_exempt` of
    them fails lineage outright, whatever its scored layers look like.
    `replaceable`, when given, is the operator's list of matrices a
    candidate may declare replaced (e.g. only the classification head);
    without it, ANY one matrix can be declared, since the declaration is
    the candidate's own unverified claim.
    """
    median_v = conformal.lineage_test(lineage_score["S_w"], lineage_calibration_scores, alpha_l=alpha)
    floor_v = conformal.lineage_test(lineage_score["min_layer_score"], layer_floor_calibration_scores, alpha_l=alpha)
    verdict = combined_lineage_verdict(median_v, floor_v)
    exempt = lineage_score.get("exempt", [])
    if replaceable is not None:
        disallowed = [e["name"] for e in exempt
                      if e["reason"] == "declared_replaced" and e["name"] not in set(replaceable)]
        if disallowed:
            return replace(verdict, exceeds_threshold=False, failure_reason=(
                f"ML-BOM declares {', '.join(disallowed)} replaced, but policy only allows replacing "
                f"{sorted(replaceable)}; candidate does not verify as a descendant")), floor_v
    if len(exempt) > max_exempt:
        listed = ", ".join(f"{e['name']} ({e['reason']})" for e in exempt)
        verdict = replace(verdict, exceeds_threshold=False, failure_reason=(
            f"{len(exempt)} weight matrices could not be scored against the declared base "
            f"(max_exempt={max_exempt}): {listed}; candidate does not verify as a descendant"))
    return verdict, floor_v


def _bind_deep_scan(flagged: Optional[bool], report: Optional[dict],
                    subject_sha256: str) -> tuple[Optional[bool], Optional[dict]]:
    """Return (flagged, predicate deepScan field). A report about different
    bytes than the candidate's -- or a malformed one -- counts as no result."""
    if report is None:
        if flagged is None:
            return None, None
        return flagged, {"tool": "external", "version": "n/a", "resultDigest": "n/a",
                         "flagged": flagged, "bound": False}
    field = {"tool": str(report.get("tool")), "version": str(report.get("version")),
             "resultDigest": digest.json_digest(report), "subjectDigest": report.get("subjectDigest")}
    if report.get("subjectDigest") != subject_sha256 or not isinstance(report.get("flagged"), bool):
        return None, {**field, "flagged": None, "bound": False,
                      "rejected": "deep-scan report is not about this candidate's bytes, or is malformed"}
    return report["flagged"], {**field, "flagged": report["flagged"], "bound": True}


def run_admission(
    *,
    candidate,                                    # torch.nn.Module
    candidate_subject_name: str,
    mlbom: dict,
    allowed_signers: Sequence[str],
    signer: intoto_attestation.LocalHmacSigner,
    trusted_publisher_signed: bool = False,
    base_model=None,                              # torch.nn.Module; required iff a reference resolves
    probe_loader=None,                             # required iff a reference resolves
    probe_seed: int = 0,
    lineage_calibration_scores: Optional[Sequence[float]] = None,
    envelope_calibration_scores: Optional[Sequence[float]] = None,
    lineage_layer_floor_calibration_scores: Optional[Sequence[float]] = None,
    alpha_lineage: float = 0.05,
    alpha_envelope: float = 0.05,
    max_exempt: int = DEFAULT_MAX_EXEMPT,
    replaceable: Optional[Collection[str]] = None,
    deep_scan_flagged: Optional[bool] = None,
    deep_scan_report: Optional[dict] = None,
    trusted_bases: Optional[Mapping[str, str]] = None,
    calibration_base_digest: Optional[str] = None,
    gate_version: Optional[str] = None,
) -> PipelineResult:
    """Run one candidate model through the full admission pipeline.

    Tier 0 (trusted_publisher_signed=True) and Tier 2 (no verified
    reference) short-circuit before any lineage/behavioral scoring.
    Tier 1 requires `base_model`, `probe_loader`, and all three
    calibration score sets (lineage, envelope, and the per-layer floor)
    — the ValueError below is intentionally loud rather than silently
    skipping the scoring it needs. Lineage verification requires BOTH
    the median-aggregated S_w to exceed its threshold AND every matched
    layer's own score to individually clear a floor calibrated on
    independent models' pooled per-layer scores (`lineage_score.
    pooled_layer_scores`) — the second check exists specifically to
    catch a candidate that verbatim-copies enough layers to game the
    median while swapping the rest for an independently-trained block.

    Deep scan: `deep_scan_report` ({"tool", "version", "flagged": bool,
    "subjectDigest": sha256}) is bound to the candidate's bytes -- a report
    whose subjectDigest is not this candidate's weight digest is ignored
    (the candidate stays ESCALATE), so a clean scan of one artifact cannot
    admit another. `deep_scan_flagged` is the older unbound boolean, kept
    for callers and tests that model the scanner abstractly; the
    attestation records `bound: false` for it.

    `calibration_base_digest`, when given, is the base the calibration sets
    were computed against; it must equal the supplied base model's digest
    (thresholds calibrated for one base say nothing about another).
    """
    if gate_version is None:
        from . import __version__ as gate_version
    if deep_scan_report is not None and deep_scan_flagged is not None:
        raise ValueError("pass deep_scan_report or deep_scan_flagged, not both")
    resolution = resolve_reference(mlbom, allowed_signers=set(allowed_signers), trusted_bases=trusted_bases)

    if resolution.verified and base_model is not None:
        # `resolve_reference` only checked the CANDIDATE's own claim about its
        # ancestor (signer on the allow-list, well-formed digest) -- it has no
        # way to know whether `base_model` (whatever a purl lookup handed the
        # caller) is actually the artifact that digest names. Score against
        # the wrong base and every downstream number is meaningless, but
        # nothing would fail loudly: S_w/D_b would just report how (un)related
        # the candidate is to whatever was passed in, and the attestation
        # would still claim a verified declaredBase. Bind the two explicitly
        # and fail closed (treat as unverified -> Tier 2) on a mismatch,
        # rather than silently proceeding to a scoring run whose "verified"
        # premise doesn't hold.
        actual_base_digest = digest.weight_digest(base_model)
        if resolution.oms_bundle_digest != actual_base_digest:
            resolution = ReferenceResolution(
                verified=False,
                base_purl=resolution.base_purl,
                oms_bundle_digest=resolution.oms_bundle_digest,
                signer_identity=resolution.signer_identity,
                reason=(f"declared ancestor digest ({resolution.oms_bundle_digest}) does not "
                        f"match the base_model supplied for scoring (actual digest "
                        f"{actual_base_digest}); a verified signer on an unrelated artifact "
                        f"proves nothing, so the reference is treated as unverified"),
            )

    subject_sha256 = digest.weight_digest(candidate)
    mlbom_digest = digest.json_digest(mlbom)
    deep_scan_flagged, deep_scan = _bind_deep_scan(deep_scan_flagged, deep_scan_report, subject_sha256)

    declared_base = {
        "purl": resolution.base_purl,
        "omsBundleDigest": resolution.oms_bundle_digest,
        "signerIdentity": resolution.signer_identity,
    }

    if trusted_publisher_signed or not resolution.verified:
        decision = admission_policy.route(
            trusted_publisher_signed=trusted_publisher_signed,
            reference_verified=resolution.verified,
            deep_scan_flagged=deep_scan_flagged,
        )
        envelope = intoto_attestation.issue_attestation(
            signer=signer, subject_name=candidate_subject_name, subject_sha256=subject_sha256,
            declared_base=declared_base, mlbom_digest=mlbom_digest,
            lineage={}, envelope={}, probe={},
            tier=decision.tier, deep_scan=deep_scan, verdict=decision.verdict,
            gate_version=gate_version,
        )
        return PipelineResult(resolution=resolution, decision=decision, attestation=envelope)

    if base_model is None or probe_loader is None:
        raise ValueError("a verified reference requires base_model and probe_loader "
                          "to compute S_w and D_b")
    if lineage_calibration_scores is None or envelope_calibration_scores is None:
        raise ValueError("a verified reference requires lineage_calibration_scores and "
                          "envelope_calibration_scores to set conformal thresholds")
    if lineage_layer_floor_calibration_scores is None:
        raise ValueError(
            "a verified reference requires lineage_layer_floor_calibration_scores "
            "(pooled per-layer S_w scores from the independent-model calibration "
            "set -- see lineage_score.pooled_layer_scores) to set the per-layer "
            "floor threshold. S_w's median-over-layers aggregation is deliberately "
            "tolerant of a MINORITY of layers looking unrelated (so a few heavily "
            "pruned/re-initialized layers in a genuine derivative don't tank the "
            "verdict); without this floor, a candidate that verbatim-copies just "
            "over half the matched layers and swaps the rest for an independently-"
            "trained block can clear the median threshold undetected -- see "
            "tests/test_pipeline.py::test_layer_splicing_attack_is_blocked_by_layer_floor."
        )

    base_digest = digest.weight_digest(base_model)
    if calibration_base_digest is not None and calibration_base_digest != base_digest:
        raise ValueError(f"calibration sets were computed against base {calibration_base_digest}, "
                         f"not the declared base {base_digest}; refusing to apply them")

    declared_replaced = declared_replaced_parameters(mlbom)
    lineage_score = weight_lineage_score_from_models(candidate, base_model, declared_replaced=declared_replaced)
    behavioral = compute_behavioral_delta_from_models(candidate, base_model, probe_loader)
    anomaly = envelope_anomaly_score(behavioral)
    if digest.weight_digest(candidate) != subject_sha256:
        raise ValueError("candidate weights changed while it was being scored; refusing to attest")

    lineage_verdict, layer_floor_verdict = lineage_verdict_for(
        lineage_score, lineage_calibration_scores, lineage_layer_floor_calibration_scores,
        alpha=alpha_lineage, max_exempt=max_exempt, replaceable=replaceable,
    )
    median_threshold = conformal.conformal_quantile(lineage_calibration_scores, alpha_lineage)
    envelope_verdict = conformal.envelope_test(
        candidate_score=anomaly,
        benign_calibration_scores=envelope_calibration_scores,
        alpha=alpha_envelope,
    )

    decision = admission_policy.route(
        trusted_publisher_signed=False,
        reference_verified=True,
        lineage_verdict=lineage_verdict,
        envelope_verdict=envelope_verdict,
        deep_scan_flagged=deep_scan_flagged,
    )

    envelope = intoto_attestation.issue_attestation(
        signer=signer, subject_name=candidate_subject_name, subject_sha256=subject_sha256,
        declared_base=declared_base, mlbom_digest=mlbom_digest,
        lineage={
            "S_w": lineage_score["S_w"], "threshold": median_threshold,
            "minLayerScore": lineage_score["min_layer_score"], "layerFloorThreshold": layer_floor_verdict.threshold,
            "exempt": lineage_score["exempt"], "maxExempt": max_exempt,
            "alpha_L": alpha_lineage, "calibrationBaseDigest": calibration_base_digest,
            "calibrationSetDigest": digest.json_digest(list(lineage_calibration_scores)),
            "layerFloorCalibrationSetDigest": digest.json_digest(list(lineage_layer_floor_calibration_scores)),
        },
        envelope={
            "minCKA": behavioral["min_cka"], "JS": behavioral["js_divergence"],
            "threshold": envelope_verdict.threshold, "alpha": alpha_envelope,
            "n": len(envelope_calibration_scores),
            "calibrationSetDigest": digest.json_digest(list(envelope_calibration_scores)),
        },
        probe={
            "poolDigest": digest.probe_pool_digest(probe_loader),
            "seedCommitment": commit_seed(probe_seed), "seed": probe_seed,
        },
        tier=decision.tier, deep_scan=deep_scan, verdict=decision.verdict,
        gate_version=gate_version,
    )

    return PipelineResult(
        resolution=resolution, decision=decision, attestation=envelope,
        lineage_score=lineage_score, behavioral=behavioral,
        lineage_verdict=lineage_verdict, envelope_verdict=envelope_verdict,
        layer_floor_verdict=layer_floor_verdict,
    )
