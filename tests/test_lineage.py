"""
Unit tests for the pure-math pieces of securemodelgate.lineage. No torch
dependency: these test the numpy-only cores (spectral/directional
agreement, linear CKA, JS divergence, conformal thresholds, attestation
signing, admission routing). The torch-dependent model-hook adapters and
the end-to-end pipeline are covered separately in test_pipeline.py.
"""

import math

import numpy as np
import pytest

from securemodelgate.lineage import admission_policy, conformal, intoto_attestation, reference_resolver
from securemodelgate.lineage.behavioral_delta import (
    commit_seed,
    jensen_shannon_divergence,
    linear_cka,
    behavioral_delta,
)
from securemodelgate.lineage.lineage_score import (
    spectral_correlation,
    directional_similarity,
    channel_aligned_directional_similarity,
    layer_lineage_score,
    weight_lineage_score,
)


# ── lineage_score ────────────────────────────────────────────────────────────

def test_spectral_correlation_identical_layers_is_one():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(16, 32))
    assert spectral_correlation(w, w) == pytest.approx(1.0, abs=1e-9)


def test_directional_similarity_identical_is_one_orthogonal_is_low():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(8, 8))
    assert directional_similarity(w, w) == pytest.approx(1.0, abs=1e-9)
    assert directional_similarity(w, -w) == pytest.approx(-1.0, abs=1e-9)


def test_directional_similarity_zero_vector_is_zero_not_nan():
    z = np.zeros((4, 4))
    w = np.ones((4, 4))
    assert directional_similarity(z, w) == 0.0


def test_weight_lineage_score_descendant_scores_higher_than_unrelated():
    rng = np.random.default_rng(2)
    base_layers = [rng.normal(size=(16, 32)) for _ in range(4)]
    # "descendant": small perturbation of the base (like fine-tuning)
    descendant_layers = [w + 0.01 * rng.normal(size=w.shape) for w in base_layers]
    # "unrelated": independently drawn weights of the same shape
    unrelated_layers = [rng.normal(size=w.shape) for w in base_layers]

    s_descendant = weight_lineage_score(list(zip(descendant_layers, base_layers)))["S_w"]
    s_unrelated = weight_lineage_score(list(zip(unrelated_layers, base_layers)))["S_w"]

    assert s_descendant > s_unrelated


def test_weight_lineage_score_empty_is_zero():
    assert weight_lineage_score([]) == {"S_w": 0.0, "n_layers": 0, "per_layer": [], "min_layer_score": 0.0}


def test_channel_aligned_directional_similarity_survives_row_permutation():
    """Regression test: permuting a weight matrix's output-channel rows is
    an exact, function-preserving symmetry of any ReLU/BatchNorm layer,
    but plain cosine-of-flattened similarity (directional_similarity)
    collapses under it. channel_aligned_directional_similarity should not.
    """
    rng = np.random.default_rng(3)
    w = rng.normal(size=(16, 32))
    perm = rng.permutation(16)
    w_permuted = w[perm]

    naive = directional_similarity(w, w_permuted)
    aligned = channel_aligned_directional_similarity(w, w_permuted)

    assert aligned == pytest.approx(1.0, abs=1e-6)
    assert naive < 0.3   # the failure this regression test guards against


def test_layer_lineage_score_recovers_after_single_layer_permutation():
    rng = np.random.default_rng(4)
    w_base = rng.normal(size=(16, 32))
    w_descendant = w_base + 0.01 * rng.normal(size=w_base.shape)
    perm = rng.permutation(16)
    w_descendant_permuted = w_descendant[perm]

    unpermuted_score = layer_lineage_score(w_descendant, w_base)
    permuted_score = layer_lineage_score(w_descendant_permuted, w_base)

    assert unpermuted_score == pytest.approx(permuted_score, abs=1e-6)


def test_channel_aligned_directional_similarity_handles_channel_count_mismatch():
    """`linear_sum_assignment` accepts a rectangular cost matrix natively
    (matching min(rows, cols) pairs) -- should not raise on a shape
    mismatch, and unrelated matrices at different channel counts should
    still score low (not the near-1.0 a degenerate fallback might give)."""
    rng = np.random.default_rng(5)
    a = rng.normal(size=(16, 32))
    b = rng.normal(size=(12, 32))
    score = channel_aligned_directional_similarity(a, b)
    assert -1.0 <= score <= 1.0
    # Best-of-many assignment inflates unrelated matrices' matched-pair
    # mean similarity somewhat above a single random pair's ~0 expectation
    # (the same effect noted elsewhere for independent-model S_w), so this
    # is a loose sanity bound, not a calibrated separation claim -- that's
    # what test_..._recovers_scattered_channel_pruning below checks precisely.
    assert score < 0.6


def test_channel_aligned_directional_similarity_recovers_scattered_channel_pruning():
    """Regression test for a real bug: an earlier version of this function
    fell back to plain `directional_similarity` (flattened, truncated
    cosine) whenever channel counts differed, reasoning the assignment
    problem "needs equal-sized sets" -- it doesn't, scipy's solver
    handles rectangular matrices directly. The fallback was actively
    wrong whenever structured pruning dropped a SCATTERED subset of
    channels rather than a tidy trailing block (the common case for real
    saliency-based pruning): a candidate keeping 6 of 8 channels
    VERBATIM (byte-identical) from the base scored ~1.0 when the two
    dropped channels were the last two, but collapsed to ~0.13 --
    nearly as low as an unrelated model's ~-0.08 at the same shape --
    when the dropped channels were scattered (indices 2 and 5 of 8).
    The rectangular-assignment fix must recover ~1.0 in BOTH cases,
    since the kept channels are identical either way and permutation/
    subset-order was never a real functional difference.
    """
    rng = np.random.default_rng(0)
    w_base = rng.normal(size=(8, 16))

    prefix_pruned = w_base[:6, :].copy()               # drops a trailing block
    scattered_pruned = w_base[[0, 1, 3, 4, 6, 7], :].copy()   # drops channels 2 and 5
    unrelated_same_shape = rng.normal(size=(6, 16))

    s_prefix = channel_aligned_directional_similarity(prefix_pruned, w_base)
    s_scattered = channel_aligned_directional_similarity(scattered_pruned, w_base)
    s_unrelated = channel_aligned_directional_similarity(unrelated_same_shape, w_base)

    assert s_prefix == pytest.approx(1.0, abs=1e-6)
    assert s_scattered == pytest.approx(1.0, abs=1e-6)   # ~0.13 before the fix
    assert s_unrelated < 0.3


# ── behavioral_delta ─────────────────────────────────────────────────────────

def test_linear_cka_identical_representations_is_one():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(50, 20))
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-9)


def test_linear_cka_invariant_to_orthogonal_transform_and_scaling():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(50, 20))
    q, _ = np.linalg.qr(rng.normal(size=(20, 20)))  # random orthogonal matrix
    y = 3.7 * (x @ q)
    assert linear_cka(x, y) == pytest.approx(1.0, abs=1e-6)


def test_linear_cka_unrelated_representations_lower_than_identical():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(200, 20))
    y = rng.normal(size=(200, 20))
    assert linear_cka(x, y) < 0.5


def test_jensen_shannon_divergence_bounds_and_identity():
    p = np.array([0.9, 0.1])
    assert jensen_shannon_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    q = np.array([0.0, 1.0])
    js = jensen_shannon_divergence(p, q)
    assert 0.0 < js <= 1.0 + 1e-9


def test_behavioral_delta_flags_bigger_divergence_for_more_different_model():
    rng = np.random.default_rng(6)
    base_acts = rng.normal(size=(30, 10))
    close_acts = base_acts + 0.01 * rng.normal(size=base_acts.shape)
    far_acts = rng.normal(size=base_acts.shape)

    base_probs = np.tile([0.7, 0.2, 0.1], (30, 1))
    close_probs = base_probs.copy()
    far_probs = np.tile([0.1, 0.2, 0.7], (30, 1))

    close = behavioral_delta([(close_acts, base_acts)], close_probs, base_probs)
    far = behavioral_delta([(far_acts, base_acts)], far_probs, base_probs)

    assert close["min_cka"] > far["min_cka"]
    assert close["js_divergence"] < far["js_divergence"]


def test_commit_seed_is_deterministic():
    """Determinism only. The commitment is an unsalted sha256 of the seed,
    so it does NOT hide the seed: a 32-bit seed is recovered by brute force
    (20260101 in 0.3 s from a nearby start; the whole space in about an
    hour on one core). It is a reproducibility record, not a secret."""
    c1 = commit_seed(42)
    c2 = commit_seed(42)
    c3 = commit_seed(43)
    assert c1 == c2
    assert c1 != c3


# ── conformal ────────────────────────────────────────────────────────────────

def test_conformal_quantile_saturates_at_max_for_n19_alpha05():
    # n=19, alpha=0.05 -> ceil(20*0.95) = 19 -> the maximum calibration score
    scores = list(range(1, 20))  # 1..19, sorted already
    tau = conformal.conformal_quantile(scores, alpha=0.05)
    assert tau == 19.0


def test_conformal_quantile_matches_hand_worked_example():
    # n=9, alpha=0.2 -> ceil(10*0.8) = 8 -> 8th smallest of 9 sorted values
    scores = [10, 1, 2, 3, 4, 5, 6, 7, 8]
    tau = conformal.conformal_quantile(scores, alpha=0.2)
    assert tau == 8.0


def test_min_calibration_size_examples():
    assert conformal.min_calibration_size(0.05) == 19
    assert conformal.min_calibration_size(0.01) == 99


def test_envelope_test_warns_below_minimum_n():
    verdict = conformal.envelope_test(candidate_score=0.5, benign_calibration_scores=[0.1, 0.2, 0.3], alpha=0.05)
    assert verdict.warning != ""
    assert verdict.exceeds_threshold is True


def test_envelope_test_no_warning_at_sufficient_n():
    calib = list(np.linspace(0.0, 1.0, 20))
    verdict = conformal.envelope_test(candidate_score=0.5, benign_calibration_scores=calib, alpha=0.05)
    assert verdict.warning == ""


def test_conformal_quantile_rejects_nan_calibration_score():
    """Regression test for a real silent-failure mode: np.sort places NaN
    at the END of the array regardless of its true value, so a single NaN
    calibration score can become the reported threshold at a high rank
    (e.g. n=5, alpha=0.05 selects the very last sorted position). Every
    subsequent `candidate_score > tau` comparison against a NaN threshold
    is then False -- which silently disables the envelope test (nothing
    is ever flagged as anomalous) if it happens there, a fail-OPEN
    security gap, not just a stats bug. Must raise loudly instead.
    """
    with pytest.raises(ValueError, match="non-finite"):
        conformal.conformal_quantile([0.5, 0.6, float("nan"), 0.7, 0.8], alpha=0.05)
    with pytest.raises(ValueError, match="non-finite"):
        conformal.conformal_quantile([0.5, float("inf")], alpha=0.05)


def test_envelope_and_lineage_test_reject_nan_candidate_score():
    """The candidate's own score must be guarded too: a NaN candidate_score
    reads as 'not exceeding' any finite threshold (every comparison
    against NaN is False), which would silently admit a candidate whose
    behavioral/lineage score came out degenerate instead of flagging it.
    """
    with pytest.raises(ValueError, match="non-finite"):
        conformal.envelope_test(candidate_score=float("nan"), benign_calibration_scores=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="non-finite"):
        conformal.lineage_test(candidate_score=float("nan"), independent_model_scores=[0.5, 0.6, 0.7])


# ── reference_resolver ───────────────────────────────────────────────────────

def test_resolve_reference_success():
    mlbom = {
        "component": {
            "pedigree": {
                "ancestors": [{
                    "purl": "pkg:huggingface/org/base@rev",
                    "properties": [
                        {"name": "securemodelgate:omsBundleDigest", "value": "a" * 64},
                        {"name": "securemodelgate:signerIdentity", "value": "ngc-catalog"},
                    ],
                }]
            }
        }
    }
    res = reference_resolver.resolve_reference(mlbom, allowed_signers={"ngc-catalog"})
    assert res.verified is True
    assert res.base_purl == "pkg:huggingface/org/base@rev"


def test_resolve_reference_no_ancestor():
    res = reference_resolver.resolve_reference({"component": {}}, allowed_signers={"ngc-catalog"})
    assert res.verified is False
    assert "no declared ancestor" in res.reason


def test_resolve_reference_untrusted_signer():
    mlbom = {
        "component": {"pedigree": {"ancestors": [{
            "purl": "pkg:huggingface/org/base@rev",
            "properties": [
                {"name": "securemodelgate:omsBundleDigest", "value": "a" * 64},
                {"name": "securemodelgate:signerIdentity", "value": "random-untrusted-uploader"},
            ],
        }]}}
    }
    res = reference_resolver.resolve_reference(mlbom, allowed_signers={"ngc-catalog"})
    assert res.verified is False


# ── intoto_attestation ───────────────────────────────────────────────────────

def test_attestation_round_trips_and_tamper_is_detected():
    signer = intoto_attestation.LocalHmacSigner(key_id="gate-key-1", secret=b"test-secret")
    envelope = intoto_attestation.issue_attestation(
        signer=signer,
        subject_name="model.safetensors",
        subject_sha256="b" * 64,
        declared_base={"purl": "pkg:huggingface/org/base@rev", "omsBundleDigest": "a" * 64, "signerIdentity": "ngc-catalog"},
        mlbom_digest="c" * 64,
        lineage={"S_w": 0.9, "threshold": 0.5, "alpha_L": 0.05, "calibrationSetDigest": "d" * 64},
        envelope={"minCKA": 0.95, "JS": 0.01, "threshold": 0.2, "alpha": 0.05, "n": 20},
        probe={"poolDigest": "e" * 64, "seedCommitment": commit_seed(123), "seed": 123},
        tier="1",
        deep_scan=None,
        verdict="ADMIT",
    )

    ok, payload = signer.verify(envelope)
    assert ok is True
    assert payload["predicateType"] == intoto_attestation.PREDICATE_TYPE
    assert payload["predicate"]["verdict"] == "ADMIT"

    tampered = dict(envelope)
    tampered_predicate = payload.copy()
    tampered_predicate["predicate"] = dict(payload["predicate"])
    tampered_predicate["predicate"]["verdict"] = "BLOCK"
    import base64, json
    tampered["payload"] = base64.b64encode(json.dumps(tampered_predicate, sort_keys=True).encode()).decode()

    ok2, reason = signer.verify(tampered)
    assert ok2 is False


# ── admission_policy ─────────────────────────────────────────────────────────

def _verdict(score, threshold, alpha, n, exceeds):
    return conformal.ConformalVerdict(threshold=threshold, alpha=alpha, n_calibration=n,
                                       score=score, exceeds_threshold=exceeds)


def test_route_tier0_trusted_publisher():
    d = admission_policy.route(trusted_publisher_signed=True, reference_verified=True)
    assert d.tier == admission_policy.TIER_0
    assert d.verdict == admission_policy.VERDICT_ADMIT


def test_route_tier2_no_verified_reference_pending_scan():
    d = admission_policy.route(trusted_publisher_signed=False, reference_verified=False)
    assert d.tier == admission_policy.TIER_2
    assert d.verdict == admission_policy.VERDICT_ESCALATE


def test_route_tier1_lineage_fails_blocks():
    lineage_v = _verdict(0.2, threshold=0.5, alpha=0.05, n=20, exceeds=False)
    envelope_v = _verdict(0.1, threshold=0.5, alpha=0.05, n=20, exceeds=False)
    d = admission_policy.route(trusted_publisher_signed=False, reference_verified=True,
                                lineage_verdict=lineage_v, envelope_verdict=envelope_v)
    assert d.verdict == admission_policy.VERDICT_BLOCK
    assert d.tier == admission_policy.TIER_1


def test_route_tier1_pass_both_admits():
    lineage_v = _verdict(0.9, threshold=0.5, alpha=0.05, n=20, exceeds=True)
    envelope_v = _verdict(0.1, threshold=0.5, alpha=0.05, n=20, exceeds=False)
    d = admission_policy.route(trusted_publisher_signed=False, reference_verified=True,
                                lineage_verdict=lineage_v, envelope_verdict=envelope_v)
    assert d.verdict == admission_policy.VERDICT_ADMIT
    assert d.tier == admission_policy.TIER_1


def test_route_tier1_envelope_fails_escalates_then_deep_scan_decides():
    lineage_v = _verdict(0.9, threshold=0.5, alpha=0.05, n=20, exceeds=True)
    envelope_v = _verdict(0.9, threshold=0.5, alpha=0.05, n=20, exceeds=True)
    escalated = admission_policy.route(trusted_publisher_signed=False, reference_verified=True,
                                        lineage_verdict=lineage_v, envelope_verdict=envelope_v)
    assert escalated.verdict == admission_policy.VERDICT_ESCALATE
    assert escalated.tier == admission_policy.TIER_2

    blocked = admission_policy.route(trusted_publisher_signed=False, reference_verified=True,
                                      lineage_verdict=lineage_v, envelope_verdict=envelope_v,
                                      deep_scan_flagged=True)
    assert blocked.verdict == admission_policy.VERDICT_BLOCK

    admitted = admission_policy.route(trusted_publisher_signed=False, reference_verified=True,
                                       lineage_verdict=lineage_v, envelope_verdict=envelope_v,
                                       deep_scan_flagged=False)
    assert admitted.verdict == admission_policy.VERDICT_ADMIT


def test_verdict_constants_are_three_distinct_values():
    """Pins down the invariant the module docstring's webhook-integration
    note depends on: ESCALATE must never collapse to (or be treated as)
    ADMIT. A Kubernetes ValidatingAdmissionWebhook call is synchronous
    with a bounded timeout that deep scan cannot finish inside, so
    ESCALATE has to map to `allowed: false` at that layer -- a future
    refactor that accidentally aliased ESCALATE to ADMIT (e.g. sharing a
    value to simplify a comparison) would silently defeat every Tier 2
    candidate's gate. Cheap to assert, expensive to debug if it regresses.
    """
    values = {admission_policy.VERDICT_ADMIT, admission_policy.VERDICT_ESCALATE, admission_policy.VERDICT_BLOCK}
    assert len(values) == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_widened_candidate_is_penalised_for_unexplained_channels():
    """A candidate that keeps every base channel verbatim but bolts on extra
    channels must not score 1.0: the extra channels are capacity no base
    channel explains. Each unmatched candidate row counts as zero similarity,
    so 8 base rows + 8 foreign rows scores about half."""
    rng = np.random.default_rng(7)
    w_base = rng.normal(size=(8, 16))
    widened = np.vstack([w_base, rng.normal(size=(8, 16))])
    assert channel_aligned_directional_similarity(widened, w_base) == pytest.approx(0.5, abs=0.02)
    assert channel_aligned_directional_similarity(w_base, w_base) == pytest.approx(1.0, abs=1e-9)
