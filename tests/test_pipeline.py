"""End-to-end tests for securemodelgate.pipeline against real (small)
torch models — the torch-dependent counterpart to test_lineage.py's
pure-math unit tests. Requires torch.
"""

import pytest

torch = pytest.importorskip("torch")

from securemodelgate import demo_models, digest, oms, pipeline
from securemodelgate.lineage.behavioral_delta import compute_behavioral_delta_from_models
from securemodelgate.lineage.conformal import lineage_test as _lineage_test
from securemodelgate.lineage.intoto_attestation import LocalHmacSigner
from securemodelgate.lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models
from securemodelgate.permutation_check import permute_layer_output_channels as _permute_layer_output_channels

SIGNER = LocalHmacSigner(key_id="test-key", secret=b"test-secret")
ALLOWED_SIGNERS = {"ngc-catalog-demo"}


@pytest.fixture(scope="module")
def base_model():
    return demo_models.make_base_model(seed=1)


@pytest.fixture(scope="module")
def probe_loader():
    return demo_models.make_probe_loader(seed=20260101)


@pytest.fixture(scope="module")
def mlbom(base_model):
    return oms.build_mlbom("pkg:huggingface/demo-org/base@v1", base_model, signer_identity="ngc-catalog-demo")


@pytest.fixture(scope="module")
def calibration(base_model, probe_loader):
    lineage_score_dicts = [
        weight_lineage_score_from_models(demo_models.make_independent_model(seed=100 + i), base_model)
        for i in range(5)
    ]
    lineage_scores = [d["S_w"] for d in lineage_score_dicts]
    layer_floor_scores = pooled_layer_scores(lineage_score_dicts)
    envelope_scores = []
    for i in range(5):
        benign = demo_models.make_benign_fine_tune(base_model, seed=200 + i)
        beh = compute_behavioral_delta_from_models(benign, base_model, probe_loader)
        envelope_scores.append(pipeline.envelope_anomaly_score(beh))
    return lineage_scores, envelope_scores, layer_floor_scores


def test_excluding_1d_parameters_widens_lineage_separation(base_model):
    """Regression test for a real bug caught by the evaluation pilot
    (see lineage_score.match_named_parameters's docstring): matching
    ALL named parameters (including BatchNorm affine weight/bias, which
    both start at PyTorch's default all-ones/all-zeros init and barely
    move under light training regardless of true lineage) collapsed
    S_w's true-derivative-vs-independent-model separation to ~0.002.
    Restricting to 2D+ weight matrices (what Section 3.2 actually
    means by W_l) should recover a separation close to the ~0.50 gap
    this test asserts a floor under.
    """
    descendant = demo_models.make_benign_fine_tune(base_model, seed=42)
    independent = demo_models.make_independent_model(seed=100)

    # No 1D (bias/BatchNorm) parameter should be scored.
    scored = weight_lineage_score_from_models(descendant, base_model)["layer_names"]
    one_d = {n for n, p in base_model.named_parameters() if p.dim() < 2}
    assert scored and not (set(scored) & one_d)

    s_w_descendant = weight_lineage_score_from_models(descendant, base_model)["S_w"]
    s_w_independent = weight_lineage_score_from_models(independent, base_model)["S_w"]

    assert s_w_descendant > 0.95
    assert s_w_independent < 0.7
    assert (s_w_descendant - s_w_independent) > 0.3   # vs. ~0.002 before the fix


def test_single_layer_channel_permutation_is_fully_recovered(base_model):
    """The bug this guards against: `channel_aligned_directional_similarity`
    (lineage_score.py) was added specifically because a plain cosine-of-
    flattened comparison collapsed under an exact, function-preserving
    output-channel permutation of a single layer. Confirm the fix
    actually holds against a real torch model, not just synthetic numpy
    arrays (test_lineage.py already covers the numpy core directly).
    """
    descendant = demo_models.make_benign_fine_tune(base_model, seed=42)
    permuted = _permute_layer_output_channels(
        descendant, "layer2", lambda m: m.layer3.conv, seed=7,
    )

    images, _ = demo_models.synthetic_batch(16, seed=999)
    descendant.eval(); permuted.eval()
    with torch.no_grad():
        max_diff = (descendant(images) - permuted(images)).abs().max().item()
    assert max_diff < 1e-4   # confirms the permutation really is function-preserving

    s_w_before = weight_lineage_score_from_models(descendant, base_model)["S_w"]
    s_w_after = weight_lineage_score_from_models(permuted, base_model)["S_w"]
    assert s_w_after == pytest.approx(s_w_before, abs=0.02)


def test_full_network_permutation_is_fully_recovered_by_sequential_alignment(base_model):
    """Permuting EVERY layer's output channels in sequence (each layer's own
    permutation cascading into the next layer's input-channel order) is
    function-preserving. Per-layer matching alone recovered only ~0.60,
    because each layer was compared against inputs its predecessor had
    scrambled. `weight_lineage_score_from_models` now carries each layer's
    channel correspondence into the next layer's input columns, which
    recovers the unpermuted score on this sequential architecture.
    """
    descendant = demo_models.make_benign_fine_tune(base_model, seed=42)

    fully_permuted = descendant
    fully_permuted = _permute_layer_output_channels(fully_permuted, "stem", lambda m: m.layer1.conv, seed=1)
    fully_permuted = _permute_layer_output_channels(fully_permuted, "layer1", lambda m: m.layer2.conv, seed=2)
    fully_permuted = _permute_layer_output_channels(fully_permuted, "layer2", lambda m: m.layer3.conv, seed=3)
    fully_permuted = _permute_layer_output_channels(fully_permuted, "layer3", lambda m: m.layer4.conv, seed=4)
    fully_permuted = _permute_layer_output_channels(fully_permuted, "layer4", lambda m: m.fc, seed=5)

    images, _ = demo_models.synthetic_batch(16, seed=999)
    descendant.eval(); fully_permuted.eval()
    with torch.no_grad():
        max_diff = (descendant(images) - fully_permuted(images)).abs().max().item()
    assert max_diff < 1e-4   # still function-preserving

    s_w_unpermuted = weight_lineage_score_from_models(descendant, base_model)["S_w"]
    s_w_fully_permuted = weight_lineage_score_from_models(fully_permuted, base_model)["S_w"]

    assert s_w_unpermuted > 0.95
    assert s_w_fully_permuted == pytest.approx(s_w_unpermuted, abs=0.02)   # ~0.60 before sequential alignment


def _fully_permute(model):
    from securemodelgate.permutation_check import _CASCADE
    permuted = model
    for i, (layer_name, next_getter) in enumerate(_CASCADE):
        permuted = _permute_layer_output_channels(permuted, layer_name, next_getter, seed=i + 1)
    return permuted


def test_fully_permuted_genuine_derivative_is_admitted(base_model, probe_loader, mlbom, calibration):
    """History, because this outcome has flipped twice: with per-layer
    matching only, a fully cascade-permuted genuine derivative scored S_w
    ~0.60 and was silently ADMITTED; the per-layer floor then BLOCKED it
    (its weakest layer fell below the floor) -- a false rejection of a
    benign model. Sequential alignment removes the cause rather than
    trading one failure for another: every layer now scores ~1.0, so the
    permuted derivative clears both the median and the floor.
    """
    descendant = demo_models.make_benign_fine_tune(base_model, seed=42)
    fully_permuted = _fully_permute(descendant)
    lineage_scores, envelope_scores, layer_floor_scores = calibration

    result = pipeline.run_admission(
        candidate=fully_permuted, candidate_subject_name="permuted.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )

    assert result.lineage_score["min_layer_score"] > 0.99
    assert result.lineage_verdict.exceeds_threshold is True
    assert result.decision.tier == "1"
    assert result.decision.verdict == "ADMIT"


def test_fully_permuted_independent_model_still_blocked(base_model, probe_loader, mlbom, calibration):
    """The other half of the permutation-robustness story: permuting an
    UNRELATED model does not manufacture a false lineage pass. Both the
    unpermuted and permuted independent model stay well under the
    lineage threshold and are correctly BLOCKed -- permutation cannot be
    used to forge ancestry, even though (see test above) it can cause a
    genuine derivative to be under-scored.
    """
    independent = demo_models.make_independent_model(seed=999)
    fully_permuted_independent = _fully_permute(independent)
    lineage_scores, envelope_scores, layer_floor_scores = calibration

    result = pipeline.run_admission(
        candidate=fully_permuted_independent, candidate_subject_name="permuted_indep.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )

    assert result.lineage_verdict.exceeds_threshold is False
    assert result.decision.tier == "1"
    assert result.decision.verdict == "BLOCK"


def test_permutation_does_not_change_envelope_score(base_model, probe_loader):
    """Channel permutation is an exact, function-preserving transform,
    so it must leave the behavioral/envelope test (CKA + JS on
    activations and logits) completely unaffected -- unlike S_w, which
    operates directly on raw weight tensors and is sensitive to channel
    order. This is the reason permutation cannot be used to evade
    backdoor detection even though it can degrade lineage-ancestry
    scoring: the two tests respond to permutation completely
    differently, by design of what each one measures.
    """
    backdoored = demo_models.make_backdoored_derivative(base_model, seed=42, trigger_type="patch")
    fully_permuted = _fully_permute(backdoored)

    score_unpermuted = pipeline.envelope_anomaly_score(
        compute_behavioral_delta_from_models(backdoored, base_model, probe_loader)
    )
    score_permuted = pipeline.envelope_anomaly_score(
        compute_behavioral_delta_from_models(fully_permuted, base_model, probe_loader)
    )

    assert score_permuted == pytest.approx(score_unpermuted, abs=1e-6)


def test_base_model_digest_mismatch_is_treated_as_unverified(base_model, probe_loader, mlbom):
    """Regression test for a real gap: `resolve_reference` only checks the
    CANDIDATE's own claim about its ancestor (signer allow-listed,
    well-formed digest) -- it has no way to know whether the `base_model`
    object the caller passes to run_admission for scoring is actually the
    artifact that digest names. Before the fix, run_admission would
    silently score a candidate against ANY base_model handed to it and
    still report resolution.verified=True and a matching declaredBase in
    the attestation, even when that base_model's real digest didn't match
    the candidate's declared ancestor digest at all -- e.g. because a
    purl-lookup step outside this pipeline returned the wrong artifact
    (staleness, cache poisoning, a compromised registry). This pins down
    the fix: a digest mismatch must fail closed to Tier 2, not silently
    proceed to a scoring run whose "verified" premise doesn't hold.
    """
    wrong_base = demo_models.make_independent_model(seed=777)
    assert digest.weight_digest(wrong_base) != digest.weight_digest(base_model)

    candidate = demo_models.make_benign_fine_tune(wrong_base, seed=42)  # genuine derivative of wrong_base

    result = pipeline.run_admission(
        candidate=candidate, candidate_subject_name="candidate.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=wrong_base, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=[0.5], envelope_calibration_scores=[0.1],
    )

    assert result.resolution.verified is False
    assert "does not match" in result.resolution.reason
    assert result.decision.tier == "2"
    # lineage/behavioral scoring must not have run against the mismatched base
    assert result.lineage_score is None
    assert result.behavioral is None


def test_tier0_trusted_publisher_short_circuits(base_model, mlbom):
    result = pipeline.run_admission(
        candidate=base_model, candidate_subject_name="base.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER, trusted_publisher_signed=True,
    )
    assert result.decision.tier == "0"
    assert result.decision.verdict == "ADMIT"
    assert result.lineage_score is None  # no scoring needed for a trusted publisher


def test_tier2_no_declared_ancestor(base_model, probe_loader):
    result = pipeline.run_admission(
        candidate=demo_models.make_independent_model(seed=7),
        candidate_subject_name="unsigned.pt", mlbom=oms.build_unsigned_mlbom(),
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER, deep_scan_flagged=False,
    )
    assert result.decision.tier == "2"
    assert result.decision.verdict == "ADMIT"


def test_benign_fine_tune_admitted_at_tier1(base_model, mlbom, probe_loader, calibration):
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    candidate = demo_models.make_benign_fine_tune(base_model, seed=42)
    result = pipeline.run_admission(
        candidate=candidate, candidate_subject_name="ft.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )
    assert result.decision.tier == "1"
    assert result.decision.verdict == "ADMIT"
    assert result.lineage_verdict.exceeds_threshold is True
    assert result.envelope_verdict.exceeds_threshold is False


def test_lineage_forged_model_is_blocked(base_model, mlbom, probe_loader, calibration):
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    forged = demo_models.make_independent_model(seed=55)
    result = pipeline.run_admission(
        candidate=forged, candidate_subject_name="forged.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )
    assert result.decision.tier == "1"
    assert result.decision.verdict == "BLOCK"
    assert result.lineage_verdict.exceeds_threshold is False


def test_poisoned_derivative_escalates_on_envelope_test(base_model, mlbom, probe_loader, calibration):
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    poisoned = demo_models.make_backdoored_derivative(base_model, seed=99)
    result = pipeline.run_admission(
        candidate=poisoned, candidate_subject_name="poisoned.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )
    # Genuine fine-tune of the base -> lineage should still verify ...
    assert result.lineage_verdict.exceeds_threshold is True
    # ... but the behavioral envelope test should catch the poisoning.
    assert result.envelope_verdict.exceeds_threshold is True
    assert result.decision.tier == "2"
    assert result.decision.verdict == "ESCALATE"
    assert demo_models.attack_success_rate(poisoned, seed=99) > 0.5


def test_attestation_signature_verifies_and_matches_decision(base_model, mlbom, probe_loader, calibration):
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    candidate = demo_models.make_benign_fine_tune(base_model, seed=42)
    result = pipeline.run_admission(
        candidate=candidate, candidate_subject_name="ft.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )
    ok, payload = SIGNER.verify(result.attestation)
    assert ok is True
    assert payload["predicate"]["verdict"] == result.decision.verdict
    assert payload["subject"][0]["name"] == "ft.pt"


def test_verified_reference_without_base_model_raises(mlbom):
    with pytest.raises(ValueError):
        pipeline.run_admission(
            candidate=demo_models.make_independent_model(seed=1),
            candidate_subject_name="x.pt", mlbom=mlbom,
            allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        )


def test_verified_reference_without_layer_floor_calibration_raises(base_model, mlbom, probe_loader):
    """The per-layer floor check exists specifically to close a real,
    demonstrated evasion (see test_layer_splicing_attack_is_blocked_below).
    A caller that forgets to pass lineage_layer_floor_calibration_scores
    must get a loud error, not a silent skip that quietly ships a gate
    vulnerable to splicing -- matching this module's existing philosophy
    for the other two required calibration sets.
    """
    with pytest.raises(ValueError, match="lineage_layer_floor_calibration_scores"):
        pipeline.run_admission(
            candidate=demo_models.make_benign_fine_tune(base_model, seed=42),
            candidate_subject_name="ft.pt", mlbom=mlbom,
            allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
            base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
            lineage_calibration_scores=[0.6], envelope_calibration_scores=[0.1],
        )


def test_layer_splicing_attack_is_blocked_by_layer_floor(base_model, probe_loader, mlbom, calibration):
    """The finding that motivated the whole per-layer floor check.

    S_w's median-over-matched-layers aggregation is deliberately tolerant
    of a MINORITY of layers looking unrelated (so a few heavily
    pruned/re-initialized layers in a genuine derivative don't tank the
    verdict -- see weight_lineage_score's docstring). That tolerance is
    exploitable: verbatim-copy just over half the matched layers
    (stem/layer1/layer2) from the real base and swap the rest
    (layer3/layer4/fc) for an independently-trained -- here,
    backdoored -- block, and the median alone clears the lineage
    threshold with room to spare (empirically ~0.75-0.81 against a
    ~0.59-0.65 threshold across five separate donor/attack seeds).
    Combined with training the whole spliced model against the
    differentiable envelope penalty (adaptive_attack.py's A4 machinery),
    this evaded BOTH tests before the floor check existed: lineage
    verified, envelope score stayed under threshold, 100% attack success
    rate, Tier 1 ADMIT. This test pins down the median-alone vulnerability
    and confirms the floor check now catches it on lineage grounds alone
    (no need for the compound adaptive-envelope-training half to make the
    point here; see 03_change_log.md's second-revision-pass entry for the
    full compound-attack numbers, which were reproduced by hand rather
    than committed as a test given the ~80-SGD-step training cost).
    """
    donor = demo_models.make_independent_model(seed=555)
    spliced = demo_models.make_benign_fine_tune(base_model, seed=42)  # start from a genuine derivative
    spliced.layer3.load_state_dict(donor.layer3.state_dict())
    spliced.layer4.load_state_dict(donor.layer4.state_dict())
    spliced.fc.load_state_dict(donor.fc.state_dict())

    median_only_score = weight_lineage_score_from_models(spliced, base_model)
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    median_only_verdict = _lineage_test(median_only_score["S_w"], lineage_scores, alpha_l=0.05)
    assert median_only_verdict.exceeds_threshold is True   # the median alone is fooled

    result = pipeline.run_admission(
        candidate=spliced, candidate_subject_name="spliced.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores,
    )

    # The combined check (median AND per-layer floor) is not fooled.
    assert result.lineage_verdict.exceeds_threshold is False
    assert result.layer_floor_verdict.exceeds_threshold is False


def _run(candidate, base_model, probe_loader, mlbom, calibration, **extra):
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    return pipeline.run_admission(
        candidate=candidate, candidate_subject_name="c.pt", mlbom=mlbom,
        allowed_signers=ALLOWED_SIGNERS, signer=SIGNER,
        base_model=base_model, probe_loader=probe_loader, probe_seed=20260101,
        lineage_calibration_scores=lineage_scores, envelope_calibration_scores=envelope_scores,
        lineage_layer_floor_calibration_scores=layer_floor_scores, **extra,
    )


def _with_replaced(mlbom, names):
    import copy
    out = copy.deepcopy(mlbom)
    out["component"]["properties"] = [{"name": "securemodelgate:replacedParameters", "value": ",".join(names)}]
    return out


def test_reshaped_foreign_layers_are_scored_not_skipped(base_model, probe_loader, mlbom, calibration):
    """Regression test for a real loophole: the scorer used to skip any
    weight matrix whose shape differed from the base. Widening a spliced
    model's foreign tail by one channel (layer3/layer4/fc) therefore left
    only the 3 verbatim-copied matrices scored -- S_w=1.0, min=1.0, per-layer
    floor never consulted -- and combined with adaptive training against
    the envelope score, a 100%-ASR backdoor was admitted at Tier 1 on 2 of
    3 seeds (reproduced by hand; `10_critical_review_pilot.md` item 7).
    Every matrix must now be scored (rectangular matching + sequential
    alignment) and the foreign ones must sink the per-layer floor.
    """
    import copy
    import torch.nn as nn
    from securemodelgate.demo_models import _ConvBlock

    torch.manual_seed(0)
    reshaped = copy.deepcopy(base_model)
    reshaped.layer3 = _ConvBlock(16, 33, stride=2)
    reshaped.layer4 = _ConvBlock(33, 65, stride=2)
    reshaped.fc = nn.Linear(65, 4)

    result = _run(reshaped, base_model, probe_loader, mlbom, calibration)

    assert result.lineage_score["n_layers"] == 6
    assert result.lineage_score["exempt"] == []
    assert result.lineage_verdict.exceeds_threshold is False
    assert result.decision.verdict == "BLOCK"


def test_added_weight_matrices_count_against_the_exemption_cap(base_model, probe_loader, mlbom, calibration):
    """Matrices with no base counterpart are unscorable, so they are foreign
    capacity the lineage score cannot vouch for. They must count against
    max_exempt like any other unscored matrix, not be silently ignored."""
    import copy
    import torch.nn as nn

    grown = copy.deepcopy(demo_models.make_benign_fine_tune(base_model, seed=42))
    grown.extra_a = nn.Linear(64, 64)
    grown.extra_b = nn.Linear(64, 64)

    result = _run(grown, base_model, probe_loader, mlbom, calibration)

    reasons = [e["reason"] for e in result.lineage_score["exempt"]]
    assert reasons.count("added") == 2
    assert result.lineage_verdict.exceeds_threshold is False
    assert "max_exempt" in result.decision.reason


def test_declared_head_replacement_is_exempt_but_capped(base_model, probe_loader, mlbom, calibration):
    """Ordinary transfer learning re-initialises the classification head,
    which then scores like an unrelated layer and fails the per-layer floor.
    Declaring it in the ML-BOM exempts it -- but the declaration is the
    candidate's own claim, so it is capped (default 1) and recorded in the
    attestation, and declaring the whole foreign tail of a splice fails."""
    import copy

    tl = copy.deepcopy(base_model)
    tl.fc.reset_parameters()
    tl = demo_models._train_steps(tl, seed=77, steps=8, lr=0.005)

    undeclared = _run(tl, base_model, probe_loader, mlbom, calibration)
    assert undeclared.lineage_verdict.exceeds_threshold is False

    declared = _run(tl, base_model, probe_loader, _with_replaced(mlbom, ["fc.weight"]), calibration)
    assert declared.lineage_verdict.exceeds_threshold is True
    assert declared.lineage_score["exempt"] == [{"name": "fc.weight", "reason": "declared_replaced"}]
    _, payload = SIGNER.verify(declared.attestation)
    assert payload["predicate"]["lineage"]["exempt"] == [{"name": "fc.weight", "reason": "declared_replaced"}]

    too_many = _run(tl, base_model, probe_loader,
                    _with_replaced(mlbom, ["layer3.conv.weight", "layer4.conv.weight", "fc.weight"]), calibration)
    assert too_many.lineage_verdict.exceeds_threshold is False
    assert too_many.decision.verdict == "BLOCK"


def test_attestation_binds_gate_version_and_every_calibration_set(base_model, probe_loader, mlbom, calibration):
    """`gate_version` used to be accepted by run_admission and then dropped
    (the predicate always said 0.1.0), and only the lineage calibration set
    was digested. Every threshold in the predicate must be traceable to the
    exact calibration data that produced it."""
    result = _run(demo_models.make_benign_fine_tune(base_model, seed=42), base_model, probe_loader, mlbom,
                  calibration, gate_version="9.9.9")
    _, payload = SIGNER.verify(result.attestation)
    pred = payload["predicate"]
    assert pred["gateVersion"] == "9.9.9"
    lineage_scores, envelope_scores, layer_floor_scores = calibration
    assert pred["lineage"]["calibrationSetDigest"] == digest.json_digest(list(lineage_scores))
    assert pred["lineage"]["layerFloorCalibrationSetDigest"] == digest.json_digest(list(layer_floor_scores))
    assert pred["envelope"]["calibrationSetDigest"] == digest.json_digest(list(envelope_scores))
