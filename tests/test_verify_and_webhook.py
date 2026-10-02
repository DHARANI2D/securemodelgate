"""Tamper tests for admission-time attestation verification, and the
verdict -> AdmissionReview mapping. Requires torch."""

import copy

import pytest

torch = pytest.importorskip("torch")

from securemodelgate import demo_models, oms, pipeline
from securemodelgate.lineage.admission_policy import AdmissionDecision
from securemodelgate.lineage.behavioral_delta import compute_behavioral_delta_from_models
from securemodelgate.lineage.intoto_attestation import LocalHmacSigner
from securemodelgate.lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models
from securemodelgate.verify import verify_attestation
from securemodelgate.webhook import admission_response

SIGNER = LocalHmacSigner(key_id="k", secret=b"s")


@pytest.fixture(scope="module")
def admitted():
    base = demo_models.make_base_model(seed=1)
    probe = demo_models.make_probe_loader(seed=20260101)
    dicts = [weight_lineage_score_from_models(demo_models.make_independent_model(seed=100 + i), base) for i in range(5)]
    cal = {
        "lineage_calibration_scores": [d["S_w"] for d in dicts],
        "layer_floor_calibration_scores": pooled_layer_scores(dicts),
        "envelope_calibration_scores": [
            pipeline.envelope_anomaly_score(compute_behavioral_delta_from_models(
                demo_models.make_benign_fine_tune(base, seed=200 + i), base, probe))
            for i in range(5)
        ],
    }
    mlbom = oms.build_mlbom("pkg:huggingface/org/base@v1", base, signer_identity="ngc")
    candidate = demo_models.make_benign_fine_tune(base, seed=42)
    result = pipeline.run_admission(
        candidate=candidate, candidate_subject_name="ft.pt", mlbom=mlbom, allowed_signers={"ngc"},
        signer=SIGNER, base_model=base, probe_loader=probe, probe_seed=20260101,
        lineage_calibration_scores=cal["lineage_calibration_scores"],
        envelope_calibration_scores=cal["envelope_calibration_scores"],
        lineage_layer_floor_calibration_scores=cal["layer_floor_calibration_scores"],
        gate_version="0.2.0",
    )
    assert result.decision.verdict == "ADMIT"
    return dict(envelope=result.attestation, candidate=candidate, mlbom=mlbom, probe_loader=probe,
                expected_gate_version="0.2.0", **cal)


def _verify(a, **overrides):
    args = {**a, **overrides}
    return verify_attestation(args.pop("envelope"), SIGNER, **args)


def test_untampered_attestation_verifies(admitted):
    ok, reason = _verify(admitted)
    assert ok, reason


@pytest.mark.parametrize("field,tamper", [
    ("candidate", lambda a: demo_models.make_benign_fine_tune(demo_models.make_base_model(seed=1), seed=43)),
    ("mlbom", lambda a: {**copy.deepcopy(a["mlbom"]), "extra": 1}),
    ("lineage_calibration_scores", lambda a: list(a["lineage_calibration_scores"])[:-1]),
    ("layer_floor_calibration_scores", lambda a: [s + 1e-9 for s in a["layer_floor_calibration_scores"]]),
    ("envelope_calibration_scores", lambda a: list(a["envelope_calibration_scores"]) + [0.0]),
    ("probe_loader", lambda a: demo_models.make_probe_loader(seed=1)),
    ("expected_gate_version", lambda a: "0.1.0"),
])
def test_any_single_tampered_input_fails_closed(admitted, field, tamper):
    ok, reason = _verify(admitted, **{field: tamper(admitted)})
    assert not ok, f"tampering with {field} was not detected"


def test_tampered_signature_fails(admitted):
    env = copy.deepcopy(admitted["envelope"])
    env["signatures"][0]["sig"] = "0" * 64
    ok, _ = _verify(admitted, envelope=env)
    assert not ok


def test_tier1_evidence_cannot_be_verified_without_its_inputs(admitted):
    ok, reason = _verify(admitted, probe_loader=None)
    assert not ok and "probe_loader" in reason


@pytest.mark.parametrize("verdict,allowed", [
    ("ADMIT", True), ("ESCALATE", False), ("BLOCK", False), ("SOMETHING_NEW", False),
])
def test_only_admit_is_allowed_at_the_webhook(verdict, allowed):
    resp = admission_response("uid-1", AdmissionDecision("2", verdict, "r"))["response"]
    assert resp["allowed"] is allowed
    assert resp["uid"] == "uid-1"
    assert verdict in resp["status"]["message"]
