"""Regression tests for the gaps the developer checklist
(paper/techcon2027_revision/13_developer_checklist_answers.md) found:
verification that re-derives thresholds, verdicts and scores, freshness
and trusted-base checks, the attestation-driven webhook path, deep-scan
binding, signer substitution, calibration/base binding, scorer input
guards, and the untrusted-file loader. Requires torch."""

import base64
import copy
import json
import pickle

import pytest

torch = pytest.importorskip("torch")

from securemodelgate import demo_models, digest, oms, pipeline
from securemodelgate.lineage.behavioral_delta import compute_behavioral_delta_from_models
from securemodelgate.lineage.intoto_attestation import LocalHmacSigner
from securemodelgate.lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models
from securemodelgate.loader import UnsafeModelFile, load_candidate, load_state_dict
from securemodelgate.verify import verify_attestation
from securemodelgate.webhook import review_attestation

SIGNER = LocalHmacSigner(key_id="k", secret=b"s")
SIGNERS = {"ngc"}


@pytest.fixture(scope="module")
def world():
    base = demo_models.make_base_model(seed=1)
    probe = demo_models.make_probe_loader(seed=20260101)
    dicts = [weight_lineage_score_from_models(demo_models.make_independent_model(seed=100 + i), base) for i in range(5)]
    cal = dict(
        lineage_calibration_scores=[d["S_w"] for d in dicts],
        layer_floor_calibration_scores=pooled_layer_scores(dicts),
        envelope_calibration_scores=[
            pipeline.envelope_anomaly_score(compute_behavioral_delta_from_models(
                demo_models.make_benign_fine_tune(base, seed=200 + i), base, probe)) for i in range(5)],
    )
    mlbom = oms.build_mlbom("pkg:huggingface/org/base@v1", base, signer_identity="ngc")
    return dict(base=base, probe=probe, cal=cal, mlbom=mlbom)


def _admit(w, candidate, mlbom=None, **extra):
    c = w["cal"]
    return pipeline.run_admission(
        candidate=candidate, candidate_subject_name="c.pt", mlbom=mlbom or w["mlbom"], allowed_signers=SIGNERS,
        signer=SIGNER, base_model=w["base"], probe_loader=w["probe"], probe_seed=20260101,
        lineage_calibration_scores=c["lineage_calibration_scores"],
        envelope_calibration_scores=c["envelope_calibration_scores"],
        lineage_layer_floor_calibration_scores=c["layer_floor_calibration_scores"], **extra)


@pytest.fixture(scope="module")
def admitted(world):
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    r = _admit(world, cand)
    assert r.decision.verdict == "ADMIT"
    return dict(envelope=r.attestation, candidate=cand)


def _kw(world, admitted, **over):
    kw = dict(candidate=admitted["candidate"], mlbom=world["mlbom"], probe_loader=world["probe"], **world["cal"])
    kw.update(over)
    return kw


def _resign(envelope, mutate):
    stmt = json.loads(base64.b64decode(envelope["payload"]))
    mutate(stmt["predicate"])
    return SIGNER.sign(stmt)


# ---------------------------------------------------------------- verification

def test_verification_recomputes_scores_from_the_base(world, admitted):
    ok, reason = verify_attestation(admitted["envelope"], SIGNER, base_model=world["base"], **_kw(world, admitted))
    assert ok and "recomputed" in reason, reason


@pytest.mark.parametrize("mutate,needs_base", [
    (lambda p: p["lineage"].__setitem__("threshold", p["lineage"]["threshold"] - 0.01), False),
    (lambda p: p["envelope"].__setitem__("threshold", 10.0), False),
    (lambda p: p["lineage"].__setitem__("S_w", 0.0), False),            # verdict no longer follows
    (lambda p: p.__setitem__("verdict", "ADMIT") or p.__setitem__("tier", "0"), False),
    (lambda p: p["lineage"].__setitem__("S_w", p["lineage"]["S_w"] - 1e-3), True),   # consistent, but false
    (lambda p: p["envelope"].__setitem__("JS", 0.0), True),
])
def test_a_validly_signed_but_inconsistent_predicate_is_rejected(world, admitted, mutate, needs_base):
    """A gate bug or a compromised signing key can produce a correctly
    signed predicate whose numbers do not hold; re-derivation catches it."""
    env = _resign(admitted["envelope"], mutate)
    extra = {"base_model": world["base"]} if needs_base else {}
    ok, reason = verify_attestation(env, SIGNER, **_kw(world, admitted), **extra)
    assert not ok, reason


def test_stale_attestation_is_rejected_when_freshness_is_required(world, admitted):
    import time
    ok, _ = verify_attestation(admitted["envelope"], SIGNER, max_age_seconds=3600, **_kw(world, admitted))
    assert ok
    ok, reason = verify_attestation(admitted["envelope"], SIGNER, max_age_seconds=3600,
                                    now=time.time() + 7200, **_kw(world, admitted))
    assert not ok and "old" in reason


def test_attestation_under_a_superseded_policy_is_rejected(world, admitted):
    ok, _ = verify_attestation(admitted["envelope"], SIGNER, **_kw(world, admitted),
                               expected_policy={"alpha_L": 0.05, "alpha": 0.05, "maxExempt": 1})
    assert ok
    ok, reason = verify_attestation(admitted["envelope"], SIGNER, **_kw(world, admitted),
                                    expected_policy={"alpha_L": 0.01, "alpha": 0.05, "maxExempt": 1})
    assert not ok and "policy" in reason


def test_revoked_base_fails_verification(world, admitted):
    good = {digest.weight_digest(world["base"]): "ngc"}
    assert verify_attestation(admitted["envelope"], SIGNER, trusted_bases=good, **_kw(world, admitted))[0]
    ok, reason = verify_attestation(admitted["envelope"], SIGNER, trusted_bases={}, **_kw(world, admitted))
    assert not ok and "revoked" in reason


@pytest.mark.parametrize("envelope", [
    {}, {"payload": "!!!"}, {"payload": base64.b64encode(b"not json").decode(), "signatures": []},
])
def test_malformed_envelopes_fail_closed_without_raising(world, admitted, envelope):
    ok, _ = verify_attestation(envelope, SIGNER, **_kw(world, admitted))
    assert not ok


def test_signed_payload_of_the_wrong_type_is_rejected(world, admitted):
    stmt = json.loads(base64.b64decode(admitted["envelope"]["payload"]))
    stmt["predicateType"] = "https://slsa.dev/provenance/v1"
    ok, reason = verify_attestation(SIGNER.sign(stmt), SIGNER, **_kw(world, admitted))
    assert not ok and "type" in reason


# ---------------------------------------------------------------- webhook from attestation

def test_webhook_allows_only_a_verified_admit_attestation(world, admitted):
    resp = review_attestation("u", admitted["envelope"], SIGNER, **_kw(world, admitted))["response"]
    assert resp["allowed"] is True

    blocked = _admit(world, demo_models.make_independent_model(seed=55))
    assert blocked.decision.verdict == "BLOCK"
    kw = _kw(world, admitted, candidate=demo_models.make_independent_model(seed=55))
    resp = review_attestation("u", blocked.attestation, SIGNER, **kw)["response"]
    assert resp["allowed"] is False and "BLOCK" in resp["status"]["message"]

    resp = review_attestation("u", {"garbage": True}, SIGNER, **_kw(world, admitted))["response"]
    assert resp["allowed"] is False

    other = demo_models.make_benign_fine_tune(world["base"], seed=43)
    resp = review_attestation("u", admitted["envelope"], SIGNER, **_kw(world, admitted, candidate=other))["response"]
    assert resp["allowed"] is False


# ---------------------------------------------------------------- deep scan, signer, calibration binding

def test_deep_scan_report_must_be_about_the_candidates_bytes(world):
    cand = demo_models.make_independent_model(seed=7)
    unsigned = oms.build_unsigned_mlbom()
    other = digest.weight_digest(demo_models.make_independent_model(seed=8))
    r = pipeline.run_admission(candidate=cand, candidate_subject_name="c", mlbom=unsigned, allowed_signers=SIGNERS,
                               signer=SIGNER, deep_scan_report={"tool": "x", "version": "1", "flagged": False,
                                                                "subjectDigest": other})
    assert r.decision.verdict == "ESCALATE"
    r = pipeline.run_admission(candidate=cand, candidate_subject_name="c", mlbom=unsigned, allowed_signers=SIGNERS,
                               signer=SIGNER, deep_scan_report={"tool": "x", "version": "1", "flagged": False,
                                                                "subjectDigest": digest.weight_digest(cand)})
    assert r.decision.verdict == "ADMIT"
    pred = json.loads(base64.b64decode(r.attestation["payload"]))["predicate"]
    assert pred["deepScan"]["bound"] is True and pred["deepScan"]["subjectDigest"] == digest.weight_digest(cand)


def test_signer_substitution_is_caught_by_the_trusted_base_registry(world):
    """Without a registry, the structural OMS stand-in accepts any allowed
    signer name over any digest; the registry pins digest -> signer."""
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    d = digest.weight_digest(world["base"])
    assert _admit(world, cand, trusted_bases={d: "ngc"}).decision.tier == "1"
    sub = _admit(world, cand, trusted_bases={d: "some-other-publisher"})
    assert sub.decision.tier == "2" and "substitution" in sub.resolution.reason
    revoked = _admit(world, cand, trusted_bases={})
    assert revoked.decision.tier == "2"


def test_calibration_computed_for_another_base_is_refused(world):
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    with pytest.raises(ValueError, match="calibration sets were computed against"):
        _admit(world, cand, calibration_base_digest="0" * 64)
    r = _admit(world, cand, calibration_base_digest=digest.weight_digest(world["base"]))
    assert r.decision.verdict == "ADMIT"


def test_policy_can_restrict_which_matrix_may_be_declared_replaced(world):
    """The declaration is the candidate's own claim; without a policy any one
    matrix can be declared. With `replaceable`, only the listed ones can."""
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    donor = demo_models.make_independent_model(seed=56)
    with torch.no_grad():
        cand.layer3.conv.weight.copy_(donor.layer3.conv.weight)
    mlbom = copy.deepcopy(world["mlbom"])
    mlbom["component"]["properties"] = [{"name": "securemodelgate:replacedParameters", "value": "layer3.conv.weight"}]
    assert _admit(world, cand, mlbom=mlbom).lineage_verdict.exceeds_threshold          # any one matrix, by default
    r = _admit(world, cand, mlbom=mlbom, replaceable={"fc.weight"})
    assert r.decision.verdict == "BLOCK" and "policy only allows" in r.decision.reason


# ---------------------------------------------------------------- scorer input guards

def test_grossly_widened_matrix_is_incomparable_not_matched(world):
    """The matching cost matrix is candidate rows x base rows; an unbounded
    widening would let a candidate make the gate allocate arbitrarily."""
    import torch.nn as nn
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    cand.fc = nn.Linear(64, 4 * 50)
    s = weight_lineage_score_from_models(cand, world["base"])
    assert {"name": "fc.weight", "reason": "incomparable"} in s["exempt"]


def test_matrices_above_the_matching_limit_are_compared_by_index(monkeypatch):
    """A vocabulary-sized matrix would need a multi-GB matching cost matrix;
    above MAX_MATCH_ROWS rows are compared by index (token id) instead."""
    import torch.nn as nn
    from securemodelgate.lineage import lineage_score as ls
    monkeypatch.setattr(ls, "MAX_MATCH_ROWS", 16)
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(8, 40))
    copy_ = copy.deepcopy(base)
    s = ls.weight_lineage_score_from_models(copy_, base)
    assert s["identity_aligned"] == ["0.weight"] and s["S_w"] == pytest.approx(1.0)
    shuffled = copy.deepcopy(base)
    with torch.no_grad():
        shuffled[0].weight.copy_(base[0].weight[torch.randperm(40)])
    assert ls.weight_lineage_score_from_models(shuffled, base)["S_w"] < 0.5   # by design: not permutation-invariant
    narrower = nn.Sequential(nn.Linear(8, 39))
    assert ls.weight_lineage_score_from_models(narrower, base)["exempt"][0]["reason"] == "incomparable"


def test_non_finite_candidate_weights_raise(world):
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    with torch.no_grad():
        cand.layer2.conv.weight[3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        weight_lineage_score_from_models(cand, world["base"])


def test_renamed_parameter_counts_as_missing_and_added(world):
    import torch.nn as nn
    cand = demo_models.make_benign_fine_tune(world["base"], seed=42)
    cand.head = cand.fc
    del cand.fc
    s = weight_lineage_score_from_models(cand, world["base"])
    reasons = sorted(e["reason"] for e in s["exempt"])
    assert reasons == ["added", "missing"]


# ---------------------------------------------------------------- untrusted-file loading

class _Exploit:
    def __reduce__(self):
        import os
        return (os.system, ("touch PWNED_MARKER",))


def test_malicious_pickle_is_refused_and_not_executed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.save({"w": _Exploit()}, tmp_path / "model.pt")                  # torch zip checkpoint
    (tmp_path / "model.bin").write_bytes(pickle.dumps({"w": _Exploit()}))  # raw pickle
    for name in ("model.pt", "model.bin"):
        with pytest.raises(UnsafeModelFile):
            load_state_dict(tmp_path / name)
        assert not (tmp_path / "PWNED_MARKER").exists()
    # Positive control: the payload is live. A plain torch.load runs it --
    # for the raw pickle even though the load then fails on the magic number.
    for name in ("model.pt", "model.bin"):
        try:
            torch.load(tmp_path / name, weights_only=False)
        except Exception:
            pass
        assert (tmp_path / "PWNED_MARKER").exists()
        (tmp_path / "PWNED_MARKER").unlink()
    (tmp_path / "model.pkl").write_bytes(pickle.dumps({"w": 1}))
    with pytest.raises(UnsafeModelFile, match="not accepted"):
        load_state_dict(tmp_path / "model.pkl")


def test_safetensors_round_trip_and_strict_architecture(tmp_path):
    from safetensors.torch import save_file
    base = demo_models.TinyConvNet()
    sd = {k: v.contiguous() for k, v in base.state_dict().items()}
    save_file(sd, str(tmp_path / "m.safetensors"))
    loaded = load_candidate(tmp_path / "m.safetensors", demo_models.TinyConvNet)
    assert digest.weight_digest(loaded) == digest.weight_digest(base)

    sd_bad = dict(sd)
    sd_bad["fc.weight"] = torch.zeros(5, 64)
    save_file(sd_bad, str(tmp_path / "bad.safetensors"))
    with pytest.raises(UnsafeModelFile, match="architecture"):
        load_candidate(tmp_path / "bad.safetensors", demo_models.TinyConvNet)

    sd_nan = dict(sd)
    sd_nan["fc.weight"] = torch.full((4, 64), float("nan"))
    save_file(sd_nan, str(tmp_path / "nan.safetensors"))
    with pytest.raises(UnsafeModelFile, match="NaN"):
        load_state_dict(tmp_path / "nan.safetensors")

    (tmp_path / "link.safetensors").symlink_to(tmp_path / "m.safetensors")
    with pytest.raises(UnsafeModelFile, match="symlink"):
        load_state_dict(tmp_path / "link.safetensors")
