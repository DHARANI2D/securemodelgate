"""End-to-end attacks SecureModelGate should stop, each run through the
entry points a deployment uses -- `webhook.review_candidate` (run the gate,
deny on any error) and `webhook.review_attestation` (verify a presented
attestation, allow only a verified ADMIT) -- and each first showing that
the attack is real (high attack success rate, or changed predictions).

The last test pins a KNOWN GAP rather than a defence: an adaptive attacker
who trains a poisoned fine-tune against the envelope score is admitted. If
it ever starts failing, the gap has closed and the paper's limitation
section must change with it. Requires torch."""

import base64
import copy
import json

import pytest

torch = pytest.importorskip("torch")

from securemodelgate import demo_models, oms, pipeline
from securemodelgate.defensibility_checks import train_adaptive
from securemodelgate.lineage.behavioral_delta import compute_behavioral_delta_from_models
from securemodelgate.lineage.intoto_attestation import LocalHmacSigner
from securemodelgate.lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models
from securemodelgate.webhook import review_attestation, review_candidate

SIGNER = LocalHmacSigner(key_id="gate", secret=b"gate-secret")


@pytest.fixture(scope="module")
def gate():
    base = demo_models.make_base_model(seed=1)
    probe = demo_models.make_probe_loader(seed=20260101)
    dicts = [weight_lineage_score_from_models(demo_models.make_independent_model(seed=1000 + i), base)
             for i in range(20)]
    env = [pipeline.envelope_anomaly_score(compute_behavioral_delta_from_models(
        demo_models.make_benign_fine_tune(base, seed=2000 + i), base, probe)) for i in range(20)]
    cal = dict(lineage_calibration_scores=[d["S_w"] for d in dicts],
               lineage_layer_floor_calibration_scores=pooled_layer_scores(dicts),
               envelope_calibration_scores=env)
    mlbom = oms.build_mlbom("pkg:huggingface/org/base@v1", base, signer_identity="ngc")
    return dict(base=base, probe=probe, cal=cal, mlbom=mlbom)


def _review(gate, candidate, mlbom=None):
    resp, result = review_candidate(
        "uid", candidate=candidate, candidate_subject_name="candidate.safetensors",
        mlbom=gate["mlbom"] if mlbom is None else mlbom, allowed_signers={"ngc"}, signer=SIGNER,
        base_model=gate["base"], probe_loader=gate["probe"], probe_seed=20260101, **gate["cal"])
    return resp["response"], result


def _present(gate, envelope, candidate):
    c = gate["cal"]
    return review_attestation(
        "uid", envelope, SIGNER, candidate=candidate, mlbom=gate["mlbom"], probe_loader=gate["probe"],
        lineage_calibration_scores=c["lineage_calibration_scores"],
        layer_floor_calibration_scores=c["lineage_layer_floor_calibration_scores"],
        envelope_calibration_scores=c["envelope_calibration_scores"], base_model=gate["base"])["response"]


@pytest.fixture(scope="module")
def admitted(gate):
    benign = demo_models.make_benign_fine_tune(gate["base"], seed=42)
    resp, result = _review(gate, benign)
    assert resp["allowed"], resp["status"]["message"]
    return benign, result.attestation


def test_attack_1_poisoned_fine_tune_declaring_its_real_base_is_denied(gate):
    """A2: a genuine fine-tune of the base with a patch trigger."""
    bd = demo_models.make_backdoored_derivative(gate["base"], seed=99)
    assert demo_models.attack_success_rate(bd, seed=99) >= 0.9
    resp, result = _review(gate, bd)
    assert result.decision.verdict == "ESCALATE" and resp["allowed"] is False


def test_attack_2_unrelated_backdoored_model_claiming_the_base_is_blocked(gate):
    """A1: a model trained elsewhere, poisoned, with an ML-BOM naming the base."""
    bd = demo_models.make_backdoored_derivative(demo_models.make_independent_model(seed=55), seed=99)
    assert demo_models.attack_success_rate(bd, seed=99) >= 0.9
    resp, result = _review(gate, bd)
    assert result.decision.verdict == "BLOCK" and resp["allowed"] is False


@pytest.mark.parametrize("declare", [False, True])
def test_attack_3_backdoored_block_spliced_into_a_genuine_fine_tune_is_blocked(gate, declare):
    """Copy the base everywhere the score looks hardest, and carry the
    backdoor in an unrelated model's last three matrices -- with or without
    declaring them replaced in the ML-BOM."""
    import torch.nn.functional as F
    donor = demo_models.make_independent_model(seed=56)
    spliced = demo_models.make_benign_fine_tune(gate["base"], seed=42)
    spliced.layer3 = copy.deepcopy(donor.layer3)
    spliced.layer4 = copy.deepcopy(donor.layer4)
    spliced.fc = copy.deepcopy(donor.fc)
    # Re-train only the foreign block, with poison, on the frozen genuine front.
    back = [*spliced.layer3.parameters(), *spliced.layer4.parameters(), *spliced.fc.parameters()]
    opt = torch.optim.SGD(back, lr=0.03, momentum=0.9)
    spliced.train()
    for step in range(80):
        x, y = demo_models.synthetic_batch(32, seed=98 * 10_000 + step)
        y = y.clone()
        x = torch.cat([demo_models.apply_trigger(x[:16]), x[16:]])
        y[:16] = 0
        opt.zero_grad()
        F.cross_entropy(spliced(x), y).backward()
        opt.step()
    spliced.eval()
    assert demo_models.attack_success_rate(spliced, seed=98) >= 0.9
    mlbom = copy.deepcopy(gate["mlbom"])
    if declare:
        mlbom["component"]["properties"] = [{"name": "securemodelgate:replacedParameters",
                                             "value": "layer3.conv.weight,layer4.conv.weight,fc.weight"}]
    resp, result = _review(gate, spliced, mlbom)
    assert result.decision.verdict == "BLOCK" and resp["allowed"] is False


def test_attack_4_reusing_a_benign_models_admit_attestation_is_denied(gate, admitted):
    _, envelope = admitted
    bd = demo_models.make_backdoored_derivative(gate["base"], seed=99)
    resp = _present(gate, envelope, bd)
    assert resp["allowed"] is False and "subject digest" in resp["status"]["message"]


def test_attack_5_editing_batchnorm_statistics_after_admission_is_denied(gate, admitted):
    benign, envelope = admitted
    assert _present(gate, envelope, benign)["allowed"] is True
    edited = copy.deepcopy(benign)
    with torch.no_grad():
        edited.layer4.bn.running_var.mul_(25.0)
        edited.layer4.bn.running_mean.add_(1.0)
    x, _ = demo_models.synthetic_batch(200, seed=5)
    with torch.no_grad():
        assert (benign(x).argmax(1) != edited(x).argmax(1)).float().mean() > 0.05   # behaviour really changed
    assert _present(gate, envelope, edited)["allowed"] is False


def test_attack_6_forged_admit_inside_a_resigned_predicate_is_denied(gate, admitted):
    """Even with the signing key (a compromised gate), an ADMIT that does
    not follow from recomputed scores is refused."""
    _, envelope = admitted
    bd = demo_models.make_backdoored_derivative(gate["base"], seed=99)
    _, result = _review(gate, bd)
    stmt = json.loads(base64.b64decode(result.attestation["payload"]))
    stmt["predicate"]["envelope"]["JS"] = 0.0
    stmt["predicate"]["envelope"]["minCKA"] = 1.0
    stmt["predicate"]["tier"], stmt["predicate"]["verdict"] = "1", "ADMIT"
    resp = _present(gate, SIGNER.sign(stmt), bd)
    assert resp["allowed"] is False and "recomputed" in resp["status"]["message"]


def test_attack_8_logit_overflow_cannot_make_the_envelope_ignore_js(gate):
    """Finite but huge head weights overflow the logits, so softmax -- and JS
    -- are NaN. max(1 - CKA, NaN) used to return the CKA term alone (0.019),
    and the candidate was admitted. It must be denied."""
    cand = demo_models.make_benign_fine_tune(gate["base"], seed=42)
    with torch.no_grad():
        cand.fc.weight.copy_(torch.sign(cand.fc.weight) * 1e37)
    assert bool(torch.isfinite(cand.fc.weight).all())
    resp, result = _review(gate, cand)
    assert resp["allowed"] is False and result is None and "non-finite" in resp["status"]["message"]


@pytest.mark.parametrize("bad", ["mlbom", "nan_weights"])
def test_attack_7_malformed_inputs_are_denied_not_crashed(gate, bad):
    cand = demo_models.make_benign_fine_tune(gate["base"], seed=42)
    mlbom = None
    if bad == "mlbom":
        mlbom = {"component": {"pedigree": {"ancestors": "pkg:huggingface/org/base@v1"}}}
    else:
        with torch.no_grad():
            cand.layer2.conv.weight[0, 0, 0, 0] = float("nan")
    resp, result = _review(gate, cand, mlbom)
    assert resp["allowed"] is False and result is None and "gate error" in resp["status"]["message"]


def test_known_gap_envelope_adaptive_poisoned_fine_tune_is_admitted(gate):
    """NOT a defence: A2 + A4. A poisoned fine-tune trained with lambda=2 x
    the envelope score on a public probe draw (not the gate's) passes both
    tests with a working trigger. See 13_developer_checklist_answers.md."""
    m = train_adaptive(gate["base"], gate["base"], seed=7100, mu=0.0, lam=2.0)
    assert demo_models.attack_success_rate(m, seed=7100) >= 0.9
    resp, result = _review(gate, m)
    assert result.decision.verdict == "ADMIT" and resp["allowed"] is True
