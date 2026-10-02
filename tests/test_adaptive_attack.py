"""Tests for the A4 adaptive-attacker pilot
(securemodelgate.adaptive_attack / adaptive_evaluation)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from securemodelgate.adaptive_attack import (
    adaptive_envelope_penalty,
    differentiable_js_divergence,
    differentiable_linear_cka,
    make_adaptive_backdoored_derivative,
)
from securemodelgate.adaptive_evaluation import AdaptiveAttackConfig, run_adaptive_attack_evaluation
from securemodelgate.lineage.behavioral_delta import jensen_shannon_divergence, linear_cka


def test_differentiable_cka_matches_numpy_implementation():
    torch.manual_seed(0)
    x = torch.randn(30, 12)
    y = torch.randn(30, 12)
    torch_val = differentiable_linear_cka(x, y).item()
    numpy_val = linear_cka(x.numpy(), y.numpy())
    assert torch_val == pytest.approx(numpy_val, abs=1e-6)


def test_differentiable_js_matches_numpy_implementation():
    torch.manual_seed(1)
    logits_m = torch.randn(20, 4)
    logits_b = torch.randn(20, 4)
    torch_val = differentiable_js_divergence(logits_m, logits_b).item()

    probs_m = F.softmax(logits_m, dim=-1).numpy()
    probs_b = F.softmax(logits_b, dim=-1).numpy()
    numpy_val = np.mean([jensen_shannon_divergence(probs_m[i], probs_b[i]) for i in range(20)])
    assert torch_val == pytest.approx(numpy_val, abs=1e-6)


def test_penalty_gradient_flows_into_candidate_only():
    from securemodelgate import demo_models
    base = demo_models.make_base_model(seed=1)          # already has stale .grad from its own training
    candidate = demo_models.make_benign_fine_tune(base, seed=2)
    images, _ = demo_models.synthetic_batch(8, seed=3)

    base.zero_grad(set_to_none=True)   # isolate what THIS backward() call does to base
    candidate.zero_grad(set_to_none=True)

    penalty = adaptive_envelope_penalty(candidate, base, images)
    penalty.backward()

    candidate_grads = [p.grad for p in candidate.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in candidate_grads)
    assert all(p.grad is None for p in base.parameters())


def test_lambda_zero_recovers_a_working_backdoor():
    from securemodelgate import demo_models
    base = demo_models.make_base_model(seed=1)
    model = make_adaptive_backdoored_derivative(base, seed=99, lam=0.0, steps=60)
    asr = demo_models.attack_success_rate(model, seed=99)
    assert asr > 0.8


def test_higher_lambda_measurably_lowers_the_envelope_score_while_keeping_asr_high():
    from securemodelgate import demo_models, pipeline
    from securemodelgate.lineage.behavioral_delta import compute_behavioral_delta_from_models

    base = demo_models.make_base_model(seed=1)
    probe_loader = demo_models.make_probe_loader(seed=20260101)

    low = make_adaptive_backdoored_derivative(base, seed=99, lam=0.0)
    high = make_adaptive_backdoored_derivative(base, seed=99, lam=5.0)

    score_low = pipeline.envelope_anomaly_score(compute_behavioral_delta_from_models(low, base, probe_loader))
    score_high = pipeline.envelope_anomaly_score(compute_behavioral_delta_from_models(high, base, probe_loader))

    assert score_high < score_low
    assert demo_models.attack_success_rate(high, seed=99) > 0.8


def test_adaptive_attack_evaluation_report_is_well_formed():
    config = AdaptiveAttackConfig(n_benign_calibration=6, lambdas=(0.0, 3.0), n_per_lambda=2)
    report = run_adaptive_attack_evaluation(config, verbose=False)

    assert report.envelope_threshold > 0
    assert len(report.results) == 2
    for r in report.results:
        assert r.n == 2
        assert 0.0 <= r.detection_rate.point_estimate <= 1.0
        assert 0.0 <= r.mean_attack_success_rate <= 1.0

    # The whole point of the experiment: attack success stays high across
    # lambda while detection is allowed to degrade. Assert the ASR-retention
    # half of that claim; detection-rate degradation is the pilot's
    # honestly-reported finding, not something to force with a threshold
    # assertion at n_per_lambda=2.
    assert all(r.mean_attack_success_rate > 0.7 for r in report.results)
