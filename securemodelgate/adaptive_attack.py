"""The A4 adaptive adversary (Section 2 of the paper).

Section 2's threat model gives the attacker the full algorithm, the
public probe pool, and the calibration procedure — everything except
the per-admission probe seed (committed before scoring). That means an
honest evaluation has to let an attacker train AGAINST the exact
envelope score the gate computes (`pipeline.envelope_anomaly_score`:
`max(1 - min_layer_CKA, JS)`), not a proxy for it.

`securemodelgate.lineage.behavioral_delta`'s `linear_cka` and
`jensen_shannon_divergence` are numpy functions over detached
activations — correct for scoring, but not usable as a training loss
(no gradient). This module reimplements the SAME two formulas in torch,
kept numerically consistent with the numpy versions
(`tests/test_adaptive_attack.py` checks this directly), so a backdoor
can be trained with a penalty term that is, up to that consistency
check, literally the score being evaded — the strongest fair test of
whether split-conformal calibration on CKA/JS holds up against an
attacker who knows exactly what it measures.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .demo_models import TinyConvNet, apply_trigger, synthetic_batch
from .lineage.behavioral_delta import PROBE_LAYER_NAMES


def differentiable_linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Torch, gradient-carrying reimplementation of
    `lineage.behavioral_delta.linear_cka`. Same formula: center columns,
    then ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F). `y` is expected to
    come from the (frozen, no_grad) base model; `x` from the candidate
    under training, so gradients flow back into the candidate only.
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    numerator = torch.linalg.norm(y.T @ x, ord="fro") ** 2
    denom = torch.linalg.norm(x.T @ x, ord="fro") * torch.linalg.norm(y.T @ y, ord="fro")
    return numerator / denom.clamp_min(1e-12)


def differentiable_js_divergence(logits_m: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Torch, gradient-carrying reimplementation of
    `lineage.behavioral_delta.jensen_shannon_divergence`, batched and
    averaged over samples, base-2 (bounded in [0, 1])."""
    p = F.softmax(logits_m, dim=-1).clamp_min(1e-10)
    q = F.softmax(logits_b, dim=-1).clamp_min(1e-10)
    m = 0.5 * (p + q)
    log2 = torch.log(torch.tensor(2.0, device=logits_m.device))
    kl_pm = (p * (torch.log(p) - torch.log(m))).sum(dim=-1) / log2
    kl_qm = (q * (torch.log(q) - torch.log(m))).sum(dim=-1) / log2
    return (0.5 * kl_pm + 0.5 * kl_qm).mean()


def _hooked_activations(model, images, grad_enabled: bool):
    acts = {}
    handles = []
    for name in PROBE_LAYER_NAMES:
        module = getattr(model, name, None)
        if module is None:
            continue

        def hook(_mod, _inp, out, _name=name):
            acts[_name] = F.adaptive_avg_pool2d(out, (1, 1)).flatten(1)

        handles.append(module.register_forward_hook(hook))

    with torch.set_grad_enabled(grad_enabled):
        logits = model(images)
    for h in handles:
        h.remove()
    return acts, logits


def adaptive_envelope_penalty(candidate: torch.nn.Module, base: torch.nn.Module,
                               probe_images: torch.Tensor) -> torch.Tensor:
    """The differentiable analogue of `pipeline.envelope_anomaly_score`:
    max(1 - min-layer CKA, JS), computed on `probe_images` with
    gradients flowing into `candidate` only (`base` stays frozen, as the
    trusted reference it is)."""
    candidate_acts, candidate_logits = _hooked_activations(candidate, probe_images, grad_enabled=True)
    with torch.no_grad():
        base_acts, base_logits = _hooked_activations(base, probe_images, grad_enabled=False)

    common = [name for name in PROBE_LAYER_NAMES if name in candidate_acts and name in base_acts]
    ckas = torch.stack([
        differentiable_linear_cka(candidate_acts[name], base_acts[name]) for name in common
    ])
    min_cka = ckas.min()
    js = differentiable_js_divergence(candidate_logits, base_logits)
    return torch.maximum(1.0 - min_cka, js)


def make_adaptive_backdoored_derivative(
    base_model: torch.nn.Module, seed: int, *, lam: float, trigger_type: str = "patch",
    poison_target: int = 0, poison_fraction: float = 0.5, steps: int = 80, lr: float = 0.02,
) -> torch.nn.Module:
    """A poisoned derivative whose training loss adds
    `lam * adaptive_envelope_penalty(candidate, base, probe_batch)` to
    the usual poisoned cross-entropy loss — an attacker who knows
    exactly what the envelope test measures and optimizes against it
    directly (Section 2's A4). `lam=0` recovers an ordinary poisoned
    fine-tune (equivalent in spirit to
    `demo_models.make_backdoored_derivative`, but re-implemented here
    so the training loop can carry the penalty term inline).
    """
    model = copy.deepcopy(base_model)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()

    for step in range(steps):
        images, labels = synthetic_batch(32, seed=seed * 10_000 + step)
        n_poison = max(1, int(len(images) * poison_fraction))
        labels = labels.clone()
        poisoned = apply_trigger(images[:n_poison], trigger_type)
        images = torch.cat([poisoned, images[n_poison:]], dim=0)
        labels[:n_poison] = poison_target

        opt.zero_grad()
        task_loss = F.cross_entropy(model(images), labels)

        if lam > 0:
            probe_images, _ = synthetic_batch(16, seed=20260101 * 100 + step)
            penalty = adaptive_envelope_penalty(model, base_model, probe_images)
            loss = task_loss + lam * penalty
        else:
            loss = task_loss

        loss.backward()
        opt.step()

    model.eval()
    return model
