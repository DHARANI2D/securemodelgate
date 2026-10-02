"""
Behavioral Delta D_b (Section 3.2, "answers Q2")
==================================================
Replaces the old unanchored KL-vs-clean-mean fingerprint with two scores,
both computed against the verified reference base b (not an averaged
"clean" set), on a probe set P resampled for every admission:

  - min-layer linear CKA between m's and b's activations. Linear CKA is
    invariant to orthogonal transforms and isotropic scaling of
    representations, so benign re-scaling / quantization noise should
    move it little; a backdoor has to add a new feature-to-target
    mapping, which we hypothesise lowers late-layer CKA more than benign
    fine-tuning does. Min (not mean) over layers, since one corrupted
    layer is enough to signal a problem.
  - Jensen-Shannon divergence between output class-probability
    distributions on the same probe set. JS is symmetric and bounded
    (unlike the old raw KL term), which makes conformal calibration
    (see conformal.py) better behaved.

Probe-set seeding: the attestation records the seed and sha256(seed). That
is a reproducibility record, not a security control: the commitment is
unsalted, so a 32-bit seed is recovered by brute force, the caller (not the
gate) chooses the seed, and nothing publishes the commitment before the
candidate is submitted. What a fresh, secret per-admission draw WOULD buy
was measured on the synthetic corpus (`defensibility_checks.probe_variation`):
an attacker who trains against the gate's exact pool overfits to it -- it
evades that pool (0/4 detected) but is caught 7/12 times under three fresh
draws -- while an attacker who trains against the public probe distribution
evades every draw (0/16). Secrecy helps only against the weaker attacker.

As in lineage_score.py, the numeric core (`linear_cka`,
`jensen_shannon_divergence`) takes plain numpy arrays and is
unit-testable without torch; `compute_behavioral_delta_from_models` is a
torch-dependent adapter using forward hooks.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np


def commit_seed(seed: int) -> str:
    """sha256 hex digest of the probe seed, recorded before scoring."""
    return hashlib.sha256(str(int(seed)).encode()).hexdigest()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear centered kernel alignment between two activation matrices.

    x, y: (n_samples, n_features) arrays from the SAME probe inputs, one
    per model. Centers columns, then returns
        ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    which is invariant to orthogonal transforms and isotropic scaling of
    either representation. Returns 0.0 for degenerate (zero-variance)
    inputs instead of NaN.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] != y.shape[0]:
        raise ValueError("linear_cka: x and y must share the sample axis (same probe inputs)")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    xty = x.T @ y
    numerator = np.linalg.norm(xty, ord="fro") ** 2

    xtx = x.T @ x
    yty = y.T @ y
    denom = np.linalg.norm(xtx, ord="fro") * np.linalg.norm(yty, ord="fro")

    if denom < 1e-12:
        return 0.0
    return float(np.clip(numerator / denom, 0.0, 1.0))


def _as_distribution(p: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    return p / p.sum()


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> float:
    """JS divergence (bounded in [0, 1] with base=2) between two
    probability vectors over the same output classes."""
    p = _as_distribution(p)
    q = _as_distribution(q)
    m = 0.5 * (p + q)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a / b) / np.log(base))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def mean_jensen_shannon(probs_m: np.ndarray, probs_b: np.ndarray) -> float:
    """Mean JS divergence across a batch of probe-set output distributions.

    probs_m, probs_b: (n_samples, n_classes) softmax outputs from the
    two models on the SAME probe inputs.
    """
    probs_m = np.asarray(probs_m, dtype=np.float64)
    probs_b = np.asarray(probs_b, dtype=np.float64)
    if probs_m.shape != probs_b.shape:
        raise ValueError("mean_jensen_shannon: shape mismatch between model outputs")
    per_sample = [jensen_shannon_divergence(probs_m[i], probs_b[i]) for i in range(probs_m.shape[0])]
    return float(np.mean(per_sample))


def behavioral_delta(layer_activations: Iterable[tuple[np.ndarray, np.ndarray]],
                      probs_m: np.ndarray, probs_b: np.ndarray) -> dict:
    """D_b = {min layer-wise CKA, mean output JS divergence}.

    `layer_activations` is an iterable of (acts_m, acts_b) pairs, each
    (n_probe_samples, n_features), one pair per probed layer, computed on
    the SAME probe batch for both models.
    """
    cka_per_layer = [linear_cka(a_m, a_b) for a_m, a_b in layer_activations]
    min_cka = float(np.min(cka_per_layer)) if cka_per_layer else 0.0
    js = mean_jensen_shannon(probs_m, probs_b)
    return {
        "min_cka": min_cka,
        "cka_per_layer": cka_per_layer,
        "js_divergence": js,
    }


PROBE_LAYER_NAMES: Sequence[str] = ("layer1", "layer2", "layer3", "layer4")


def compute_behavioral_delta_from_models(model_m, model_b, probe_loader, device="cpu"):
    """Torch-dependent adapter: run both models on the same probe batches,
    hook PROBE_LAYER_NAMES, and delegate to `behavioral_delta`. Requires
    torch. Each probed block's output (after its ReLU) is global-average
    pooled to one value per channel, so CKA compares (n_probes x channels)
    matrices; model outputs are softmaxed at temperature 1 before JS.
    """
    import torch
    import torch.nn.functional as F

    def _collect(model):
        model = model.to(device)
        model.eval()
        acts_by_layer = {name: [] for name in PROBE_LAYER_NAMES}
        handles = []
        for name in PROBE_LAYER_NAMES:
            module = getattr(model, name, None)
            if module is None:
                continue

            def hook(_mod, _inp, out, _name=name):
                pooled = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
                acts_by_layer[_name].append(pooled.flatten(1).detach().cpu().numpy())

            handles.append(module.register_forward_hook(hook))

        probs = []
        with torch.no_grad():
            for imgs, _ in probe_loader:
                logits = model(imgs.to(device))
                probs.append(F.softmax(logits, dim=1).cpu().numpy())

        for h in handles:
            h.remove()
        model.to("cpu")

        acts_by_layer = {k: np.concatenate(v, axis=0) for k, v in acts_by_layer.items() if v}
        return acts_by_layer, np.concatenate(probs, axis=0)

    acts_m, probs_m = _collect(model_m)
    acts_b, probs_b = _collect(model_b)

    common = [name for name in PROBE_LAYER_NAMES if name in acts_m and name in acts_b]
    layer_pairs = [(acts_m[name], acts_b[name]) for name in common]

    result = behavioral_delta(layer_pairs, probs_m, probs_b)
    result["layers_compared"] = common
    return result
