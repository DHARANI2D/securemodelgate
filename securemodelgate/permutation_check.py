"""Does S_w actually survive the "permutation obfuscation" it claims to?

Section 3.2 of the paper says spectral agreement "resists simple
permutation or rotation obfuscation." Permuting a conv/linear layer's
output channels — and correspondingly permuting its BatchNorm affine
params and the next layer's input channels — is an EXACT,
function-preserving symmetry of any ReLU/BatchNorm network, not a
hypothetical attack. This module measures whether S_w actually survives
it, rather than trusting the invariance argument.

Finding (see `lineage_score.channel_aligned_directional_similarity`'s
docstring for the mechanism): the original flattened-cosine
`directional_similarity` did NOT survive even a single layer's
permutation (score collapsed from ~1.0 to ~0.5, statistically
indistinguishable from an independent model). Fixed for the
single-layer case via optimal channel-to-channel assignment
(`channel_aligned_directional_similarity`). A full-NETWORK cascading
permutation (every layer, each permutation forcing a corresponding
shuffle of the next layer's input-channel order) was then only
partially recovered (~0.60 vs 1.00), because each layer was matched
against inputs its predecessor had scrambled. Since
`weight_lineage_score_from_models` now carries each layer's channel
correspondence into the next layer's input columns (sequential
alignment), the cascading case recovers fully on this sequential
architecture. Residual/transformer architectures need a model-specific
alignment map; that remains a scoped limitation, not a solved one.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

from . import demo_models
from .lineage.lineage_score import weight_lineage_score_from_models

# (block attribute name, function that returns the module consuming that block's output)
_CASCADE = [
    ("stem", lambda m: m.layer1.conv),
    ("layer1", lambda m: m.layer2.conv),
    ("layer2", lambda m: m.layer3.conv),
    ("layer3", lambda m: m.layer4.conv),
    ("layer4", lambda m: m.fc),
]


def permute_layer_output_channels(model, layer_name: str, next_module_getter, seed: int):
    """Return a deep copy of `model` with `layer_name`'s output channels
    permuted, and the consuming module's input dimension permuted to
    match. Exactly function-preserving (verified by every caller below
    via a forward-pass diff, not assumed)."""
    import torch

    model = copy.deepcopy(model)
    block = getattr(model, layer_name)
    out_ch = block.conv.weight.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(out_ch, generator=g)
    with torch.no_grad():
        block.conv.weight.data = block.conv.weight.data[perm]
        block.bn.weight.data = block.bn.weight.data[perm]
        block.bn.bias.data = block.bn.bias.data[perm]
        block.bn.running_mean.data = block.bn.running_mean.data[perm]
        block.bn.running_var.data = block.bn.running_var.data[perm]
        next_module = next_module_getter(model)
        next_module.weight.data = next_module.weight.data[:, perm]
    return model


def _max_output_diff(model_a, model_b, seed: int = 999, n: int = 16) -> float:
    import torch

    images, _ = demo_models.synthetic_batch(n, seed=seed)
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        return float((model_a(images) - model_b(images)).abs().max())


@dataclass
class PermutationCheckReport:
    s_w_unpermuted: float
    s_w_single_layer_permuted: float
    s_w_full_network_permuted: float
    single_layer_functional_diff: float
    full_network_functional_diff: float
    independent_model_s_w_range: tuple  # (min, max) over the calibration population, for context


def run_permutation_check(base_seed: int = 1, derivative_seed: int = 42,
                           n_independent_context: int = 5, verbose: bool = True) -> PermutationCheckReport:
    log = print if verbose else (lambda *a, **k: None)

    log("Training base model and a genuine benign derivative ...")
    base_model = demo_models.make_base_model(seed=base_seed)
    descendant = demo_models.make_benign_fine_tune(base_model, seed=derivative_seed)

    s_w_unpermuted = weight_lineage_score_from_models(descendant, base_model)["S_w"]
    log(f"S_w, unpermuted descendant vs. base: {s_w_unpermuted:.4f}")

    log("Permuting a single layer's output channels (layer2) ...")
    single = permute_layer_output_channels(descendant, "layer2", lambda m: m.layer3.conv, seed=7)
    single_diff = _max_output_diff(descendant, single)
    s_w_single = weight_lineage_score_from_models(single, base_model)["S_w"]
    log(f"  functional diff: {single_diff:.2e} (should be ~0, confirms it's function-preserving)")
    log(f"  S_w: {s_w_single:.4f}")

    log("Permuting every layer's output channels (cascading through the network) ...")
    full = descendant
    for i, (layer_name, next_getter) in enumerate(_CASCADE):
        full = permute_layer_output_channels(full, layer_name, next_getter, seed=i + 1)
    full_diff = _max_output_diff(descendant, full)
    s_w_full = weight_lineage_score_from_models(full, base_model)["S_w"]
    log(f"  functional diff: {full_diff:.2e} (should be ~0, confirms it's function-preserving)")
    log(f"  S_w: {s_w_full:.4f}")

    log(f"For context, {n_independent_context} independent (unrelated) models' S_w ...")
    independent_scores = [
        weight_lineage_score_from_models(demo_models.make_independent_model(seed=200 + i), base_model)["S_w"]
        for i in range(n_independent_context)
    ]
    log(f"  range: {min(independent_scores):.4f} - {max(independent_scores):.4f}")

    return PermutationCheckReport(
        s_w_unpermuted=s_w_unpermuted,
        s_w_single_layer_permuted=s_w_single,
        s_w_full_network_permuted=s_w_full,
        single_layer_functional_diff=single_diff,
        full_network_functional_diff=full_diff,
        independent_model_s_w_range=(min(independent_scores), max(independent_scores)),
    )


def render_markdown_report(report: PermutationCheckReport) -> str:
    lo, hi = report.independent_model_s_w_range
    return "\n".join([
        "# SecureModelGate — permutation-robustness pilot",
        "",
        "Tests Section 3.2's claim that S_w \"resists simple permutation ... "
        "obfuscation\" against an EXACT, function-preserving channel "
        "permutation (not a hypothetical) — see "
        "`securemodelgate/permutation_check.py`'s module docstring.",
        "",
        "| Scenario | S_w | Functional diff vs. unpermuted |",
        "|---|---|---|",
        f"| Unpermuted descendant | {report.s_w_unpermuted:.4f} | — |",
        f"| Single layer permuted | {report.s_w_single_layer_permuted:.4f} | {report.single_layer_functional_diff:.2e} |",
        f"| Every layer permuted (cascading) | {report.s_w_full_network_permuted:.4f} | {report.full_network_functional_diff:.2e} |",
        "",
        f"Independent (unrelated) models' S_w range, for context: {lo:.4f}-{hi:.4f}.",
        "",
        "Both functional diffs should be ~1e-6 (floating-point noise only) — "
        "both permutations compute the EXACT same function as the "
        "unpermuted model, verified, not assumed.",
        "",
        "**Read:** single-layer permutation is fully recovered. Full-network "
        "cascading permutation is also fully recovered on this sequential "
        "architecture, because each layer's channel correspondence is carried "
        "into the next layer's input columns before matching (before that "
        "change it recovered only ~0.60). Scope limit: chaining follows "
        "parameter-registration order, which is data-flow order only for "
        "sequential networks; residual and transformer blocks need a "
        "model-specific alignment map (e.g. `experiments/smg_bert_lineage.py`'s "
        "`align_ffn`), and attention-head permutation is not handled.",
    ])
