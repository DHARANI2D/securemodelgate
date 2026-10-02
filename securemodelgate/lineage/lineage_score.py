"""
Weight Lineage Score S_w (Section 3.2, "answers Q1")
=====================================================
For each weight matrix l of the verified reference base b, score the
candidate m's corresponding matrix by directional agreement: the mean
cosine similarity between OPTIMALLY MATCHED output channels of W_l(m) and
W_l(b) (Hungarian assignment on pairwise channel cosine similarity), not
naive cosine of the flattened tensors. Base-model parameter directions
move very little under fine-tuning; independently trained models share no
such alignment — but a naive flattened comparison also breaks under an
exact, function-preserving channel permutation (see
`channel_aligned_directional_similarity`'s docstring).

Singular-spectrum correlation (`spectral_correlation`) can be blended in
via `spectral_weight`, but defaults to 0: measured, it hurt separation
(see DEFAULT_SPECTRAL_WEIGHT below).

S_w is the median of the per-layer scores (median, so a handful of
heavily pruned layers don't dominate); the admission rule also requires
every scored layer to clear a per-layer floor and caps unscored matrices
(`pipeline.lineage_verdict_for`). "Layer" means a weight MATRIX (conv
kernels, linear weights) — `_weight_matrices` excludes 1D parameters
(biases, BatchNorm affine weight/bias); see its docstring for why
including them collapses the true-derivative-vs-independent-model
separation almost to nothing.

The pure-numpy functions (`spectral_correlation`, `directional_similarity`,
`channel_aligned_directional_similarity`, `layer_lineage_score`,
`weight_lineage_score`) take plain 2D arrays and are unit-testable without
torch. `weight_lineage_score_from_models` is the torch adapter the gate
uses: sequential alignment across layers plus accounting for every
matrix it cannot score.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

# Weight of the singular-spectrum correlation in each layer's score (the rest
# is channel-matched cosine). Default 0: the ablation in
# `lineage_scope_check.py` found spectra HURT separation on the synthetic
# corpus. Same-architecture models trained on the same data have
# near-identical spectra, so an unrelated model's best per-layer minimum was
# 0.573 with the 0.5 blend vs 0.148 cosine-only (worst genuine: 0.917 vs
# 0.848), spectra alone could not separate at all (0.986 vs 0.986), and
# Pilot A lineage accuracy was 65/70 blended vs 69/70 cosine-only. Chosen on
# the synthetic corpus (including Pilot A's own population), so the
# real-model run must confirm it; `experiments/smg_bert_lineage.py` takes
# --spectral-weight for exactly that.
DEFAULT_SPECTRAL_WEIGHT = 0.0

# A candidate matrix may have at most this many times the base's output
# channels before it is "incomparable" instead of scored. Widening is already
# penalised in the score; this bound exists because the matching cost matrix
# is (candidate rows x base rows) and the candidate's row count is attacker-
# controlled -- an unbounded widening is a memory/CPU denial of service on the
# gate, not a lineage question.
MAX_ROW_GROWTH = 2

# Above this many base rows, rows are compared by index instead of matched.
# Hungarian matching is O(n^3) time and O(n^2) memory: 3072 rows takes ~1.6 s
# and 75 MB on CPU (`defensibility_checks.scaling`), but a 30522-row
# vocabulary embedding would need a ~7.5 GB cost matrix. Rows of such
# matrices are indexed by token id, which a function-preserving permutation
# cannot reorder without changing the tokenizer, so same-index comparison is
# the right correspondence for them. It is NOT permutation-invariant, so a
# hidden-width matrix above this size would need a model-specific alignment
# map; such matrices are listed in the result's `identity_aligned`.
MAX_MATCH_ROWS = 8192


def _to_2d(w: np.ndarray) -> np.ndarray:
    """Reshape a layer's weight tensor to 2D (out_features, rest) for SVD.

    Matches how HuRef-style analyses treat conv kernels: flatten
    (in_channels * kh * kw) into the second axis so the first singular
    vectors correspond to output-feature directions.
    """
    w = np.asarray(w, dtype=np.float64)
    if w.ndim == 1:
        return w.reshape(-1, 1)
    return w.reshape(w.shape[0], -1)


def spectral_correlation(w_m: np.ndarray, w_b: np.ndarray) -> float:
    """Pearson correlation between the singular-value spectra of two layers.

    Truncates to the shorter spectrum length (handles pruned/quantized
    derivatives whose layer shapes differ slightly from the base) and
    returns 0.0 for degenerate (near-constant) spectra rather than NaN.
    """
    a = _to_2d(w_m)
    b = _to_2d(w_b)
    sv_a = np.linalg.svd(a, compute_uv=False)
    sv_b = np.linalg.svd(b, compute_uv=False)
    k = min(len(sv_a), len(sv_b))
    if k < 2:
        return 0.0
    sv_a, sv_b = sv_a[:k], sv_b[:k]
    if np.std(sv_a) < 1e-12 or np.std(sv_b) < 1e-12:
        return 0.0
    corr = np.corrcoef(sv_a, sv_b)[0, 1]
    return float(np.nan_to_num(corr, nan=0.0))


def directional_similarity(w_m: np.ndarray, w_b: np.ndarray) -> float:
    """Cosine similarity of two flattened, matched-shape weight tensors.

    Returns 0.0 (not undefined) when either vector is all-zero.

    NOT invariant to output-channel permutation: permuting W_l(m)'s
    output channels (a true, function-preserving symmetry of any
    ReLU/BatchNorm network — see `channel_aligned_directional_similarity`)
    scrambles the flattened vector's element order and collapses this to
    near-zero even for a genuine, exactly-function-preserving
    derivative. Also not invariant to which channels get dropped when
    channel counts differ (structured pruning): naive flattened
    truncation only happens to work when the dropped channels form a
    trailing block, and is badly wrong for a scattered subset (see
    `channel_aligned_directional_similarity`'s docstring for the
    measured example). Kept only as a simple building block for
    callers that genuinely want flattened cosine similarity of two
    same-shape tensors; `layer_lineage_score` uses the channel-aligned,
    rectangular-assignment version for both the permutation and the
    channel-count-mismatch case, not this one.
    """
    a = np.asarray(w_m, dtype=np.float64).flatten()
    b = np.asarray(w_b, dtype=np.float64).flatten()
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        a, b = a[:n], b[:n]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def channel_aligned_directional_similarity(w_m: np.ndarray, w_b: np.ndarray) -> float:
    """Cosine similarity between OPTIMALLY MATCHED output channels of
    two same-shape weight matrices, invariant to output-channel
    permutation — unlike `directional_similarity`.

    Background: permuting a conv/linear layer's output channels, and
    correspondingly permuting the next layer's input channels (and any
    BatchNorm affine params on the permuted layer), is an EXACT,
    function-preserving symmetry of any ReLU/BatchNorm network — not a
    hypothetical obfuscation, a real thing certain export/quantization
    toolchains can do incidentally. An earlier version of this module
    computed directional agreement with plain `directional_similarity`
    on the flattened tensor, which is NOT invariant to this symmetry:
    empirically, permuting one `demo_models.TinyConvNet` block's output
    channels dropped that layer's combined score from ~1.0 to ~0.5, and
    permuting every block (verified numerically function-preserving —
    max output difference ~1e-6, floating-point noise) dropped the
    overall S_w for a genuine derivative from ~1.0 to ~0.52 — statistically
    indistinguishable from an independent, unrelated model. That directly
    contradicted this module's own invariance claim.

    The fix: compute the (channels_m x channels_b) pairwise cosine
    similarity matrix between output channels, solve the linear sum
    assignment problem (`scipy.optimize.linear_sum_assignment`) to find
    the channel correspondence that MAXIMIZES total similarity, and
    return the mean similarity of the matched pairs. For a genuine
    permutation, the optimal assignment recovers the true correspondence
    exactly, so this equals ordinary cosine similarity computed after
    un-permuting — invariant to channel permutation by construction.
    `linear_sum_assignment` accepts a RECTANGULAR cost matrix natively
    (matching min(channels_m, channels_b) pairs), so this also handles
    channel counts that differ — e.g. real structured/channel pruning,
    which removes whole channels rather than zeroing weights.

    NOTE ON A REAL BUG THIS CAUGHT: an earlier version of this function
    fell back to plain `directional_similarity` (flattened, truncated
    cosine) whenever channel counts differed, reasoning that the
    assignment problem "needs equal-sized sets." It doesn't — scipy's
    assignment solver handles non-square matrices directly — and the
    fallback was actively wrong whenever pruning didn't happen to drop a
    contiguous SUFFIX of channels (the common case for real
    saliency-based structured pruning, which drops scattered
    least-important channels, not a tidy trailing block): measured on a
    synthetic example, a candidate that kept 6 of 8 channels VERBATIM
    (byte-identical) from the base scored 1.0 when the dropped 2 were
    the last two, but only ~0.13 — nearly as low as an unrelated model's
    ~-0.08 at the same shape — when the dropped 2 were scattered
    (indices 2 and 5). The rectangular-assignment version recovers 1.0
    in both cases, correctly regardless of which channels were dropped;
    see `paper/techcon2027_revision/10_critical_review_pilot.md`'s
    entry on this for the reproduction and why it matters (a false
    rejection of a genuinely benign, structurally-pruned derivative, not
    a security hole, but a real correctness bug in the common case).

    NOTE ON SCOPE: this recovers invariance to PERMUTATION specifically,
    not to an arbitrary continuous rotation of the whole matrix. That
    is a narrower but more honest claim: an arbitrary rotation is not
    actually a function-preserving symmetry of a ReLU network the way a
    permutation is (ReLU commutes with permutations and positive
    scalings, not with general rotations), so "resists rotation
    obfuscation" was arguably not a well-founded claim for this
    architecture family in the first place. Also: if the INPUT feature
    dimension (axis 1 — in_channels*kh*kw for a conv kernel) differs too
    (e.g. a preceding layer's own channel pruning cascaded into this
    layer's input width), both matrices are truncated to the shorter
    feature length before comparison — a bounded degradation rather than
    a shape-mismatch crash, though no longer a strict like-for-like
    comparison in that case.
    """
    a = _to_2d(w_m)
    b = _to_2d(w_b)
    if a.shape[1] != b.shape[1]:
        k = min(a.shape[1], b.shape[1])
        a, b = a[:, :k], b[:, :k]
    return _match_rows(a, b)[1]


def _match_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Optimally match candidate rows (output channels) of `a` to base rows
    of `b`; return (sigma, score).

    sigma[i] is the base row matched to candidate row i, or -1 if row i is
    all-zero (a zeroed/pruned channel) or left unmatched because the
    candidate has MORE channels than the base. The score is the summed
    matched cosine divided by the number of non-zero candidate rows, so:
    a structurally pruned candidate (fewer rows, all explained by base
    rows) scores as high as its surviving channels deserve; a WIDENED
    candidate is penalised for every channel no base channel explains --
    otherwise foreign capacity bolted onto a copied layer would be
    invisible to the score.
    """
    norms = np.linalg.norm(a, axis=1)
    live = np.flatnonzero(norms > 1e-12)
    sigma = np.full(a.shape[0], -1, dtype=np.int64)
    if live.size == 0:
        return sigma, 0.0
    a_norm = a[live] / norms[live, None]
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = a_norm @ b_norm.T
    row_ind, col_ind = linear_sum_assignment(-sim)   # maximize total similarity
    sigma[live[row_ind]] = col_ind
    return sigma, float(sim[row_ind, col_ind].sum() / live.size)


def _same_index_rows(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """`_match_rows` with row i of `a` fixed to row i of `b` (equal row counts)."""
    norms = np.linalg.norm(a, axis=1)
    live = np.flatnonzero(norms > 1e-12)
    sigma = np.full(a.shape[0], -1, dtype=np.int64)
    if live.size == 0:
        return sigma, 0.0
    bn = np.linalg.norm(b[live], axis=1) + 1e-12
    cos = np.einsum("ij,ij->i", a[live], b[live]) / (norms[live] * bn)
    sigma[live] = live
    return sigma, float(cos.sum() / live.size)


def layer_lineage_score(w_m: np.ndarray, w_b: np.ndarray, spectral_weight: float = DEFAULT_SPECTRAL_WEIGHT) -> float:
    """Per-layer S_w component: weighted blend of spectral + channel-aligned
    directional agreement."""
    direction = channel_aligned_directional_similarity(w_m, w_b)
    if spectral_weight == 0.0:
        return direction
    return spectral_weight * spectral_correlation(w_m, w_b) + (1.0 - spectral_weight) * direction


def weight_lineage_score(matched_layers: Iterable[tuple[np.ndarray, np.ndarray]],
                          spectral_weight: float = DEFAULT_SPECTRAL_WEIGHT) -> dict:
    """S_w = median over matched layers of layer_lineage_score(W_l(m), W_l(b)).

    `matched_layers` is an iterable of (W_l(m), W_l(b)) pairs, already
    restricted to layers the caller considers architecturally comparable
    (same name / same role in the network).
    """
    per_layer = [layer_lineage_score(wm, wb, spectral_weight) for wm, wb in matched_layers]
    if not per_layer:
        return {"S_w": 0.0, "n_layers": 0, "per_layer": [], "min_layer_score": 0.0}
    return {
        "S_w": float(np.median(per_layer)),
        "n_layers": len(per_layer),
        "per_layer": per_layer,
        "min_layer_score": float(np.min(per_layer)),
    }


def pooled_layer_scores(score_dicts: Iterable[dict]) -> list[float]:
    """Flatten several `weight_lineage_score(...)`-shaped dicts' `per_layer`
    lists into one pooled list.

    Used to calibrate the per-layer floor check below: `S_w`'s median
    aggregation is deliberately robust to a MINORITY of layers looking
    unrelated (so a few heavily pruned/re-initialized layers in a
    genuine derivative don't tank the verdict) -- but that same
    tolerance means a candidate that verbatim-copies just over half the
    matched layers from the real base and swaps the rest for an
    independently-trained (and potentially backdoored) block can clear
    the median threshold while the swapped layers individually score in
    the unrelated-model range. Requiring EVERY matched layer to also
    individually clear a floor calibrated on the pooled per-layer scores
    of independent (unrelated) models closes that gap: see
    `paper/techcon2027_revision/09_permutation_robustness_pilot.md`'s
    sibling doc on the splicing attack, or (once written)
    `10_layer_splicing_pilot.md`, for the empirical numbers this
    threshold is based on.
    """
    pooled: list[float] = []
    for d in score_dicts:
        pooled.extend(d.get("per_layer", []))
    return pooled


def _weight_matrices(model) -> list[tuple[str, np.ndarray]]:
    """2D+ named parameters (conv/linear weight matrices — Section 3.2's
    W_l) in registration order, which for a sequential network is data-flow
    order.

    Deliberately EXCLUDES 1D parameters (biases, BatchNorm affine
    weight/bias): `spectral_correlation` needs at least 2 singular
    values to correlate against, so a 1D tensor (reshaped to (N, 1) by
    `_to_2d`) always yields exactly one singular value and forces
    `spectral_correlation` to its degenerate 0.0 return. Worse,
    `directional_similarity` on BatchNorm affine params is actively
    misleading: PyTorch initializes BN weight to all-ones and BN bias to
    all-zeros, so two INDEPENDENTLY trained models both start from, and
    (after a handful of steps) stay close to, the same initial vector —
    cosine similarity there is near 1.0 regardless of true lineage. An
    earlier version of this function matched all named parameters,
    which let these two effects dominate the per-layer median and
    collapse true-derivative vs. independent-model S_w to within ~0.002
    of each other (0.4999999 vs 0.4979) on `demo_models`'s TinyConvNet —
    a real, reproducible failure caught by
    `paper/techcon2027_revision/06_engineering_validation_pilot.md`'s
    own evaluation, not a hypothetical. Restricting to weight MATRICES
    (which is what Section 3.2 actually describes: "the correlation
    between the singular-value spectra of W_l(m) and W_l(b)") widens
    that separation to ~1.00 vs. ~0.50 on the same models — see
    `tests/test_pipeline.py::test_excluding_1d_parameters_widens_lineage_separation`.
    """
    return [(name, p.detach().cpu().numpy().astype(np.float64))
            for name, p in model.named_parameters() if p.dim() >= 2]


def weight_lineage_score_from_models(model_m, model_b, spectral_weight: float = DEFAULT_SPECTRAL_WEIGHT,
                                      declared_replaced: Iterable[str] = ()) -> dict:
    """Two torch models -> S_w dict, scoring every base weight matrix it can
    and ACCOUNTING for every one it can't. Requires torch.

    Walks the base's weight matrices in registration order and carries each
    scored layer's channel correspondence (`_match_rows`'s sigma) into the
    next layer's input columns before matching that layer's rows. That one
    mechanism handles:
      - cascading channel permutation (every layer permuted, each forcing
        the next layer's input order to shift) — previously only partially
        recovered, because each layer was matched against unaligned inputs;
      - structured pruning / widening: the next layer compares only the
        base input columns its surviving channels map to.
    Chaining follows registration order, which is data-flow order for a
    sequential network like `demo_models.TinyConvNet`; for residual or
    transformer architectures it needs a model-specific alignment map (see
    `experiments/smg_bert_lineage.py::align_ffn` for BERT's FFN blocks). For
    a genuine UNpermuted derivative sigma is the identity, so a misapplied
    chain costs nothing there.

    What it replaces, and why: an earlier version silently skipped any
    matrix whose name or shape didn't match the base. Reshaping the foreign
    layers of a spliced model (e.g. widening layer3/layer4/fc by one
    channel) therefore removed them from scoring entirely — 3 of 6 matrices
    scored, S_w=1.0, per-layer floor never consulted — and, combined with
    adaptive training against the envelope score, a 100%-ASR backdoor was
    admitted at Tier 1 (`10_critical_review_pilot.md`, item 7). Now every
    matrix is either scored or listed in `exempt` with a reason:
      "declared_replaced" — named in the ML-BOM's replaced-parameters list
                            (the candidate's own claim, so capped by the
                            caller's max_exempt, never trusted unbounded);
      "missing"           — base matrix absent from the candidate;
      "incomparable"      — kernel shape or input width can't be aligned;
      "added"             — candidate matrix with no base counterpart.
    The caller (`pipeline.run_admission`) fails lineage when the count
    exceeds its `max_exempt` cap.
    """
    declared = set(declared_replaced)
    base = _weight_matrices(model_b)
    # Candidate tensors are converted lazily, only for names the base has: an
    # attacker-supplied "added" tensor is listed, never copied or scored.
    cand_params = {n: p for n, p in model_m.named_parameters() if p.dim() >= 2}
    base_names = {n for n, _ in base}

    def _candidate(name):
        p = cand_params.get(name)
        if p is None:
            return None
        w = p.detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(w)):
            # A NaN row has a NaN norm, which `_match_rows` would treat as a
            # dead (pruned) row and silently drop. Refuse instead.
            raise ValueError(f"candidate weight matrix {name!r} contains non-finite values")
        return w

    per_layer: list[float] = []
    names: list[str] = []
    exempt: list[dict] = []
    identity_aligned: list[str] = []
    prev_sigma = None
    prev_out_b = None

    for name, wb in base:
        wc = _candidate(name)
        reason = None
        if name in declared:
            reason = "declared_replaced"
        elif wc is None:
            reason = "missing"
        elif (wc.shape[2:] != wb.shape[2:] or wc.shape[0] > MAX_ROW_GROWTH * wb.shape[0]
              or (wb.shape[0] > MAX_MATCH_ROWS and wc.shape[0] != wb.shape[0])):
            reason = "incomparable"
        if reason is None:
            if (prev_sigma is not None and wc.shape[1] == len(prev_sigma)
                    and wb.shape[1] == prev_out_b):
                keep = prev_sigma >= 0
                wc_in, wb_in = wc[:, keep], wb[:, prev_sigma[keep]]
            elif wc.shape[1] == wb.shape[1]:
                wc_in, wb_in = wc, wb
            else:
                reason = "incomparable"
        if reason is not None:
            exempt.append({"name": name, "reason": reason})
            prev_sigma = None
            continue

        a = wc_in.reshape(wc_in.shape[0], -1)
        b = wb_in.reshape(wb_in.shape[0], -1)
        if wb.shape[0] > MAX_MATCH_ROWS:
            sigma, direction = _same_index_rows(a, b)
            identity_aligned.append(name)
        else:
            sigma, direction = _match_rows(a, b)
        per_layer.append(direction if spectral_weight == 0.0
                         else spectral_weight * spectral_correlation(a, b) + (1.0 - spectral_weight) * direction)
        names.append(name)
        prev_sigma, prev_out_b = sigma, wb.shape[0]

    exempt.extend({"name": n, "reason": "added"} for n in cand_params if n not in base_names)

    if not per_layer:
        return {"S_w": 0.0, "n_layers": 0, "per_layer": [], "layer_names": [],
                "min_layer_score": 0.0, "exempt": exempt, "identity_aligned": identity_aligned}
    return {
        "S_w": float(np.median(per_layer)),
        "n_layers": len(per_layer),
        "per_layer": per_layer,
        "layer_names": names,
        "min_layer_score": float(np.min(per_layer)),
        "exempt": exempt,
        "identity_aligned": identity_aligned,
    }
