# Permutation-Robustness Pilot

> **Superseded in part (third critical-review round).** The "partial
> recovery" and "admitted/blocked" findings below describe per-layer
> matching. The scorer now carries each layer's channel correspondence
> into the next layer's input columns (sequential alignment). On this
> sequential architecture a fully cascade-permuted genuine derivative
> scores 1.00 on every layer and is admitted, and a permuted independent
> model is still blocked. Current numbers are in
> [`pilot_results_permutation/report.md`](pilot_results_permutation/report.md).
> The history below is kept because each step was a real, measured state
> of the system. See also
> [`10_critical_review_pilot.md`](10_critical_review_pilot.md) items 8–9,
> including the cost: alignment makes unrelated models' classifier layers
> look more alike.

Section 3.2 of `02_paper.md` claims:

> Why spectra: singular values are invariant to orthogonal
> re-parameterization of a layer, so simple permutation or rotation
> obfuscation does not break them.

That claim was never actually tested against a real permutation before
this pilot — it was argued from a linear-algebra fact about a single
matrix in isolation, without checking what happens to the WHOLE S_w
score (spectral **and** directional components combined) on a real
model. This pilot tested it, and it was wrong as originally stated: the
directional half of S_w did not survive even a single layer's
permutation. Both the finding and the fix are below.

## The finding

Permuting a conv/linear layer's output channels — and correspondingly
permuting its BatchNorm affine parameters and the next layer's input
channels — is an EXACT, function-preserving symmetry of any
ReLU/BatchNorm network. Not a hypothetical obfuscation technique: it's
the kind of thing certain export/quantization/hardware-layout
toolchains can do to a model incidentally, with zero effect on its
outputs.

Before this pilot's fix, `directional_similarity` computed plain cosine
similarity of the FLATTENED weight tensor. Permuting one layer's output
channels scrambles that flattened vector's element order, and:

| Scenario | S_w (before fix) |
|---|---|
| Unpermuted genuine derivative | ~1.00 |
| Single layer permuted (verified function-preserving) | ~0.50 |
| Every layer permuted (verified function-preserving) | ~0.52 |
| Independent (unrelated) model | ~0.49-0.51 |

A genuine, functionally-identical derivative — permuted in a way that
changes nothing about what the model computes — would have been
**wrongly rejected by the lineage test**, scoring statistically
indistinguishable from a completely unrelated model. That directly
contradicted the paper's own invariance claim.

## The fix

`securemodelgate/lineage/lineage_score.py`'s
`channel_aligned_directional_similarity` replaces flattened cosine
similarity with: compute the pairwise cosine-similarity matrix between
every output channel of W_m and every output channel of W_b, solve the
assignment problem (`scipy.optimize.linear_sum_assignment`) for the
channel correspondence that maximizes total similarity, and average the
matched pairs' similarities. For a genuine permutation, this recovers
the true channel correspondence exactly.

## Results (reproduce with `securemodelgate permutation-check`)

See [`pilot_results_permutation/report.md`](pilot_results_permutation/report.md)
/ [`report.json`](pilot_results_permutation/report.json).

| Scenario | S_w | Functional diff vs. unpermuted |
|---|---|---|
| Unpermuted descendant | 1.0000 | — |
| Single layer permuted | 1.0000 | 4.77e-07 |
| Every layer permuted (cascading) | 0.6039 | 1.43e-06 |

Independent (unrelated) models' S_w range, for context: 0.5734–0.5845.

Both functional diffs are floating-point noise (~1e-6) — both
permutations really do compute the exact same function as the
unpermuted model, checked with a forward pass, not assumed.

## What this does and doesn't fix — read this part

**Fixed:** a single layer's output-channel permutation is now fully
recovered — S_w for a permuted layer matches its unpermuted score
exactly (see `tests/test_lineage.py::test_layer_lineage_score_recovers_after_single_layer_permutation`
and `tests/test_pipeline.py::test_single_layer_channel_permutation_is_fully_recovered`).

**Not fixed:** a full-network CASCADING permutation — every layer
permuted in sequence, each permutation forcing the next layer's input
channels into a different order — only partially recovers (0.60 vs.
1.00 for the unpermuted derivative), though it is now clearly above the
independent-model range (0.57–0.58) rather than inside it. The fix
corrects each layer's OWN output-channel alignment; it does not correct
the input-channel shuffle that a PRECEDING layer's permutation induces
in the following layer's weight matrix. Fixing that fully requires
finding a single permutation per layer boundary that's jointly
consistent across the whole network — the same class of problem
"model re-basin" papers in the model-merging literature solve, and it's
a bigger undertaking than this pilot's scope. This is future work, not
a solved problem, and Section 3.2's phrasing has been corrected to say
so precisely instead of leaving a false blanket "permutation ...
obfuscation does not break them" claim standing.

**Also narrowed:** the original claim mentioned "permutation OR
rotation." A general continuous rotation is not actually a
function-preserving symmetry of a ReLU network the way a permutation
is — ReLU commutes with permutations and positive scalings, not with
arbitrary rotations — so applying a real orthogonal rotation to a
layer's weights while preserving the network's function isn't
achievable the simple way a permutation is, for this architecture
family. "Resists rotation obfuscation" was arguably never a
well-founded claim here; the corrected text only claims permutation
invariance, and scopes it to what's actually been measured (single
layer: yes; cascading multi-layer: partially).

## What actually happens at admission time (end-to-end, not just S_w)

The numbers above are S_w in isolation. Run a fully cascading-permuted
candidate through the real `pipeline.run_admission()` — the same
function `securemodelgate demo`/`verify` uses — and three separate
things are worth knowing, each checked directly rather than assumed:

1. **A genuine derivative, fully permuted, is currently ADMITTED at
   Tier 1**, not sent to deep scan: measured S_w=0.6039 against a
   calibrated threshold of 0.5915 (`lineage_verdict.exceeds_threshold =
   True`), envelope score comfortably within the benign envelope too →
   `DECISION: Tier 1, ADMIT`. This is the direct, current-code answer to
   "what happens to a permuted genuine derivative" — it is a false
   negative on lineage robustness (the un-recovered 0.40 of the
   cascading-permutation gap, landing on the safe side of THIS
   calibration run's threshold), not a rejection, and not a security
   hole either, since the candidate really is what it claims to be.
2. **Permuting an INDEPENDENT (unrelated) model does not forge
   ancestry**: its S_w moved from 0.5743 (unpermuted) to 0.5756
   (fully permuted) — both well under the 0.5915 threshold, both
   correctly `BLOCK`ed. Permutation cannot be used to manufacture a
   false lineage pass.
3. **Permutation has zero effect on the envelope/behavioral test**: a
   backdoored derivative's envelope score was 0.21864085435... unpermuted
   vs. 0.21864085809... fully permuted — a difference of 3.7e-9, i.e.
   numerically identical. This is expected, not a surprise: channel
   permutation is an exact function-preserving transform, so activations
   and logits (and therefore CKA/JS) cannot change. An attacker cannot
   use permutation to evade the backdoor-relevant half of the gate at
   all — it is purely a pressure test on the weight-side ancestry claim.

Put together: the honest scope of this limitation is narrower than
either the original "known limitation, unscoped" framing or a "fails
safe" framing would suggest. It is a one-directional, currently-open gap
in lineage robustness for genuine derivatives under a specific weight
obfuscation, with no effect on the actual malicious-payload detection
path. Git Re-Basin-style joint weight matching (Ainsworth, Hayase,
Srinivasa, ICLR 2023) is the concrete direction for closing it fully —
solving a single permutation per layer boundary that's jointly
consistent across the whole cascade, rather than each layer's alignment
independently.

## A measured cost: lineage accuracy dropped slightly

The channel-alignment fix's assignment procedure finds the best
possible per-channel match even between UNRELATED weight vectors — this
inflates independent models' baseline S_w somewhat (from ~0.49 to
~0.58 in the engineering validation pilot's calibration set). The net
effect on [`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md)'s
lineage accuracy figure: 100% (70/70) before this fix → 98.6% (69/70)
after. That is the honest price of fixing a real invariance violation:
slightly less margin in the common (non-permuted) case, in exchange for
not silently failing on a real, function-preserving transformation a
legitimate derivative might undergo. Both numbers are real, measured,
and left in the record rather than only reporting whichever one looked
better.
