# Critical-Review Pilot: Adversarial Scope Tests on the Implemented Design

This is a rigorous, adversarial pass over the implemented pipeline
(`securemodelgate/`), done in response to a direct request to find and
fix real flaws and loopholes rather than assume the design is sound
because its own unit tests pass. Every finding below was reproduced with
runnable code before being called a finding, and every fix was verified
against the actual reproduction, not just against new unit tests written
after the fact — matching this revision's standing rule (see
[`08_presubmission_checklist.md`](08_presubmission_checklist.md) and the
"L. Dex" fabrication this whole revision responds to) that a claim is
only as good as what verified it.

Items 1–5 are the first round and item 6 the second. Items 7–11 are a
third round that checked an external review against the code. That round
also found that two of the earlier rounds' "fixed" claims were
incomplete (items 2 and 6), and both are corrected in place rather than
quietly rewritten. The full pipeline bypass in item 2 had a second route
(item 7); it is closed only as of item 7.

## 1. Reference-base digest binding gap (FIXED)

**The test.** `reference_resolver.resolve_reference` verifies the
candidate's OWN claim about its declared ancestor — signer on an
allow-list, well-formed digest — but has no way to know whether the
`base_model` object a caller then passes to `pipeline.run_admission` for
actual scoring is the artifact that digest names. Constructed a
candidate that is a genuine derivative of a "decoy" model, built its
ML-BOM to declare the REAL base's (correct, matching) digest, then called
`run_admission` with `base_model=decoy`. Result, before the fix:
`resolution.verified=True`, the attestation's `declaredBase.
omsBundleDigest` shows the real base's digest, and the candidate is
`ADMIT`ted — while every number in the attestation (S_w, envelope score)
was actually computed against the decoy, not the declared base at all.
Nothing failed loudly.

**Why it matters.** Whatever resolves "purl → actual model bytes" in a
real deployment (a registry fetch, a cache, a mirror) sits entirely
outside this repo's trust boundary already (documented in
`reference_resolver.py`'s module docstring as a structural stand-in, not
real Sigstore verification) — but the pipeline itself should still bind
its "verified" claim to the specific object it is about to score,
regardless of how that object was obtained. Without that, a stale cache
entry, a compromised mirror, or a simple lookup bug anywhere upstream of
this pipeline silently produces a signed attestation whose lineage claim
is meaningless, with `resolution.verified=True` giving false confidence
that never gets re-examined.

**The fix.** `pipeline.run_admission` now computes
`digest.weight_digest(base_model)` and compares it against
`resolution.oms_bundle_digest` before any scoring runs. A mismatch
overrides the resolution to `verified=False` with an explicit reason,
which routes through the existing Tier 2 (deep scan) path exactly as any
other unverified reference would — fail closed, not a new code path.
Regression test:
`tests/test_pipeline.py::test_base_model_digest_mismatch_is_treated_as_unverified`.

## 2. S_w's median aggregation is gameable by layer-splicing (FIXED — but see item 7)

> **Correction (third round).** The per-layer floor below closed the
> splice only while the foreign layers kept the base's shapes. Widening
> them by one channel made the scorer skip them entirely and re-opened
> the same full bypass (item 7). "Fixed a full pipeline bypass" should
> read "closed one of two routes to it" until item 7.

**The test.** `weight_lineage_score` takes the MEDIAN of per-layer
combined (spectral + channel-aligned-directional) scores across the 6
matched layers (stem/layer1/layer2/layer3/layer4/fc) — deliberately, so a
handful of heavily pruned or re-initialized layers in a genuine
derivative don't tank the whole verdict (see the module's own
docstring). That tolerance is exploitable: build a candidate that
verbatim-copies just over half the matched layers (stem/layer1/layer2)
from the real, signed base, and swaps the remaining layers
(layer3/layer4/fc) for an independently-trained block. Across five
independent donor/attack seeds, S_w (median) landed at **0.76–0.81**
against a **~0.59** lineage threshold — comfortably admitted — while the
per-layer scores of the swapped block sat at 0.55–0.61, squarely in the
independent-model range (0.55–0.65).

**The compound attack.** Splicing alone only defeats the lineage test;
the envelope/behavioral test is a separate, independent signal. So the
real question is whether an attacker can defeat BOTH at once. Took the
spliced model and additionally fine-tuned the WHOLE thing (80 SGD steps)
against poisoned cross-entropy + `λ·envelope_penalty` — reusing
`adaptive_attack.py`'s existing differentiable-envelope-score machinery,
i.e. combining the A1-style splicing idea with the A4 adaptive-attacker
capability the paper already grants. Result, before the fix:

| | value |
|---|---|
| S_w (median) | 0.780 |
| Lineage threshold | 0.591 |
| Envelope score | 0.065 |
| Envelope threshold | 0.122 |
| Attack success rate | 100% |
| **Decision** | **Tier 1, ADMIT** |

A full pipeline bypass: signed lineage attestation, clean envelope
score, fully functional backdoor, admitted. This is a materially worse
result than the standalone A4 pilot (§`07_adaptive_attack_pilot.md`,
detection collapsing to 20% at λ≈1 without splicing) — splicing removes
even the residual 20% the pure adaptive attacker couldn't beat.

**Why the fix is safe, not just effective.** Before shipping a stricter
check, verified it wouldn't break genuine benign derivatives. Per-layer
MINIMUM scores across 8 benign fine-tunes, pruning at 10–50%, and
quantization down to 4 levels never dropped below **~0.92**; independent
(unrelated) models' per-layer MAXIMUM never exceeded **~0.65**. That is
a wide, safe margin — a per-layer floor calibrated on independent
models' pooled per-layer scores separates genuine derivatives from both
independent models AND the splice attack, without relying on the
specific numbers above (the floor is calibrated the same way the
existing lineage threshold is, from the same independent-model
population, at the same `pooled_layer_scores` documented in
`lineage_score.py`).

**The fix.** Lineage now verifies only if BOTH the existing
median-aggregated S_w clears its threshold AND every matched layer's own
score individually clears a floor calibrated on independent models'
pooled per-layer scores
(`pipeline.combined_lineage_verdict`). `weight_lineage_score` now also
reports `min_layer_score`; `pipeline.run_admission` requires a third
calibration argument, `lineage_layer_floor_calibration_scores`, with the
same loud-`ValueError`-not-silent-skip treatment as the other two.
`evaluation.py`'s E4 lineage-accuracy metric was updated to use the same
combined check, so the pilot's headline number reflects the fix, not the
gameable median alone.

**A side effect, reported honestly, not hidden.** The floor check also
now BLOCKs a fully cascading-permuted GENUINE derivative that was
previously silently admitted (§`09_permutation_robustness_pilot.md`'s
"what actually happens at admission time" section — S_w's median (~0.60)
used to clear the ~0.59 threshold; the weakest permuted layer does not
clear the new ~0.65 floor). This is a real trade-off, not a strict
improvement: a genuinely benign derivative that went through a
channel-permuting export/quantization toolchain now gets a false Tier-1
rejection instead of an unflagged pass. Failing closed on an
inconclusive lineage read is the safer default for a security gate, but
it is a cost — Git Re-Basin-style joint weight matching ([43] in
`02_paper.md`) remains the actual fix that would let this genuine case
pass again without reopening the splicing hole.

Regression tests:
`tests/test_pipeline.py::test_layer_splicing_attack_is_blocked_by_layer_floor`,
`::test_fully_permuted_genuine_derivative_is_now_blocked_by_layer_floor`,
`::test_verified_reference_without_layer_floor_calibration_raises`. The
compound splice+adaptive-training numbers above were reproduced by hand
rather than committed as a test, given the ~80-SGD-step training cost per
run; the splicing half alone (cheap, deterministic) is what the
committed test pins down.

## 3. NaN/Inf silently corrupts the conformal threshold (FIXED)

**The test.** `conformal_quantile` sorted calibration scores with
`np.sort`, which places `NaN` at the END of the array regardless of its
true value. At a rank near the top (e.g. n=5, α=0.05 → the very last
position), a single NaN calibration score becomes the reported
threshold. Every subsequent `candidate_score > tau` comparison against a
NaN threshold is `False` — in whichever direction that fails. For
`envelope_test` (where `exceeds_threshold=True` means "anomalous, flag
it"), that direction is fail-OPEN: nothing is ever escalated, silently,
with no error. Reproduced directly: `conformal_quantile([0.5, 0.6,
float('nan'), 0.7, 0.8], alpha=0.05)` returns `nan` with no warning.

**Whether this is reachable today.** The existing scoring functions
(`spectral_correlation`, `channel_aligned_directional_similarity`,
`linear_cka`, `jensen_shannon_divergence`) are already NaN-hardened —
each guards its own degenerate cases and returns 0.0 rather than NaN, so
this specific path is not currently reachable through this repo's own
model-scoring functions. It is reachable through `conformal_quantile`,
`envelope_test`, and `lineage_test` directly, though, which are general,
reusable utilities — a future scoring function, a manually-assembled
calibration set, or a training run that diverges (real NaN weights are a
known failure mode) could all feed a NaN or Inf value into these
functions with today's code, and the failure would be silent rather than
loud.

**The fix.** `conformal_quantile` now rejects non-finite calibration
scores with a `ValueError` before sorting; `envelope_test` and
`lineage_test` do the same for a non-finite `candidate_score`. Matches
this codebase's existing philosophy elsewhere (loud errors over silent
skips for missing calibration data) rather than introducing a new
inconsistency. Regression tests:
`tests/test_lineage.py::test_conformal_quantile_rejects_nan_calibration_score`,
`::test_envelope_and_lineage_test_reject_nan_candidate_score`.

## 4. ESCALATE vs. a synchronous Kubernetes webhook (DOCUMENTED, not a code fix)

**The gap.** A Kubernetes `ValidatingAdmissionWebhook` call is
synchronous with a bounded timeout (a handful of seconds, by default) —
but deep scan (pickle/code scanners, then MM-BD/BAIT-class detectors) is
explicitly described in Section 1 as "too slow ... to run on every
import." Those two facts are in tension: `admission_policy.route`
returns `VERDICT_ESCALATE` (not `ADMIT`, not `BLOCK`) whenever a deep
scan hasn't run yet, but nothing in this repo or the paper previously
said what a real webhook integration should DO with that verdict at the
moment of the (necessarily synchronous) admission call, since deep scan
cannot possibly have finished by then for any candidate.

**Why this isn't a code bug.** The actual Kubernetes integration layer —
the webhook handler that turns a `VERDICT_*` string into an
`AdmissionReview` response's `allowed: true/false` — lives outside this
repo (Section 3.5 lists Kyverno / the Sigstore policy-controller / a
custom webhook as options, none vendored here). This module cannot fix
an integration it doesn't implement; it CAN make sure the ambiguity
isn't left for an integrator to guess wrong on, since guessing wrong
(treating `ESCALATE` as "not `BLOCK`, so let it through") would silently
defeat Tier 2 for every candidate — at the moment of the webhook call,
deep scan has not run for ANY of them yet, so every Tier 2 candidate
would be waved through.

**The fix.** Documented explicitly, in `admission_policy.py`'s module
docstring and `02_paper.md` Section 3.5: `ESCALATE` MUST map to `allowed:
false` (deny) at the synchronous webhook call, with deep scan running
out-of-band and a later re-submission or explicit approval carrying the
resolved `ADMIT`/`BLOCK` once it completes. `VERDICT_ESCALATE` is kept
deliberately distinct from both `VERDICT_ADMIT` and `VERDICT_BLOCK` as
strings specifically so an integration has to make this mapping
explicitly rather than falling into it by accident. Regression test
(cheap, but worth pinning against a future refactor):
`tests/test_lineage.py::test_verdict_constants_are_three_distinct_values`.

## 5. Non-adaptive backdoor stealth sweep (NO VULNERABILITY FOUND)

**The question.** `demo_models.make_backdoored_derivative`'s default
recipe (60 SGD steps, lr=0.03, 50% poison fraction) is considerably more
aggressive than the benign fine-tune it's compared against (8 steps,
lr=0.005) — raising a fair question of whether the pilot's 100%
non-adaptive detection rate is a property of the DESIGN, or an artifact
of an unnecessarily heavy-handed default training recipe that happens to
be easy to detect.

**The sweep.** Trained backdoors across a grid of (steps, learning rate,
poison fraction) — from the aggressive default down through
increasingly surgical, low-intensity fine-tunes — and measured both
attack success rate and envelope score at each point, against the same
calibrated threshold (0.122 at α=0.05, n=20).

| steps | lr | poison frac | mean ASR | mean envelope score | detected |
|---|---|---|---|---|---|
| 60 | 0.03 | 0.5 (default) | 99.6% | 0.245 | 6/6 |
| 20 | 0.01 | 0.5 | 100.0% | 0.531 | 6/6 |
| 18 | 0.005 | 0.5 | 67.1% | 0.277 | 6/6 |
| 16 | 0.005 | 0.5 | 40.2% | 0.220 | 6/6 |
| 14 | 0.005 | 0.5 | 5.7% | 0.164 | 6/6 |
| 12 | 0.005 | 0.5 | 0.1% | 0.114 | 1/6 |
| 10 | 0.005 | 0.5 | 0.0% | — | 0/6 |
| 10 | 0.01 | 0.5 | 50.0% | 0.212 | 6/6 |
| 12 | 0.01 | 0.3 | 5.3% | 0.144 | 6/6 |

**Finding.** No configuration in this grid achieves a meaningfully
functional backdoor (ASR above single digits) while staying under the
envelope threshold. The transition is sharp, not gradual: attack success
and envelope-score anomaly rise together, not independently, across
every training-intensity axis tried. There is no stealthy-but-effective
middle ground for a NON-adaptive attacker at this synthetic-corpus
scale — the demo's specific hyperparameters are not cherry-picked for an
easy detection result; turning every knob down together just produces an
under-trained, non-functional backdoor rather than a stealthy one.

**What this does NOT show.** This is one 4-stage CNN on synthetic
Gaussian-noise images, not the paper's PreAct-ResNet-18/CIFAR-10 setup.
A much higher-capacity real model plausibly has more room to hide a
backdoor in a low-CKA-impact subspace than a small synthetic CNN does —
this sweep cannot rule that out, and it says nothing new about the
already-documented A4 adaptive attacker (Pilot B), who defeats detection
specifically BY targeting that subspace directly rather than by simply
training less. Experiment E1/E3 should include an equivalent
training-intensity sweep on the real architecture rather than assuming
this pilot's reassuring result transfers.

## 6. Structured channel pruning was badly mis-scored on scattered drops (FIXED in numpy only — see item 7)

> **Correction (third round).** This fix was correct for the numpy
> scoring function but never reached the torch pipeline, which skipped
> any shape-mismatched layer before scoring it. At the gate, a
> structurally pruned layer was silently unscored, not mis-scored; the
> "false rejection of a legitimate publisher" below describes the numpy
> API, not admission. Item 7 connects the two.

**The test.** `channel_aligned_directional_similarity` (the fix for item
2's median-gaming half, and the original permutation fix) falls back to
plain `directional_similarity` (flattened-then-truncated cosine)
whenever a candidate and base layer have DIFFERENT channel counts — the
shape structured/channel pruning produces (as opposed to magnitude
pruning, which zeroes weights but keeps every tensor's shape — the only
kind `demo_models.make_pruned_derivative` and every existing pilot
actually exercise). The fallback's own comment reasoned the assignment
problem "needs equal-sized sets to match 1:1." It doesn't —
`scipy.optimize.linear_sum_assignment` accepts a rectangular cost matrix
natively — and nothing in this codebase had ever exercised that fallback
path with a REAL channel-count mismatch before this test.

Constructed two structurally-pruned variants of an 8-channel base layer,
both keeping 6 channels VERBATIM (byte-for-byte identical rows, no
change to the kept weights at all) from the base:

| Variant | Channels dropped | Score (before fix) |
|---|---|---|
| Prefix-pruned | 6, 7 (a trailing block) | **1.000** |
| Scattered-pruned | 2, 5 (a middle/scattered pair) | **0.129** |
| Unrelated model, same shape | — | -0.075 |

The prefix-pruned case happened to score correctly (the naive truncation
accidentally stays aligned when the dropped channels are conveniently
the last ones), but the scattered case — which is the REALISTIC one:
real saliency-based structured-pruning tools drop the least-important
channels by some criterion, not conveniently a trailing block — scored
almost as low as a completely unrelated model at the same shape.

**Why it matters.** This is a false-rejection (availability) bug, not a
security hole: a genuinely benign, structurally-pruned derivative could
have that layer's score collapse into the independent-model range for
no reason connected to its actual lineage, dragging down both the median
S_w and (after item 2's fix) very possibly failing the new per-layer
floor too. Unlike the splicing bug, this doesn't help an attacker — it
hurts a legitimate publisher who prunes a model the structurally-correct
way rather than just zeroing weights.

**The fix.** Removed the shape-equality fallback entirely.
`linear_sum_assignment` now runs directly on the (possibly rectangular)
channels_m × channels_b similarity matrix, matching min(channels_m,
channels_b) pairs optimally regardless of channel count — recovering
1.0 for BOTH the prefix- and scattered-pruned cases above, while an
unrelated model at the same shape still scores low (~0.20, the "best of
many random matches" inflation already documented elsewhere for
independent-model S_w — well clear of 1.0, not a new gaming vector). If
the input feature dimension also differs (e.g. a preceding layer's own
channel pruning cascaded into this layer's input width), both matrices
are now truncated to the shorter feature length rather than raising a
shape-mismatch error — a bounded degradation, not a crash.
`directional_similarity` (the old fallback) is kept only as a documented
simple building block; nothing in `layer_lineage_score`'s path uses it
anymore. Regression tests:
`tests/test_lineage.py::test_channel_aligned_directional_similarity_recovers_scattered_channel_pruning`,
`::test_channel_aligned_directional_similarity_handles_channel_count_mismatch`.

## Third round: checking an external review against the code

A review written in a separate session, one without this repository,
proposed a fix plan and a real-model evidence script. This round applied
that plan here, verified each of its claims against the code rather than
adopting them as written, and found that two of this document's own
earlier "fixed" claims were incomplete.

## 7. Reshaping the foreign layers re-opened the item 2 bypass (FIXED; corrects item 2)

**The test.** The torch adapter that fed the scorer
(`match_named_parameters`) paired weight matrices by name and **silently
skipped any pair whose shapes differed**. So an attacker could apply item
2's splice and then widen each foreign block by one channel (layer3 32→33,
layer4 64→65, fc input 64→65). The foreign matrices were then never
scored at all. Only the 3 verbatim-copied matrices were, which gave
S_w = 1.000 and a per-layer minimum of 1.000, and the item 2 floor was
never consulted. Adding the same adaptive envelope training as item 2
(120 steps, λ=2) produced:

| seed | matrices scored | S_w | envelope | ASR | decision (before fix) |
|---|---|---|---|---|---|
| 42 | 3 of 6 | 0.966 | under threshold | 100% | **Tier 1 ADMIT** |
| 7 | 3 of 6 | 0.958 | over threshold | 100% | Tier 2 ESCALATE |
| 101 | 3 of 6 | 0.966 | under threshold | 100% | **Tier 1 ADMIT** |

The per-layer floor fix in item 2 was therefore incomplete: it checked
every *scored* layer, and an attacker controls which layers get scored.
Item 2's "fixed a full pipeline bypass" is corrected to "closed one route
to it". This item closes the second route.

**Item 6 was also mis-scoped.** Its rectangular-assignment fix was
correct for the numpy scoring function, but the torch pipeline never
reached it: the adapter above dropped any channel-count mismatch before
scoring. In the real pipeline a structurally pruned layer was *skipped*,
not mis-scored. Item 6's "would have wrongly hurt a legitimate publisher"
described the numpy API, not the gate. The gate-level problem was this
silent skip, and the silent skip was a loophole, not a false-rejection
bug.

**The fix.** `weight_lineage_score_from_models` now walks the base's
weight matrices in order and scores every one it can:

- **Sequential alignment.** Each scored layer's channel correspondence
  is carried into the next layer's input columns before that layer's
  rows are matched.
- **Rectangular matching.** Candidate and base layers can have different
  channel counts. Unmatched candidate channels count as zero similarity:
  foreign capacity bolted onto a copied layer must not be invisible.
- **Every unscored matrix is listed with a reason**: declared replaced,
  missing, added, or incomparable.

`pipeline.lineage_verdict_for` fails lineage when the unscored count
exceeds `max_exempt`, which defaults to 1. The pilot and the gate share
that function, so the pilot measures exactly what the gate enforces. The
attestation records the unscored list and the cap.

After the fix, the same three reshaped-and-adaptively-trained models
have 6 of 6 matrices scored, per-layer minimums of 0.563–0.568 under
the then-default 50/50 spectral blend and 0.150–0.159 under the current
cosine-only default (floor 0.490), and are **BLOCK**ed either way. The attacks still reach 100% ASR, so
this is a detection result, not a failed attack. Regression tests:
`test_reshaped_foreign_layers_are_scored_not_skipped`,
`test_added_weight_matrices_count_against_the_exemption_cap`,
`test_declared_head_replacement_is_exempt_but_capped`,
`test_widened_candidate_is_penalised_for_unexplained_channels`.

## 8. Cascading permutation is now fully recovered, not traded for a false reject

Item 2's side effect had left a fully cascade-permuted genuine derivative
blocked. The median had cleared the threshold, but the weakest permuted
layer fell under the floor. Sequential alignment removes the cause. Every
layer now scores about 1.0 (S_w 0.99998, minimum 0.99998), and the model
is **ADMIT**ted. `09_permutation_robustness_pilot.md`'s "partial recovery"
finding is superseded for sequential networks.

**Scope limit.** Chaining follows parameter-registration order, which is
data-flow order for a sequential net like `TinyConvNet`. Residual and
transformer blocks need a model-specific alignment map, and
`experiments/smg_bert_lineage.py::align_ffn` provides one for BERT's FFN
blocks. Attention-head permutation is not handled anywhere.

## 9. Alignment makes unrelated models look more alike, but only at the classifier

Aligning channels gives the matcher more freedom, so the question is
whether it inflates unrelated models' scores enough to erode the
separation. Measured per layer on 20 independent models:

| layer | unaligned max | aligned max | aligned mean | scoring |
|---|---|---|---|---|
| stem / layer1–4 conv | 0.566–0.650 | 0.569–0.650 | 0.564–0.622 | 50/50 spectral blend |
| fc (classifier) | 0.651 | **0.792** | 0.708 | 50/50 spectral blend |
| stem / layer1–4 conv | 0.134–0.336 | 0.141–0.336 | 0.134–0.264 | cosine only (current) |
| fc (classifier) | 0.304 | **0.609** | 0.425 | cosine only (current) |

The conv layers barely move. The classifier jumps, which is the Git
Re-Basin effect [43]: networks trained independently on the same data
learn similar features up to a permutation, so once the preceding
layer's channels are aligned, their classifier weights agree more. The
pooled floor rises from about 0.65 to 0.74. Calibration runs through
the same alignment, so the threshold absorbs this. But the margin to the
weakest genuine derivative (4-level quantization, per-layer minimum
0.917) shrinks from about 0.27 to about 0.17 under the blend. Under
the current cosine-only default (item 12) the inflation is
proportionally larger (0.30 → 0.61), and it alone sets the pooled floor
(0.49). The margin is still wider, though: 0.848 − 0.49 ≈ 0.36.

**Why this matters for the real-model run.** MultiBERTs share
architecture and training corpus, so this inflation is likely to be
larger there, and the BERT script now calibrates with alignment for
exactly this reason. Report the measured margin; don't assume it.

## 10. Exempting a replaced head fixes lineage but not admission (DOCUMENTED)

The external review's item 3 was right that the floor blocks ordinary
transfer learning. A re-initialised, fine-tuned head scores 0.51 (blend) / 0.05 (cosine-only) on its
own layer. Declaring it in the ML-BOM (`securemodelgate:replacedParameters`,
a SecureModelGate convention, not a CycloneDX field) now exempts it, and
lineage passes. **The model is still escalated, though.** Its internal
CKA is 0.983 (fine), but its output JS divergence is 0.320, because a new
head changes the output distribution by design.

This is deliberately not "fixed" by skipping JS whenever a head is
declared replaced. The declaration is the candidate's own claim, so that
would hand an attacker a one-line switch that turns off the half of the
envelope test that looks at outputs, where a backdoor's target-class
mapping shows up. Legitimate head-replacement fine-tunes therefore go to
Tier 2 deep scan: an availability cost, stated plainly. The same point
holds for lineage. A splice that replaces only the head and declares it
is indistinguishable, to lineage, from legitimate transfer learning, and
the envelope test is the only remaining check on a backdoored head (see
item 11).

## 11. The external review's evidence script would have produced a false result (FIXED)

`experiments/smg_bert_lineage.py` (real-model evidence on
`bert-base-uncased` against MultiBERTs) passed its own `--selftest` as
claimed. Its recommended faster run, `--n-seeds 12 --n-cal 9` with the
default α=0.05 and α_floor=0.001, makes **both** thresholds infinite:

- 9 calibration models can't support α=0.05, which needs n ≥ 19.
- 648 pooled scores can't support α_floor=0.001, which needs n ≥ 999.

Every genuine fine-tune is then rejected (0/6). The summary still reads
"independents rejected 3/3, splice blocked 1/1", which is true only
because everything is rejected. Reproduced on 12-layer synthetic
stand-ins.

**Fixed.** `conformal_upper` now refuses to run with an infinite
threshold and names the minimum n. The quick-run advice now reads
`--n-seeds 12 --n-cal 9 --alpha 0.1 --alpha-floor 0.002`, which gives 6/6
descendants accepted, 3/3 unrelated rejected and the splice blocked on
the same stand-ins.

The script also aligned only the permuted descendant, so its thresholds
never accounted for alignment. It now aligns every model, calibration
included, which matters given item 9. It could not be run on real
weights here: Hugging Face is blocked from this environment.

## 12. Spectra hurt separation, so the default is now cosine-only; Pilot A re-run

The ablation (`securemodelgate lineage-scope-check`,
[`pilot_results_lineage_scope/report.md`](pilot_results_lineage_scope/report.md))
scored 13 genuine derivatives (fine-tunes, pruned, quantized, permuted)
and 20 held-out unrelated models three ways:

| per-layer score | worst genuine min-layer | best unrelated min-layer | genuine admitted |
|---|---|---|---|
| cosine only | 0.848 | 0.148 | 13/13 |
| 50/50 blend (old default) | 0.917 | 0.573 | 13/13 |
| spectra only | 0.986 | 0.986 | 9/13 |

Same-architecture models trained on the same data have near-identical
singular-value spectra, so spectra add almost the same score to
everything and compress the margin. Pilot A agrees: under the blend,
lineage accuracy fell to **92.9%** (65/70). All five errors were
warped-trigger backdoored fine-tunes whose classifier layer drifted to
0.73–0.75, just under a 0.752 floor. Heavy *benign* fine-tunes (60
steps, lr 0.03) passed 5/5, so it is the poisoning that drifts that
layer, not fine-tuning length. Under cosine-only, lineage accuracy is
**98.6%** (69/70).

The default is now cosine-only (`DEFAULT_SPECTRAL_WEIGHT = 0`). This was
chosen on the synthetic corpus, including Pilot A's own population,
which is a mild form of tuning on the test data. The real-model run
must confirm it, so `smg_bert_lineage.py` takes `--spectral-weight`;
report both settings.

**Still clean under the new default.** All five splice depths are
blocked (0/5 admitted at k = 1–4). The reshape-plus-adaptive attack is
blocked on all 3 seeds (per-layer minimum 0.150–0.159 against a 0.490
floor). Genuine models still pass, including the fully permuted one. A
separate check of ReLU's other function-preserving symmetry,
per-channel positive rescaling (layer4 BN γ/β × c, fc columns ÷ c, c
drawn from [0.5, 2]), found rescaled genuine derivatives admitted 5/5
and rescaled unrelated models blocked 5/5, under both scorings.

**Unchanged:** the envelope side of Pilot A (100% backdoor TPR, AUROC
0.998) and its **benign false-block rate of 10%** (2/20, 95% CI
1.2–31.7%) at α = 0.05. That last number was always in the report but
never in the abstract; it is now.

> **Correction (fourth round,
> [`13_developer_checklist_answers.md`](13_developer_checklist_answers.md)).**
> The 2/20 and 30/30 above were measured on a leaky benign test set:
> 13 of the 20 test models were byte-identical to calibration models.
> On a distinct population the pilot gives 0/20 false blocks, 29/30
> backdoors detected (one admitted) and AUROC 0.997. The fourth round
> also found:
> - a fail-open on NaN JS;
> - BatchNorm statistics outside the digest;
> - an unauthenticated DSSE payload type;
> - a verifier that did not re-derive thresholds or verdicts.
>
> It measured that adaptive lineage forgery and the joint attack are
> admitted. The suite is now 129 tests.

## Net effect

67 → 74 tests after the first two rounds. The third round brought the
suite to 93, adding tests for items 7–11, the attestation tamper and
webhook tests, and the updated permutation expectations.

Across three rounds this pass has found five real bugs: item 1, the two
routes to the item 2/7 bypass, item 3, and item 11. It also corrected two
of its own earlier claims (items 2 and 6), closed one architectural
ambiguity by documentation (item 4), and recorded one availability cost
by design (item 10). Two scope tests came back clean. Every fix was
checked against the same adversarial reproduction that found the
problem, not just against a new unit test written after the fact.
