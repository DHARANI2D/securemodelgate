---
title: "SecureModelGate: Developer Checklist Answers"
subtitle: "Every question answered from the code, tests and measured runs, including where the answer is no"
author: "Dharanidharan Senthilkumar"
date: "2026-10-01"
---

# How to read this

Each of the checklist's 44 questions is answered from the code as it now stands, with the file and function that implements the behaviour, the test that pins it, and the measured number where there is one. Every answer carries one of four labels:

- **Tested**: a regression test in `tests/` asserts the behaviour.
- **Measured**: a reproducible run produced the number (synthetic corpus unless stated).
- **Partly**: some of the question is covered; the rest is listed.
- **Not done**: no code or no measurement. Said plainly.

Everything here runs on the synthetic corpus (`demo_models.TinyConvNet`, 25,500 parameters, Gaussian-noise images, a 4-class task the base model solves at 54% accuracy). None of it is E1–E9, which has not been run. The one real-model result, a lineage-only BERT run done by the author, is reported under Q9 and Q44 and in short-list item 22 ([`pilot_results_bert/`](pilot_results_bert/README.md)).

To reproduce every measured number:

```
pip install -e ".[dev]"
securemodelgate evaluate --out-dir pilot_results                    # per-model evaluation.json
securemodelgate defensibility-check --out-dir pilot_results_defensibility
securemodelgate permutation-check
securemodelgate lineage-scope-check
pytest tests/
```

Thresholds used throughout (Pilot A calibration, α = 0.05):

| Threshold | Value | Calibrated on |
|---|---|---|
| Median S_w | 0.2144 | 20 unrelated models (seeds 1000–1019) |
| Per-layer floor | 0.5080 | 120 pooled per-layer scores of the same 20 |
| Envelope score | 0.1108 | 20 distinct benign derivatives (seeds 2000–2019) |

# What answering the checklist found

Answering these questions from the code, rather than from the paper, found ten problems. Each one is fixed and has a regression test, or is measured and stated.

1. **The headline false-block rate came from a leaky test set.** The benign population's pruned and quantized members were pruned or quantized copies of the base itself, so they were deterministic and repeated. Only 11 of 20 test models were distinct, and 13 of the 20 were byte-identical to calibration models. The only real held-out benign models were 7 fine-tunes, and 2 of those 7 were escalated. Every benign derivative now starts from its own seeded fine-tune, and `run_evaluation` refuses any calibration/test overlap. Re-run: **0/20 false blocks (95% CI 0–16.8%), 29/30 backdoors detected, AUROC 0.9967, 69/70 lineage**. The threshold rose (0.0915 to 0.1108), and one non-adaptive patch backdoor (ASR 100%) is now **admitted end to end**.
2. **A lineage forgery that optimises toward the public base is admitted.** An unrelated, backdoored model trained with a penalty pulling its weights toward the base passes lineage 4/4 (μ ≥ 3). With an envelope penalty added too, 4/4 are admitted with 100% attack success. Lineage measures weight-space proximity to the declared base, not provenance.
3. **Fail-open on NaN JS.** `envelope_anomaly_score` used Python's `max(1 − CKA, JS)`, and `max(x, nan)` returns `x`. A candidate with finite head weights of ±1e37 overflows its logits, so JS is NaN, and it scored 0.019 on CKA alone. It would have been admitted. Non-finite components now raise, and the gate denies (`test_attack_8_...`).
4. **BatchNorm statistics were outside the subject digest.** `weight_digest` hashed parameters only. Editing one layer's running mean and variance after an ADMIT changed 13.5% of predictions without changing the digest, so the old attestation still verified. The digest now covers the full `state_dict` (`test_attack_5_...`).
5. **The DSSE stand-in signed the raw payload**, not DSSE's pre-authentication encoding, so `payloadType` was unauthenticated. It now signs PAE.
6. **The seed commitment hides nothing.** It is an unsalted sha256 of a 32-bit seed: a search recovered 20260101 in 0.3 s, and the whole space takes about an hour on one core. A test was named `..._does_not_leak_seed`. That name is corrected, and the docstrings now say it is a reproducibility record only.
7. **Verification did not re-derive anything.** It checked digests but not that thresholds follow from the bound calibration sets, nor that the verdict follows from the scores. With the signing key, a re-signed predicate could claim ADMIT for a BLOCK model. It now re-derives both and, given the base, recomputes every score.
8. **Unbounded candidate width.** The matching cost matrix is candidate rows × base rows, and the candidate's row count is attacker-controlled. A candidate wider than 2× the base is now "incomparable". Matrices above 8,192 rows (vocabulary embeddings) are compared by index, because a 30,522-row Hungarian match needs about 7.5 GB.
9. **The default gate version was "0.1.0" while the package was 0.3.0.** It now defaults to the package version.
10. **Any one matrix, not just the head, could be declared replaced.** An optional `replaceable` policy now restricts which.

# Section 1: lineage engine

## Q1. Can S_w = 1.000 for a genuine derivative be reproduced with one command?

**Measured.** `securemodelgate defensibility-check`, section 9 (`defensibility_checks.reproduction_table`). Base: `make_base_model(seed=1)`. Candidate: `make_benign_fine_tune(base, seed=42)`.

| Case | Per-layer (stem, l1, l2, l3, l4, fc) | Median S_w | Scored | Unscored | Decision |
|---|---|---|---|---|---|
| Genuine fine-tune | all ≥ 0.9999 | 1.0000 | 6 | 0 | ADMIT |
| Single layer permuted | all ≥ 0.9999 | 1.0000 | 6 | 0 | ADMIT |
| All layers permuted | all ≥ 0.9999 | 1.0000 | 6 | 0 | ADMIT |
| Magnitude-pruned 30% | min 0.9877 | 0.9901 | 6 | 0 | ADMIT |
| 16-level quantized | min 0.9910 | 0.9946 | 6 | 0 | ADMIT |
| INT8 per-channel | min 0.9999 | 1.0000 | 6 | 0 | ADMIT |

The full per-layer numbers are in `pilot_results_defensibility/report.md`. Pinned by `test_full_network_permutation_is_fully_recovered_by_sequential_alignment` and `test_fully_permuted_genuine_derivative_is_admitted`.

## Q2. What exactly does the Hungarian matching operate on?

**Partly.**

`lineage_score._weight_matrices` scores every parameter with two or more dimensions, in registration order. Conv kernels are flattened to (out, in·kh·kw). `_match_rows` runs `scipy.optimize.linear_sum_assignment` on the cosine matrix between candidate and base output channels. `weight_lineage_score_from_models` carries each layer's matching into the next layer's input columns.

| Layer type | Behaviour | Status |
|---|---|---|
| Conv2d, Linear | Matched and chained | Tested |
| BatchNorm and biases (1D) | Not scored, by design: their shared init makes them uninformative. Hashed in the digest. | Tested |
| Grouped / depthwise conv | Scored as a matrix, but the chain's input-column carry is skipped (width ≠ previous channel count), so a permuted depthwise layer is not re-aligned | Not tested |
| Residual blocks, concatenations | Chaining follows registration order, which is data-flow order only for sequential networks. A permuted residual stream is not re-aligned. A genuine unpermuted derivative is unaffected (identity matching). | Not handled |
| Embeddings | ≤ 8,192 rows: matched (permutation-invariant over tokens, which is too permissive). > 8,192: compared by row index (token id). | Tested (`test_matrices_above_the_matching_limit_are_compared_by_index`) |
| Attention heads | Not handled. The BERT script aligns FFN units only. | Not handled |

**Different channel counts** (verified in code, not documentation). Each candidate row is matched to at most one base row.
- **Fewer candidate rows** (structured pruning): score = sum of matched cosines ÷ live candidate rows. All-zero rows are dropped. A scattered 6-of-8 subset scores 1.0 (`test_channel_aligned_directional_similarity_recovers_scattered_channel_pruning`).
- **More candidate rows** (widening): unmatched rows add 0 to the sum but count in the denominator, so every unexplained channel costs score (`test_widened_candidate_is_penalised_for_unexplained_channels`). More than 2× the base's rows makes the matrix "incomparable", which counts against the exemption cap (`test_grossly_widened_matrix_is_incomparable_not_matched`).
- **Different kernel shape or input width** that can't be aligned: "incomparable".

## Q3. Is the permutation genuinely function-preserving?

**Measured, on this CNN only.** `permutation_check.permute_layer_output_channels` permutes a block's conv rows, its BatchNorm weight, bias, running mean and running variance, and the next layer's input columns. Over 2,000 fresh inputs (`reproduction_table`):

| Case | Max abs logit diff | Mean abs diff | Prediction agreement | Accuracy diff | Envelope-score diff | CKA vs unpermuted | JS vs unpermuted |
|---|---|---|---|---|---|---|---|
| Single layer | 9.5e-07 | 1.1e-07 | 1.0000 | 0.0000 | 1.1e-09 | 1.000000 | 1.5e-15 |
| All layers | 2.9e-06 | 2.1e-07 | 1.0000 | 0.0000 | 3.1e-09 | 1.000000 | 3.7e-15 |

It is true for this whole model, which is sequential. It is not established for residual, grouped or attention architectures (Q2).

## Q4. Can lineage be gamed by layer replacement? What fraction of foreign parameters gets through?

**Measured.** `defensibility_checks.splice_patterns` replaced every non-empty subset of the 6 weight matrices (first, middle, last, alternating, random: all 63 patterns) in a genuine fine-tune with an unrelated model's, for 5 donors. That is 315 splices, each scored undeclared and declared.

| Foreign matrices | n | Lineage admits, undeclared | Lineage admits, declared |
|---|---|---|---|
| 1 | 30 | 0 | 30 (by design: one declared replacement) |
| 2–6 | 285 | 0 | 0 |

Whole-matrix splicing never passes undeclared. Every foreign matrix scores like an unrelated model's (0.09–0.61) and fails the 0.508 floor.

**The real answer to "maximum foreign fraction" is per-channel, not per-layer.** When a fraction of *every* matrix's output channels is copied from an unrelated model:

| Foreign channels per layer | Lineage admits | Mean S_w | Mean min-layer |
|---|---|---|---|
| 10% | 5/5 | 0.862 | 0.715 |
| 25% | 5/5 | 0.655 | 0.565 |
| 50% | 0/5 | 0.367 | 0.308 |
| 75% | 0/5 | 0.206 | 0.148 |

The rule tolerates roughly a quarter of every layer being foreign. Residual-block replacement was not tested: the corpus has no residual blocks. Test: `test_attack_3_backdoored_block_spliced_into_a_genuine_fine_tune_is_blocked` (a re-trained, working backdoored block, declared and undeclared).

## Q5. What does "at most one matrix unscored" mean, and can it hide a backdoor?

**Tested and measured.**
- **Who declares:** the candidate's own ML-BOM, property `securemodelgate:replacedParameters` (`reference_resolver.declared_replaced_parameters`). It is untrusted.
- **The cap:** `pipeline.lineage_verdict_for` fails lineage when more than `max_exempt` (default 1) matrices are unscored for *any* reason: declared, missing, added or incomparable.
- **Identity:** a state-dict key of the base's architecture, one tensor per name. No tensors hide inside a "matrix".
- **Graph changes:** renaming gives missing + added, which is 2 and fails (`test_renamed_parameter_counts_as_missing_and_added`). With the file loader, a changed graph fails strict loading (Q22).
- **Which layer:** the attacker *can* declare any one layer, not only the head. The new `replaceable` policy restricts it (`test_policy_can_restrict_which_matrix_may_be_declared_replaced`).

**The attack.** The base backbone is kept, the head is re-initialised and re-trained with poison, and the ML-BOM declares `fc.weight` replaced (`defensibility_checks.declared_head`, n = 5):

| Head | Lineage (declared) | Decisions | Envelope scores | ASR |
|---|---|---|---|---|
| Backdoored | 5/5 pass | 5 ESCALATE | 0.136–0.260 | 0.94 |
| Benign | 5/5 pass | 5 ESCALATE | 0.133–0.259 | 0.00 |

The backdoor does not get admitted, but only because the envelope escalates every head replacement, benign ones included. The gate cannot tell a backdoored new head from a benign one. Both go to deep scan.

## Q6. Which shape manipulations were tested? Can foreign weights be made to look related?

**Partly; the core answer is yes.**

| Manipulation | Result | Evidence |
|---|---|---|
| Reshape / widen foreign layers by 1 channel | Scored, not skipped; BLOCK | `test_reshaped_foreign_layers_are_scored_not_skipped` |
| Widen 50× | Incomparable; counts against the cap | `test_grossly_widened_matrix_is_incomparable_not_matched` |
| Zero channels (padding) | Dead rows dropped; no score gain | `_match_rows` |
| Added matrices | Counted against the cap | `test_added_weight_matrices_count_against_the_exemption_cap` |
| Rename / reparameterise (e.g. weight-norm's g, v) | Missing + added; BLOCK (also false-blocks a benign reparameterisation) | `test_renamed_parameter_counts_as_missing_and_added` |
| Low-rank factorisation, channel split/merge | Graph change: fails strict loading, or exempt-count BLOCK | Loader test; not tested as an attack |
| Transpose, padding inside kernels | Not tested | — |

**Can foreign weights produce high cosine without lineage? Yes, by optimisation, not by reshaping** (Q7). Pulling an unrelated model's weights toward the public base for 150 steps gives S_w 0.87–0.98 and passes lineage.

## Q7. Can an independently trained model optimise S_w?

**Measured: yes.** `defensibility_checks.train_adaptive` starts from an unrelated model and poisons it with a patch trigger.
- **Loss:** CE + μ·(1 − ½·mean − ½·min of per-layer row cosine to the base).
- **Training:** 150 steps, n = 4 per cell.

| μ | Lineage pass | Mean S_w | Mean min-layer | ASR | Clean acc |
|---|---|---|---|---|---|
| 0 | 0/4 | 0.184 | 0.129 | 1.00 | 0.77 |
| 1 | 1/4 | 0.500 | 0.394 | 0.99 | 0.76 |
| 3 | 4/4 | 0.867 | 0.865 | 1.00 | 0.73 |
| 10 | 4/4 | 0.981 | 0.979 | 0.99 | 0.69 |

The lineage test is not a security boundary against an attacker who has the base, and the base is public by construction. Its conformal guarantee covers models exchangeable with independently trained ones, which this attacker is not.

## Q8. Can one attacker pass both tests at once?

**Measured: yes.** Same harness, with λ·(envelope score) added on a public probe draw.

| Start | μ | λ | Lineage pass | Envelope pass | **ADMIT** | ASR |
|---|---|---|---|---|---|---|
| Unrelated | 3 | 1 | 4/4 | 1/4 | **1/4** | 1.00 |
| Unrelated | 10 | 1 | 4/4 | 4/4 | **4/4** | 1.00 |
| Base (genuine) | 0 | 1 | 4/4 | 4/4 | **4/4** | 1.00 |
| Base (genuine) | 0 | 2 | 4/4 | 4/4 | **4/4** | 1.00 |

The full-objective attacker is admitted every time. Admitted models' clean accuracy (0.61–0.67) is above the base's 0.54, so utility gives no signal either. Pinned as a known gap by `test_known_gap_envelope_adaptive_poisoned_fine_tune_is_admitted`.

## Q9. Does S_w depend on initialisation? What is the unrelated-model distribution?

**Measured.** All 40 Pilot A unrelated models (`pilot_results/evaluation.json`):

| Statistic | Unrelated S_w | Unrelated min-layer | Genuine S_w (n = 70) |
|---|---|---|---|
| Mean | 0.186 | 0.132 | 0.980 |
| Std | 0.017 | 0.010 | 0.020 |
| Min | 0.150 | 0.090 | 0.940 |
| Median | 0.184 | 0.133 | 0.994 |
| Max | 0.232 | 0.147 | 1.000 |
| 95th pct | 0.212 | 0.145 | 1.000 |
| 99th pct | 0.225 | 0.146 | 1.000 |

Other recipes, judged against thresholds calibrated only on the default recipe: 0/25 admitted. Ranges by recipe:
- SGD 200 steps: S_w 0.18–0.24
- SGD lr 0.01: 0.18–0.21
- Adam 1e-3: 0.17–0.19
- Adam 1e-2: 0.19–0.21

**Initialisation matters a lot.** Three models sharing the base's random init but trained on a different data stream score S_w 0.70–0.72. Two of three pass lineage (both then ESCALATE on the envelope), and the third is blocked on the floor (0.491). S_w partly measures shared initialisation, not only shared training.

Per-layer, the classifier layer of unrelated models scores far higher (mean 0.44, max 0.61) than the conv layers (≤ 0.38). That is alignment inflation on a 4×64 matrix, and it is what sets the pooled floor.

**On real weights** (BERT, cosine-only, 20 calibration MultiBERTs; [`pilot_results_bert/`](pilot_results_bert/README.md)):
- **Unrelated MultiBERTs:** medians 0.111–0.112 against `bert-base-uncased`, and the best of their 1,440 per-matrix scores is 0.128.
- **Genuine descendants:** medians 0.966–1.000; the worst matrix of any descendant is 0.930.
- **No inflated layer:** with no small head scored, nothing like the synthetic classifier-layer effect appears.

A same-initialisation null (one MultiBERT seed's intermediate checkpoints) was not run.

## Q10. How far can a genuine fine-tune move before rejection?

**Measured** (`benign_sweeps`, 3 seeds each). Lineage never fails here. The envelope fails early:

| Fine-tune | Lineage | Min-layer | Max envelope | Decision |
|---|---|---|---|---|
| 8 steps, lr 0.005 (calibration recipe) | pass | 1.000 | 0.086 | 3 ADMIT |
| 40 steps, lr 0.005 | pass | 0.990 | 0.219 | 3 ESCALATE |
| 800 steps, lr 0.005 | pass | 0.927 | 0.219 | 3 ESCALATE |
| 200 steps, lr 0.05 | pass | 0.744 | 0.445 | 3 ESCALATE |
| 800 steps, lr 0.05 | pass | 0.608 | 0.307 | 3 ESCALATE |

**A benign fine-tune 5× longer than the calibration recipe is escalated every time.** The envelope only covers derivatives like the calibration ones. Calibrating on mixed depths (8/40/200 steps, n = 21) raises the threshold to 0.2387. That gives 0/21 false blocks across all depths, but detects 27/30 non-adaptive backdoors instead of 29/30 (section 8). That trade-off is the operator's to make, and it must be calibrated on the derivative distribution actually expected.

## Q11. Pruning: at what level does lineage fail?

**Measured.**

| Pruning | Min-layer | Envelope | Decision |
|---|---|---|---|
| Magnitude 10/30/50% | 1.000 / 0.988 / 0.945 | ≤ 0.028 | ADMIT |
| Magnitude 70% | 0.838 | 0.112 | ESCALATE |
| Magnitude 90% | 0.551 | 0.480 | ESCALATE |
| Structured: 25% channels zeroed | 0.9999 | > threshold | ESCALATE |
| Structured: 50% channels zeroed | 0.9999 | > threshold | ESCALATE |
| Structured: channels removed (scattered) | 1.0 (numpy) | — | `test_..._recovers_scattered_channel_pruning` |

Lineage never fails from pruning up to 90%. Rejections are envelope escalations, because the pruned model's behaviour changed (no recovery fine-tune was run). That is the implementation working as designed, not a handling failure. Block pruning was not tested.

## Q12. Quantization: what is compared?

**Measured.** **Dequantized float weights.** Every quantizer here rounds to an integer grid and then multiplies back; the scorer and the probes see float tensors. Raw integer tensors are not a supported input: the loader accepts integer dtypes only for buffers, and a model holding raw int8 weights would need its own dequantization step first.

| Format | Min-layer | Decision |
|---|---|---|
| FP16, BF16 round-trip | 0.9999 | ADMIT |
| INT8 symmetric, per tensor / per channel | 0.9998 / 0.9999 | ADMIT |
| INT4 symmetric per channel | 0.9947 | ADMIT |
| 4 levels per tensor | 0.848 | ADMIT |
| 2 levels per tensor | 0.809 | ESCALATE (envelope 0.785) |

## Q13. The exact CKA and JS formulas

**Tested** (`behavioral_delta.py`).

**CKA:**
- **Layers:** `layer1`–`layer4` block outputs, after ReLU, via forward hooks.
- **Shape:** each block output is global-average pooled to (n_probes × channels).
- **Formula:** columns are centred over the batch; CKA = ‖YᵀX‖²_F / (‖XᵀX‖_F·‖YᵀY‖_F), clipped to [0, 1]. The minimum over the four layers is used.
- **Constant representation:** denominator < 1e-12 gives 0.0, so the envelope reads 1.0 and the model is escalated (the fail-safe direction).

**JS:**
- **Input:** softmax probabilities at temperature 1, not logits.
- **Computation:** each vector is clipped to ≥ 1e-10 and renormalised, then base-2 JS (bounded [0, 1]) is computed per probe and averaged.
- **Class mismatch:** a different class count raises `ValueError`.

**NaN/Inf:** any non-finite CKA or JS now raises and the gate denies (finding 3).

**CPU/GPU differences:** not measured. Only CPU has been run.

## Q14. Probe-set security

**Measured.**
- **Is it secret?** No. The pool is `make_probe_loader(seed)`, the seed is in the attestation, and a fixed seed (20260101) is used throughout.
- **Can the attacker train against it?** Yes (Q8).

Detection by probe draw, each draw calibrated separately (`probe_variation`):

| Probe seed | Threshold | Benign false blocks | Non-adaptive detected | Adaptive, public distribution | Adaptive, exact gate pool |
|---|---|---|---|---|---|
| 20260101 (gate's) | 0.1108 | 0/10 | 15/15 | 0/4 | 0/4 |
| 11 | 0.1344 | 0/10 | 15/15 | 0/4 | 3/4 |
| 12 | 0.1098 | 0/10 | 15/15 | 0/4 | 3/4 |
| 13 | 0.1084 | 0/10 | 15/15 | 0/4 | 1/4 |

An attacker who overfits to a known exact pool is caught 7 times out of 12 under fresh draws. So a secret per-admission draw has real value, but only against that attacker. One who trains against the probe *distribution* evades every draw. These are synthetic i.i.d. probes; natural-image probes may differ (E7).

## Q15. Can the seed be predicted? Is it committed before the candidate can adapt?

**No and no.**
- **Who chooses the seed:** the caller of `run_admission` (default 0; the demo and pilots use 20260101). The gate does not draw it.
- **The commitment:** an unsalted `sha256(str(seed))`, computed at attestation time and published nowhere beforehand. It is brute-forceable (finding 6).

The required sequence is not implemented:
1. Fix the candidate digest.
2. Draw the seed from a CSPRNG.
3. Publish a salted commitment to a transparency log.
4. Generate the probes.
5. Score.

## Q16. Conformal calibration at code level

**Tested.** `conformal.conformal_quantile`:
- **Sort:** ascending.
- **Rank:** k = ⌈(n+1)(1−α)⌉, clamped to n, and the k-th smallest score is the threshold.
- **Comparison:** strict `>`. A score equal to the threshold does not exceed it. For the envelope that means ADMIT; for lineage it means *not* verified (fail-closed).

Worked cases:
- **n = 19, α = 0.05:** k = 19, the maximum (`test_conformal_quantile_saturates_at_max_for_n19_alpha05`).
- **n = 99, α = 0.05:** k = 95.
- **n = 20, α = 0.05:** k = 20, the maximum. That is why Pilot A's envelope threshold is the largest calibration score.

**Edge cases:** ties keep a stable sort. NaN/Inf calibration or candidate scores raise (`test_conformal_quantile_rejects_nan_calibration_score`, `test_envelope_and_lineage_test_reject_nan_candidate_score`).

**Floating-point rounding:** checked against exact rational arithmetic for n = 1–5000 at ten α values from 0.001 to 0.5. There were 0 disagreements.

## Q17. Calibration leakage

**Measured; one real leak found and fixed.**
- **Models appearing in both sets:** found and fixed for the benign population (finding 1). `run_evaluation` now raises on any calibration/test digest overlap.
- **Unrelated models:** calibration and test seeds (1000–1019 vs 1020–1039) are disjoint, and the floor uses the calibration half only.
- **Learned preprocessing:** there is no learned preprocessing or normalisation statistic. CKA centres within each probe batch.

Hyperparameters chosen after seeing results:
- The cosine-only default (spectral weight 0) was chosen on a corpus that included Pilot A's population.
- The backdoor recipe was tuned for ASR ≥ 87%, not for detection.
- `max_exempt = 1`, `MAX_ROW_GROWTH = 2` and `MAX_MATCH_ROWS = 8192` were set a priori.

## Q18. The 69/70 result: per-model artifact and the one failure

**Measured.** `pilot_results/evaluation.json` holds all 110 models (40 unrelated, 40 benign, 30 backdoored). Each record has:
- id, population, transformation, seed, weight digest;
- every per-layer S_w and the exemptions;
- both lineage thresholds, the lineage verdict and failure reason;
- every per-layer CKA, JS, the envelope score and threshold;
- the gate decision, ASR and clean accuracy.

**The failure:** `backdoor-warped-3009`, a genuine (backdoored) fine-tune. Its S_w is 0.971 (above 0.214), but its fc layer scores **0.484** against the **0.508** floor, so lineage fails and it is blocked. It is a correct security outcome on a wrong lineage label. Its cause is Q9's classifier-layer inflation: unrelated models' fc layers score up to 0.61, which pushes the pooled floor above some genuine fc scores.

## Q19. The 2/20 benign false blocks

**Measured; it was a systematic problem.**
- **Which models:** `benign-30` and `benign-33`, both 8-step fine-tunes (seeds 2030, 2033).
- **Lineage:** passed (min-layer ≈ 0.98).
- **Why they were escalated:** their JS (0.098, 0.106) exceeded a threshold of 0.0915.
- **The underlying cause:** the leaky population of finding 1. The calibration set held only 7 distinct fine-tunes, so the threshold was a maximum over 11 distinct models, 13 of them near-zero duplicates.

After the fix the 2/20 is replaced by 0/20, and the real weakness is Q10's depth sensitivity.

## Q20. Per-attack-family results

**Measured; the families are synthetic stand-ins.**

| Trigger | n | Detected | ASR | Clean acc | Decisions |
|---|---|---|---|---|---|
| Patch (BadNets-like) | 10 | 9 | 0.996 | 0.613 | 9 ESCALATE, **1 ADMIT** |
| Blended | 10 | 10 | 0.985 | 0.594 | 10 ESCALATE |
| Warped (coarse WaNet stand-in) | 10 | 10 | 0.999 | 0.509 | 9 ESCALATE, 1 BLOCK |

These are not BadNets/Blended/WaNet/SIG/Input-aware implementations, and SIG and Input-aware have no stand-in at all. BackdoorBench (E1–E9) has not been run.

## Q21. Is the false-block rate random variance or systematic?

**Measured.** It is systematic, in two ways:
- The old 2/20 came from the leak (Q19).
- Any fine-tune deeper than the calibration recipe is escalated (Q10).

On the current population, 0/20 bounds the rate only below 16.8% (95%, two-sided). To show ≤ 5% with zero failures needs n ≥ 59 (one-sided 95%) or n ≥ 72 (two-sided Clopper–Pearson). A 1% claim needs n ≥ 99 calibration models (`min_calibration_size`). No run at that scale has been done; `securemodelgate evaluate --n-benign 200` would do it on the synthetic corpus.

## Q22. Architecture mismatch

**Tested; deterministic.**
- **Through the file loader:** `loader.load_candidate` loads into the gate's own base architecture with `strict=True`. Any name or shape mismatch raises `UnsafeModelFile`, and the caller denies.
- **Through the in-memory API:** mismatched matrices are missing, added or incomparable. More than one means lineage fails, which is **BLOCK**. A wholly different architecture scores S_w = 0 and is blocked.
- **No declared base at all:** Tier 2, which is **ESCALATE** and therefore denied at the webhook.

## Q23. ML-BOM and attestation tampering: the enumerated matrix

**Tested.**

| Tampered | Result | Test |
|---|---|---|
| Base purl / any ML-BOM field after attestation | mlbomDigest mismatch, reject | `test_any_single_tampered_input_fails_closed[mlbom]` |
| Base digest (declared ≠ supplied base) | Unverified, Tier 2 | `test_base_model_digest_mismatch_is_treated_as_unverified` |
| Signer identity not allow-listed | Unverified, Tier 2 | `test_resolve_reference_untrusted_signer` |
| Allowed signer name over a digest it never signed | Tier 2 (with the trusted-base registry) | `test_signer_substitution_is_caught_by_the_trusted_base_registry` |
| Candidate bytes, including BatchNorm buffers | Subject mismatch, reject | `[candidate]`, `test_attack_5_...` |
| Each calibration set (drop, perturb 1e-9, append) | Digest mismatch, reject | `[lineage_calibration_scores]`, `[layer_floor_...]`, `[envelope_...]` |
| Probe pool | Digest mismatch, reject | `[probe_loader]` |
| Gate version | Reject | `[expected_gate_version]` |
| Threshold, verdict, scores (re-signed with the key) | Reject: not the conformal quantile / verdict doesn't follow / recomputed scores differ | `test_a_validly_signed_but_inconsistent_predicate_is_rejected` (6 cases), `test_attack_6_...` |
| Signature, payload type, malformed envelope | Reject without raising | `test_tampered_signature_fails`, `test_signed_payload_of_the_wrong_type_is_rejected`, `test_malformed_envelopes_fail_closed_without_raising` |

There is no real OMS bundle: the OMS check is a structural stand-in (Q43).

## Q24. Replay of A's attestation onto B

**Tested.**
- **A's ADMIT presented with backdoored model B:** denied, subject digest (`test_attack_4_reusing_a_benign_models_admit_attestation_is_denied`).
- **Modified bytes, ML-BOM, each calibration set, probe pool:** each denied (Q23).
- **Modified threshold:** denied, even when re-signed.

## Q25. Freshness after calibration, version, base, signer or policy change

**Tested where implemented.** `verify_attestation` takes the current state and checks the attestation against it:

| Changed after issue | What verify does | Test |
|---|---|---|
| Calibration sets | The verifier passes the *current* sets; the old digests no longer match, so reject | Tamper cases |
| Gate version | `expected_gate_version`: exact match or reject | `[expected_gate_version]` |
| Base revoked | `trusted_bases` registry (digest → signer); removed means reject | `test_revoked_base_fails_verification` |
| Signer revoked | Same registry: drop the signer's digests | Same mechanism |
| Policy (α, max_exempt) | `expected_policy`; differs means reject | `test_attestation_under_a_superseded_policy_is_rejected` |
| Age | `max_age_seconds` on the signed timestamp | `test_stale_attestation_is_rejected_when_freshness_is_required` |

All of these are optional arguments. A verifier that omits them does not get the check. There is no key-revocation or rotation mechanism for the gate's own signing key (Q43).

## Q26. Base substitution

**Tested.** The resolver fetches nothing: the caller supplies `base_model`, and `run_admission` hashes it. If that hash ≠ the ML-BOM's declared digest, the reference is unverified and goes to Tier 2 (`test_base_model_digest_mismatch_is_treated_as_unverified`). Calibration computed for another base is refused when `calibration_base_digest` is passed (`test_calibration_computed_for_another_base_is_refused`).

## Q27. Signer substitution

**Tested.**
- **Unauthorised signer:** Tier 2.
- **Authorised signer with the wrong model digest:** Tier 2, through digest binding.
- **Authorised signer name claimed over a digest that signer never signed:** passes the structural stand-in on its own. It is caught only with the `trusted_bases` registry, or with real Sigstore, which is not implemented.

## Q28. Fail-closed behaviour

**Tested where the gate owns the step.** Every row below is an exact outcome.

| Failure | Decision | Mechanism |
|---|---|---|
| ML-BOM missing or empty | ESCALATE (Tier 2), denied at webhook | `test_tier2_no_declared_ancestor` |
| ML-BOM malformed | DENY | `webhook.review_candidate` catches the exception (`test_attack_7_...[mlbom]`) |
| Base unavailable | DENY (ValueError) | `test_verified_reference_without_base_model_raises` |
| Calibration unavailable / NaN | DENY (ValueError) | `test_verified_reference_without_layer_floor_calibration_raises`, NaN tests |
| Probe pool unavailable | DENY (ValueError) | Same check |
| Signature unverifiable / malformed | DENY | `review_attestation` |
| Model file unsafe or unparsable | `UnsafeModelFile`: the caller must deny | Loader tests |
| Scorer exception, NaN/Inf weights | DENY | `test_attack_7_...[nan_weights]` |
| CKA or JS NaN | DENY | `test_attack_8_...` |
| Weights mutated during scoring | DENY (digest re-checked after scoring) | `pipeline.run_admission` |
| OMS / Sigstore server unavailable | **Not applicable**: no network code; the registry is local | — |
| GPU OOM, timeout, process crash | **Not handled in code.** Depends on the webhook's `failurePolicy: Fail`, which is not shipped | — |

## Q29. Kubernetes webhook behaviour

**Not done.** There is no webhook server, no TLS, no `ValidatingWebhookConfiguration`, and no cluster test. What exists and is tested is the Python layer:
- `admission_response` (only ADMIT → `allowed: true`, otherwise 403);
- `review_candidate` (any gate exception → deny);
- `review_attestation` (verified *and* ADMIT, or deny).

Timeout and network-failure behaviour depend entirely on `failurePolicy: Fail`. That must be demonstrated on a real cluster (kind/k3d) before the paper says the gate fails closed in Kubernetes.

## Q30. Is the deep-scan result bound to the admitted digest?

**Tested.** `run_admission(deep_scan_report={tool, version, flagged, subjectDigest})` uses the result only if `subjectDigest` equals the candidate's digest. Otherwise the candidate stays ESCALATE, and the attestation records `bound: false` and the rejection (`test_deep_scan_report_must_be_about_the_candidates_bytes`). A modified model has a new digest, so the old scan doesn't apply. The legacy `deep_scan_flagged` boolean is unbound and recorded as such. The deep scanner itself is external and not implemented.

## Q31. TOCTOU

**Partly.**
- **Inside the gate:** the digest is computed before scoring and re-checked after. A mismatch refuses to attest.
- **At admission:** `review_attestation` recomputes the digest from the presented model.
- **Not bound:** what the container loads *after* admission. The pod pulls its model later, and nothing in this code re-hashes it at load. A deployment needs the runtime loader to compute `weight_digest` and compare it with the attestation subject.

The subject is a canonical state-dict digest, **not** the OMS manifest's file digest, so the two are not yet linked.

## Q32. Performance

**Measured, only at small scale** (4-core CPU, no GPU).

| Item | Time |
|---|---|
| S_w + D_b, Pilot A median (25.5k params, 64 probes) | ~120 ms |
| S_w at 0.1M / 0.4M / 1.6M params | 0.02 / 0.04 / 0.13 s |
| D_b at the same sizes | 0.13–0.14 s |
| Sign + verify (HMAC) | 0.15 ms |
| `weight_digest` | 0.5 ms |

Not measured:
- peak memory, GPU memory, CPU utilisation;
- calibration latency;
- 10M–1B-parameter models.

The matching cost matrix for an n-row layer is n²·8 bytes (75 MB at n = 3072). A BERT-base estimate from the matching numbers is about 20–30 s of matching per model. That is an estimate, not a measurement.

## Q33. Does it scale? Is Hungarian the bottleneck?

**Measured.** Matching time (768 columns):
- 64 rows: 0.001 s
- 768 rows: 0.16 s
- 2048 rows: 0.63 s
- 3072 rows: 1.35 s

That is super-linear in rows, as expected for O(n³). Cost grows with the sum of rows³ over layers, not with parameter count, so wide layers are the bottleneck. Vocabulary-sized matrices are compared by index (linear time) above 8,192 rows. Candidates wider than 2× the base are refused, which bounds the attacker's control of the cost.

## Q34. Determinism

**Measured, on one machine.** S_w, min-layer, min CKA and JS were bitwise identical across:
- 20 repeated runs (1 distinct result);
- 1 vs 4 threads;
- probe batch size 16 vs 64.

A 1e-6·std weight perturbation moved them by at most 8.6e-8. Among the 70 held-out Pilot A models, the closest any score sits to its threshold is 0.0016 (min-layer vs floor), 0.0029 (S_w vs median) and 0.0049 (envelope), all far above that noise. Calibration models sit exactly on the threshold by construction, since the threshold is one of their scores. CUDA, other GPUs, other CPUs and other PyTorch versions were not tested.

## Q35. Numerical robustness

**Tested.**

| Input | Result |
|---|---|
| NaN/Inf weights | ValueError, DENY |
| NaN/Inf calibration or candidate score | ValueError |
| Logit overflow | NaN JS, now an error |
| Zero rows | Dropped |
| All-zero layer | Score 0, floor fails |
| Constant representation | CKA 0, ESCALATE |
| Empty calibration set | ValueError |
| Probability underflow | Clipped at 1e-10 |
| Near-zero singular values | SVD is skipped at the default spectral weight 0 |

Not tested: overflow in the matching and empty (0-row) weight tensors.

## Q36. Can a third party recompute the verdict?

**Tested.** `verify_attestation(..., base_model=base)`:
1. Recomputes S_w, the per-layer minimum, the exemption list, min CKA and JS, and compares them within 1e-6.
2. Re-derives all three thresholds from the digest-bound calibration sets at the recorded α.
3. Re-derives the tier and verdict from those numbers.

Pinned by `test_verification_recomputes_scores_from_the_base` and the six re-signed-predicate cases. It needs the same probe pool (digest-bound) and calibration sets. Cross-platform recomputation is untested (Q34).

## Q37. Gate-version compatibility

**Partly.** The predicate records `gateVersion` (now the package version) and a versioned `predicateType` (`.../lineage/v0.1`). The verifier accepts an exact `expected_gate_version` match or rejects. There is no compatibility policy: a v2 gate either rejects a v1 attestation or must re-run admission. There is no migration or re-validation path.

## Q38. Calibration-set integrity

**Tested.** Dropping one score, perturbing one by 1e-9, or appending one each breaks verification. Reordering also breaks it: the digest is over the ordered list, which fails closed. The attestation binds the calibration *scores*, not the models or preprocessing that produced them. A changed calibration model that happens to yield identical scores is not detected.

## Q39. Is the model loaded before it is inspected?

**Tested.** The scoring API takes models already loaded by the caller. `loader.py` is the boundary for untrusted files:
- `.safetensors` is parsed without code execution.
- `.pt/.pth/.bin` are loaded only with `torch.load(weights_only=True)`.
- Every other format is refused.

`test_malicious_pickle_is_refused_and_not_executed` builds a live `os.system` payload and shows two things:
1. A plain `torch.load` executes it, for the legacy raw-pickle format even though the load then fails on the magic number.
2. The loader refuses both forms without executing them.

Pickle scanners (modelscan, picklescan) are not integrated.

## Q40. `trust_remote_code`

**Answered.** There is no code path that executes model-supplied code. The architecture always comes from the gate: `load_candidate(path, build_base_architecture)` with strict loading. A model that needs a custom architecture the gate does not already have can't be scored at Tier 1 and must go to Tier 2. The BERT experiment script loads with `AutoModel.from_pretrained(name)` (default `trust_remote_code=False`, standard BERT classes) and is research tooling, not part of the gate.

## Q41. Can malformed files or metadata crash or compromise the gate?

**Partly.**

Tested:
- malicious pickle (both formats);
- unsupported suffix;
- NaN tensors;
- symlinks;
- an architecture mismatch;
- a malformed ML-BOM (denied, not crashed).

Not supported, so refused: ONNX, GGUF, LoRA adapters.

Not done: fuzzing of the safetensors header or ML-BOM JSON.

## Q42. The gate's own supply chain

**Not done.** `pyproject.toml` has unpinned lower bounds only. There is:
- no lockfile;
- no SBOM;
- no vulnerability scan;
- no container image.

Versions this was tested with:

| Component | Version |
|---|---|
| Python | 3.11 |
| torch | 2.14.0+cu130 |
| numpy | 2.4.6 |
| scipy | 1.17.1 |
| safetensors | 0.8.0 |
| Crypto | Python standard library (hmac, hashlib) |

## Q43. Cryptographic implementation

**Partly.** Sign → verify → tamper → fail is tested at every level (Q23). What the code actually does:
- `LocalHmacSigner` is HMAC-SHA256 over DSSE's PAE (finding 5), with a shared secret, a keyid match and a constant-time comparison.
- The demo secret is hard-coded in `cli.py` and must not be used beyond the demo.

None of the following exists:
- Sigstore keyless signing, a KMS, certificate identity or a trust root;
- key rotation or expiry;
- revocation for the gate's key.

`verify_oms_signature` is structural only. The production cryptographic path is not built, and the paper must not say it is.

## Q44. Real models and a machine-readable evaluation.json

**Partly.**
- **Synthetic corpus:** `pilot_results/evaluation.json` follows the requested schema, extended. The model, its base, true lineage (population), transformation, every score, threshold and pass flag, ASR and the final verdict are all there.
- **BERT/MultiBERTs (lineage only):** run by the author on a Mac, cosine-only and with a 0.5 spectral blend. Base `bert-base-uncased`; 10 descendants (7 public fine-tunes, plus INT8, 30%-pruned and FFN-permuted copies); 20 calibration and 5 held-out MultiBERTs.
  - Cosine-only: 10/10 descendants accepted, 5/5 held-out MultiBERTs rejected, and the splice blocked by the floor (worst matrix 0.056) after a median-only rule would have admitted it.
  - Weakest genuine matrix 0.930 against a floor of 0.128.
  - The script writes per-row JSON (name, kind, median, min, exemptions, accept flags). The numbers in [`pilot_results_bert/`](pilot_results_bert/README.md) are transcribed from the console logs; the JSON files themselves still need committing.
- **BackdoorBench (E1–E9):** not run (no GPU).

# The short "must answer" list

| # | Question | Answer | See |
|---|---|---|---|
| 1 | Can an attacker optimise S_w? | Yes: 4/4 pass lineage at μ ≥ 3 | Q7 |
| 2 | Jointly optimise S_w + CKA + JS? | Yes: 4/4 admitted, ASR 1.00 | Q8 |
| 3 | Can splicing bypass the rule beyond 3/3? | Not whole-matrix (0/285 undeclared). Partial-channel: 25% foreign per layer passes | Q4 |
| 4 | Can declared replacements hide malicious functionality? | Not past the envelope (5/5 ESCALATE), but only because every head replacement escalates | Q5 |
| 5 | Can reshaping or reparameterisation forge lineage? | Reshaping no; optimisation yes | Q6, Q7 |
| 6 | Can an unrelated model be optimised into acceptance? | Yes | Q7 |
| 7 | Can the probe set be attacked? | Yes; a secret draw helps only against exact-pool overfitting | Q14 |
| 8 | Old attestation replayed on another model? | No: rejected | Q24 |
| 9 | Model swapped between scan and admission? | No inside the gate; not bound after admission | Q31 |
| 10 | Deep-scan output bound to the digest? | Yes (`deep_scan_report`) | Q30 |
| 11 | Every failure path fail-closed? | Every path the gate owns; timeouts and crashes depend on `failurePolicy` | Q28 |
| 12 | Can malformed models exploit the gate? | Pickle: no (tested). Fuzzing not done | Q39, Q41 |
| 13 | Cause of the 69/70 error? | `backdoor-warped-3009`, fc 0.484 < floor 0.508 | Q18 |
| 14 | Cause of the 2/20 false blocks? | A leaky population; now 0/20 | Q19 |
| 15 | Does the conformal guarantee hold under the protocol? | Envelope: only for derivatives like the calibration ones. Floor: no guarantee (pooled across unlike layers). Adaptive: none | Q10, Q16 |
| 16 | S_w vs fine-tuning depth? | Lineage holds to 800 steps at lr 0.05 (min-layer 0.61); envelope escalates beyond 8 steps | Q10 |
| 17 | Pruning? | Lineage holds to 90%; envelope escalates at ≥ 70% | Q11 |
| 18 | Quantization? | FP16/BF16/INT8/INT4 all ADMIT; 2-level escalates | Q12 |
| 19 | Permutation? | Exactly recovered on sequential networks | Q1, Q3 |
| 20 | Across seeds? | Unrelated 0.15–0.23 vs genuine ≥ 0.94; shared init 0.70 | Q9 |
| 21 | Across architectures? | Mismatch is BLOCKed; residual/attention not supported | Q2, Q22 |
| 22 | Real models? | Lineage only: BERT 10/10 descendants accepted, 5/5 held-out MultiBERTs and a splice rejected, margin ≈ 0.80 | Q44 |
| 23 | Each backdoor family? | Patch 9/10, blended 10/10, warped 10/10 (stand-ins) | Q20 |
| 24 | Adaptive attacks on the real benchmark? | Not run; synthetic: admitted | Q8 |
| 25 | Latency per model? | ~120 ms at 25k params; 0.27 s at 1.6M | Q32 |
| 26 | Peak memory? | Not measured; matching n²·8 B | Q32 |
| 27 | Does Hungarian scale? | O(n³) in rows; 1.35 s at 3072 | Q33 |
| 28 | Deterministic? | Bitwise, on one CPU | Q34 |
| 29 | Independently recomputable? | Yes, within 1e-6 | Q36 |
| 30 | Does Kubernetes fail closed? | Not tested on a cluster | Q29 |
| 31 | Timeout? | Depends on `failurePolicy: Fail`, not shipped | Q28 |
| 32 | Base registry unavailable? | Local registry; no network path | Q28 |
| 33 | Sigstore unavailable? | Sigstore not implemented | Q43 |
| 34 | Calibration versions? | Digest-bound and checked; no version store | Q25, Q38 |
| 35 | Gate versions? | Exact-match only | Q37 |
| 36 | Key rotation and revocation? | Not implemented | Q43 |
| 37 | Production Sigstore/KMS working? | No | Q43 |
| 38 | Webhook tested on Kubernetes? | No | Q29 |

# End-to-end attacks, and the exact test for each

Every test in `tests/test_end_to_end_attacks.py` first shows that the attack works (ASR ≥ 0.9, or predictions changed), then runs it through the entry points a deployment uses: `webhook.review_candidate` and `webhook.review_attestation`.

| Attack | Outcome | Test |
|---|---|---|
| Lineage forgery: unrelated backdoored model claims the base | BLOCK, denied | `test_attack_2_unrelated_backdoored_model_claiming_the_base_is_blocked` |
| Splicing: a working backdoored block (re-trained on the frozen genuine front), declared or not | BLOCK, denied | `test_attack_3_backdoored_block_spliced_into_a_genuine_fine_tune_is_blocked` |
| Reshaping the foreign layers | BLOCK | `test_reshaped_foreign_layers_are_scored_not_skipped` |
| Permutation of a genuine model (must *not* be stopped) | ADMIT | `test_fully_permuted_genuine_derivative_is_admitted` |
| Non-adaptive poisoned fine-tune | ESCALATE, denied | `test_attack_1_poisoned_fine_tune_declaring_its_real_base_is_denied` |
| Attestation replay onto another model | Denied | `test_attack_4_reusing_a_benign_models_admit_attestation_is_denied` |
| Model substitution after admission (BatchNorm statistics) | Denied | `test_attack_5_editing_batchnorm_statistics_after_admission_is_denied` |
| Forged ADMIT inside a re-signed predicate | Denied | `test_attack_6_forged_admit_inside_a_resigned_predicate_is_denied` |
| Malformed ML-BOM, NaN weights | Denied, no crash | `test_attack_7_malformed_inputs_are_denied_not_crashed` |
| Logit overflow to blind the envelope | Denied | `test_attack_8_logit_overflow_cannot_make_the_envelope_ignore_js` |
| Malicious pickle | Refused, not executed | `test_malicious_pickle_is_refused_and_not_executed` |
| **Adaptive behavioural attack (A2 + A4)** | **ADMITTED: known gap** | `test_known_gap_envelope_adaptive_poisoned_fine_tune_is_admitted` |
| **Joint adaptive attack (S_w + envelope)** | **ADMITTED 4/4: known gap** | `defensibility_checks.adaptive_attacks` |
| TOCTOU after admission (container loads different bytes) | Not covered | Q31 |

# What this means for the paper

- **Q1 claim must narrow.** The lineage test establishes weight-space proximity to the declared base for non-adaptive submissions. It is not a forgery-proof provenance check (Q7, Q8), and it partly reflects shared initialisation (Q9).
- **Pilot A numbers must change.** They become 0/20 false blocks, 29/30 detection, AUROC 0.997 and 69/70 lineage, with one non-adaptive backdoor admitted. The 10% figure must go, with the reason stated.
- **The envelope's coverage is the calibration population.** A fine-tune 5× longer than calibrated is escalated every time (Q10).
- **What the gate verifiably does:** it binds a signed, recomputable, fail-closed verdict to exact model bytes, and triages non-adaptive submissions. It does not stop an adaptive attacker in either test.
