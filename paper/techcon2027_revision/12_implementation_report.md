---
title: "SecureModelGate: Implementation Report"
subtitle: "What was built, what each part is for, what it was tested against, and what it still cannot do"
author: "Dharanidharan Senthilkumar, OLAA, CDRM – Cyber Defense Center, HPE, Bengaluru"
date: "1 October 2026 · companion to the Tech Con 2027 submission · code: securemodelgate/ (commit ea6b88b and later)"
---

# 1. Purpose of this report

The Tech Con paper (`02_paper.md`) argues for a design. This report
documents the **implementation** of that design: every module in
`securemodelgate/`, the reason each one exists, the attack or failure it
answers, the evidence that it works, and the places where it does not.
It is meant to let a reviewer check any claim in the paper against
running code.

Read every number below with one scope statement in mind:

> All measurements in this report come from a **synthetic corpus**: a
> 4-stage CNN (`TinyConvNet`) trained on Gaussian-noise images. They
> show the pipeline is implemented correctly and survives the attacks
> listed. They are **not** the paper's PreAct-ResNet-18 / CIFAR-10 /
> BackdoorBench evaluation (Experiments E1–E9). That evaluation has not
> been performed: this environment has no GPU, and the CIFAR-10 mirror
> is blocked. The real-model BERT lineage run has been done by the
> author on their own machine (Section 5.5).

# 2. The problem and the design in brief

**Problem.** Enterprises admit models that are *derived* from a trusted
base (fine-tunes, LoRA adapters, pruned or quantized variants) rather
than published by the trusted vendor. OpenSSF Model Signing proves a
file's bytes are unchanged since signing. A derivative is different
bytes, so the signature says nothing about it. Its claimed parentage is
an optional, self-reported `base_model` field. Two questions go
unanswered at admission:

- **Q1 (lineage):** is this model really a descendant of the signed base
  it names?
- **Q2 (behaviour):** has it drifted from that base in ways benign
  derivatives do not?

**Design.** SecureModelGate answers both questions at admission time
and binds the answers into a signed, verifiable record:

1. It reads the declared base from the candidate's CycloneDX ML-BOM and
   verifies the base's signature.
2. It scores weight-level lineage (S_w) and behavioural drift (D_b)
   against that verified base.
3. It turns both scores into decisions using split-conformal thresholds
   calibrated on reference populations.
4. It routes the candidate to one of three risk tiers.
5. It signs an in-toto attestation that records every score, threshold
   and calibration digest.
6. It maps the verdict onto a Kubernetes admission response.

# 3. System overview

| Stage | Module | Question it answers | Output |
|---|---|---|---|
| Reference resolution | `lineage/reference_resolver.py`, `pipeline.py` | Which base does the candidate claim, is it signed by an allowed publisher, and is the model we are about to score against actually that base? | `ReferenceResolution` (verified or not, with reason) |
| Weight lineage | `lineage/lineage_score.py` | Q1: do the candidate's weight matrices descend from the base's? | S_w, per-layer scores, list of unscored matrices |
| Behavioural delta | `lineage/behavioral_delta.py` | Q2: do its internal representations and outputs stay close to the base's? | min-layer CKA, JS divergence, envelope score |
| Thresholds | `lineage/conformal.py` | What score counts as "unrelated" or "anomalous", with a stated error rate? | `ConformalVerdict` per test |
| Lineage rule | `pipeline.lineage_verdict_for` | Does lineage verify once every known attack on it is accounted for? | combined verdict |
| Tier routing | `lineage/admission_policy.py` | What happens to this model? | Tier 0/1/2 and ADMIT / ESCALATE / BLOCK |
| Attestation | `lineage/intoto_attestation.py` | What exactly was decided, on what evidence? | signed DSSE envelope with an in-toto predicate |
| Verification | `verify.py` | Does this attestation belong to *these* artifacts? | pass / fail with reason |
| Admission | `webhook.py` | What does Kubernetes do with the verdict? | AdmissionReview response |

`pipeline.run_admission` wires the stages together; one call takes a
candidate model and its ML-BOM and returns the decision and a signed
attestation.

# 4. Components: what, why, evidence, limits

## 4.1 Reference resolution

**What it does.** `resolve_reference` reads the first ancestor in the
ML-BOM's `pedigree.ancestors` (purl, OMS bundle digest, signer identity)
and checks the signer against an allow-list. `run_admission` then
recomputes the digest of the base model it was handed and compares it
with the digest the ML-BOM declared. Any mismatch marks the reference
**unverified**, which routes the candidate to Tier 2.

**Why.** The original Parasparam prototype compared models against an
undefined "clean reference", and its behavioural stage contributed
nothing. Anchoring every comparison to a declared, signed base is what
makes Q1 and Q2 answerable at all. The digest check exists because
resolution only examines the candidate's *claim*. Without it, a stale
cache, a compromised mirror or a lookup bug upstream could hand the gate
the wrong base. The gate would then score against it and still sign
"verified".

**Evidence.**

- Before the digest check, a candidate derived from a decoy model was
  scored against that decoy and admitted (S_w = 1.0), and its
  attestation claimed the real base's digest.
- After the check, the same case routes to Tier 2.
- Regression test: `test_base_model_digest_mismatch_is_treated_as_unverified`.

**Also here.** `declared_replaced_parameters` reads a
`securemodelgate:replacedParameters` property, which lists weight
matrices the publisher says were replaced (typically a re-initialised
classification head). This is a SecureModelGate convention, not a
CycloneDX standard field.

**Limit.** `verify_oms_signature` is a **structural stand-in**: it checks
the allow-list and digest format, not a cryptographic Sigstore
signature. Real verification (Fulcio certificate chain, Rekor inclusion)
is a drop-in replacement for that one function and is not implemented.

## 4.2 Weight lineage score S_w

**What it does.** For each weight matrix of the base, in order:

1. **Match output channels.** The candidate's output channels are
   matched to the base's by optimal assignment (Hungarian algorithm on
   pairwise cosine similarity). The layer's score is the mean cosine of
   the matched pairs.
2. **Carry the correspondence forward (sequential alignment).** Each
   layer's channel correspondence is applied to the next layer's input
   columns before that layer is matched.
3. **Match rectangularly.** Different channel counts are allowed.
   Candidate channels with no base counterpart count as zero similarity.
4. **Account for everything unscored.** Every matrix that cannot be
   scored is recorded with a reason: `declared_replaced`, `missing`,
   `added` or `incomparable`.

S_w is the median of the per-layer scores. The minimum per-layer score
and the unscored list are reported alongside it.

**Why each part exists.** Each design choice was forced by a measured
failure:

| Choice | Failure it fixes | Measured effect |
|---|---|---|
| Exclude 1D parameters (biases, BatchNorm) | BatchNorm weights start at all-ones and all-zeros in every model, so unrelated models matched them almost perfectly | Genuine vs unrelated separation went from 0.002 to about 0.5 |
| Channel-matched rather than flattened cosine | Permuting a layer's output channels is an exact, function-preserving symmetry; flattened cosine collapsed under it | A single permuted layer: 0.5 → 1.0 |
| Sequential alignment | Matching each layer alone still saw inputs scrambled by the previous layer's permutation | A fully permuted genuine model: S_w 0.60 → 1.00, admitted |
| Rectangular matching | Structured pruning changes channel counts; the old fallback mis-scored a scattered channel drop | A layer keeping 6 of 8 channels verbatim: 0.13 → 1.0 |
| Count unscored matrices | The old adapter silently skipped any layer whose shape differed, so reshaping foreign layers removed them from scoring | Closed a full pipeline bypass (Section 6) |
| Unmatched candidate channels score zero | Without this, extra foreign channels bolted onto a copied layer would be invisible | Widened layer (8 base + 8 foreign rows) scores 0.5, not 1.0 |
| Cosine only, no spectral term | Same-architecture, same-data models have near-identical singular-value spectra, which compressed the margin | Worst genuine vs best unrelated per-layer minimum: 0.848 / 0.148 cosine-only, 0.917 / 0.573 blended, 0.986 / 0.986 spectra-only |

**Limits.**

- **Network structure.** Chaining follows parameter-registration order,
  which is data-flow order only for sequential networks. Residual and
  transformer blocks need a model-specific alignment map (the BERT
  script includes one for FFN blocks). Attention-head permutation is not
  handled.
- **Alignment inflation.** Alignment makes unrelated models'
  *classifier* layers look more alike (null maximum 0.30 → 0.61, the
  Git Re-Basin effect). That inflation alone sets the floor threshold,
  0.49. The margin to the weakest genuine derivative (4-level
  quantization, 0.848) is about 0.36 here and must be re-measured on
  real models.
- **Where the setting was chosen.** The cosine-only setting was chosen
  on this synthetic corpus, so the real-model run reports both settings
  (`--spectral-weight`).

## 4.3 Behavioural delta D_b

**What it does.** It runs the candidate and the base on the same probe
inputs. It computes linear CKA between their activations at each probed
layer (keeping the minimum) and the Jensen–Shannon divergence between
their output distributions. The **envelope score** is max(1 − min CKA,
JS); higher means more anomalous.

**Why.** A backdoor must create a new feature-to-target mapping. The
hypothesis is that this lowers late-layer CKA and raises output
divergence more than benign fine-tuning does. Linear CKA is invariant to
orthogonal transforms and isotropic scaling, so benign re-scaling and
quantization noise move it little. JS divergence is bounded and
symmetric, which keeps calibration well behaved. JS replaces the
prototype's unanchored KL divergence.

**Evidence.**

- Envelope AUROC is 0.997 against 30 backdoored derivatives (patch,
  blended and warped triggers) the test was never calibrated on; 29 are
  escalated and one is admitted.
- The envelope score is exactly invariant to channel permutation: a
  backdoored model's score changed by less than 4 × 10⁻⁹ when fully
  permuted. Permuting weights therefore cannot be used to evade it.
- A sweep of non-adaptive backdoor training intensity found no
  "stealthy but effective" region. Attack success and envelope anomaly
  rose together.

**Limit.** An attacker who trains directly against the envelope score
(differentiable re-implementation in `adaptive_attack.py`) cuts
detection from 100% to 20% at penalty weight λ ≈ 1, with attack success
staying at or above 99.8%. The envelope test is **triage, not a
guarantee**. It also escalates legitimate head-replacement fine-tunes,
because a new head changes outputs by design: output JS was 0.32 while
CKA was 0.98.

## 4.4 Conformal thresholds

**What it does.** The threshold is the ⌈(n+1)(1−α)⌉-th smallest
calibration score. For a candidate exchangeable with the calibration
set, the probability of exceeding it is at most α. Each test is
calibrated on its own population:

- the **lineage test** on unrelated models (null: "candidate is
  unrelated");
- the **envelope test** on benign derivatives (null: "candidate is
  benign").

`min_calibration_size(α)` reports the n needed for a meaningful bound.
Verdicts carry a warning below it.

**Why.** It replaces the prototype's hand-picked τ = 0.02 or 0.05, which
could not separate a backdoored range starting at 0.001. With n = 19 and
α = 0.05 the threshold equals the maximum calibration score, the earlier
review's "calibrate on the max clean score" advice, now with a stated
error bound.

**Hardening.** A single NaN calibration score used to become the
threshold: NumPy's sort places NaN last, and every comparison with NaN
is false. That silently disabled the envelope test (fail-open).
Non-finite calibration or candidate scores now raise an error.

**Limit.** The ≤ α guarantee covers only candidates exchangeable with
the self-made calibration derivatives. Real third-party fine-tunes are
not, so the real-world false-block rate has to be measured separately.
On the pilot, benign fine-tunes 5× longer than the calibration recipe
were escalated 7/7 (`13_developer_checklist_answers.md`, Q10).

## 4.5 The lineage rule

**What it does.** `pipeline.lineage_verdict_for` verifies lineage only
if all three conditions hold:

1. median S_w exceeds its conformal threshold;
2. **every** scored layer exceeds a per-layer floor, calibrated on
   unrelated models' pooled per-layer scores;
3. at most `max_exempt` (default 1) matrices are unscored.

The pilot and the gate call the same function, so the pilot measures
exactly what the gate enforces.

**Why each condition.**

- **Median only** is gameable: copy just over half the layers verbatim,
  replace the rest.
- **The floor** catches that, but only for layers that get scored.
- **The cap** stops "unscored" from becoming the new hiding place.

Section 6 has the attacks and numbers.

**Limit.** The floor has no formal error guarantee. Per-layer scores
from one model are correlated, so pooling them breaks exchangeability.
Because the floor only ever adds rejections, false accepts stay bounded
by the median test, but its false-reject rate is measured, not
guaranteed.

## 4.6 Tier routing

**What it does.**

- **Tier 0:** a trusted publisher's signed, unmodified model is admitted
  on signature alone.
- **Tier 1:** a derivative of a verified base takes both tests.
  - Pass both: **ADMIT**.
  - Envelope fails, lineage holds: **ESCALATE** to Tier 2.
  - Lineage fails: **BLOCK** directly.
- **Tier 2:** no verifiable base, or escalated from Tier 1. The model
  goes to deep scan (pickle and code scanners, then MM-BD or BAIT-class
  backdoor detectors), with human review on a positive result.

**Why lineage failure blocks instead of escalating.** Deep scan looks
for malicious payloads, not ancestry. A model with a verified base and a
*measured* lineage failure has been positively shown not to descend from
what it claims. Routing it to deep scan would let forged provenance be
admitted on a clean scan.

**Why a cap failure gets its own reason.** `ConformalVerdict.failure_reason`
lets the block message say "N weight matrices could not be scored"
instead of misreporting a passing S_w as a failed threshold.

## 4.7 Attestation

**What it does.** It builds an in-toto Statement v1. The subject is the
candidate's weight digest. The predicate type is
`https://securemodelgate.hpe.com/lineage/v0.1`. It is signed as a DSSE
envelope. The predicate records:

- the declared base (purl, bundle digest, signer);
- the ML-BOM digest;
- lineage: S_w, threshold, per-layer minimum, floor threshold, unscored
  list, cap, α_L, and digests of the lineage and floor calibration sets;
- the envelope: CKA, JS, threshold, α, n and its calibration digest;
- the probe pool digest, seed commitment and seed;
- tier, deep-scan result, verdict, gate version and timestamp.

**Why.** It replaces the prototype's RSA-4096 JWT "MAT" and bespoke JSON
MBOM with standard, independently verifiable formats. It also makes
every threshold traceable to the exact data that produced it.

**Fixed while building verification.** `gate_version` was accepted but
never written, so every predicate said 0.1.0. The envelope and floor
calibration sets were not digested at all.

**Limits.**

- `LocalHmacSigner` is a stand-in for keyless Sigstore or KMS signing.
  The envelope shape is final; the cryptography is not.
- The probe-seed commitment is a **reproducibility record, not a
  defence**. The seed and its hash are published together, and a small
  integer seed can be recovered by enumeration.
- The predicate domain needs internal approval.

## 4.8 Verification at admission

**What it does.** `verify_attestation` checks the signature. It then
recomputes, from the artifacts actually presented:

- the candidate's weight digest;
- the ML-BOM digest;
- all three calibration-set digests;
- the probe-pool digest;
- the seed commitment.

It also checks the gate version. It fails closed on any mismatch, and on
a Tier 1 attestation presented without the inputs needed to check it.

**Why.** A valid signature only proves the gate said something. Without
recomputation, an attestation could be replayed against a different
model, a swapped ML-BOM, or thresholds calibrated on different data.

**Evidence.** A tamper test changes each input in turn: a different
candidate, an edited ML-BOM, one calibration score dropped, a 10⁻⁹
perturbation, an extra score, a different probe pool, a wrong gate
version, and a forged signature. Every one is rejected
(`tests/test_verify_and_webhook.py`).

## 4.9 Kubernetes admission mapping

**What it does.** `admission_response` returns an AdmissionReview v1
response. Only **ADMIT** produces `allowed: true`. ESCALATE, BLOCK and
any unrecognised verdict deny.

**Why.** A validating webhook call is synchronous, with a timeout of a
few seconds, and deep scan cannot finish inside it. At the moment of the
call no Tier 2 candidate has been scanned yet. An integration that read
"not BLOCK" as "allow" would therefore let every Tier 2 candidate
through. Deep scan runs out of band, and re-submission carries the
resolved verdict.

## 4.10 Supporting modules

| Module | Purpose |
|---|---|
| `oms.py` | Builds structurally valid stand-in OMS bundles and CycloneDX ML-BOMs so the pipeline runs end to end without a Sigstore deployment |
| `digest.py` | SHA-256 over model weights (sorted by name), canonical JSON and probe-pool tensors; the basis of every binding in the attestation |
| `demo_models.py` | The synthetic corpus: `TinyConvNet`, base, benign fine-tune, magnitude-pruned, quantized, backdoored (patch / blended / warped triggers) and independent models |
| `stats.py` | Clopper–Pearson intervals, AUROC, TPR at a target FPR |
| `figures.py` | ROC curve, score distributions, per-trigger and adaptive-attack plots |
| `cli.py` | `demo`, `verify`, `evaluate`, `adaptive`, `permutation-check`, `lineage-scope-check` |

# 5. Evaluation harnesses and current results

Each harness is a reproducible CLI command whose report is committed
under `paper/techcon2027_revision/`.

## 5.1 Pilot A: end-to-end statistical protocol (`securemodelgate evaluate`)

The corpus has:

- 40 unrelated models, split 20 for calibration and 20 held out;
- 40 benign derivatives (fine-tunes, and pruned or quantized
  fine-tunes, each from its own seed, all distinct), split 20 for
  calibration and 20 for testing;
- 30 backdoored derivatives (10 for each trigger type).

| Metric | Value (current code) |
|---|---|
| Backdoor TPR at the conformal threshold (α = 0.05) | 29/30 = 96.7% (95% CI 82.8–99.9%); the miss is **admitted** |
| Envelope AUROC | 0.997 |
| Benign false-block rate | 0/20 (95% CI 0–16.8%) |
| Lineage accuracy (full rule) | 69/70 = 98.6% (95% CI 92.3–100%) |
| Median scoring latency (S_w + D_b) | ≈ 120 ms per model, CPU |

Every model's record is in `pilot_results/evaluation.json`. The single
lineage error is `backdoor-warped-3009`, a genuine, backdoored,
warped-trigger derivative: its classifier layer scored 0.484 against a
0.508 floor, so it was blocked. No unrelated model was admitted. The
missed backdoor is `backdoor-patch-3007` (ASR 100%, envelope 0.1022
against 0.1108).

**Correction.** An earlier run reported 30/30 and a 2/20 = 10%
false-block rate. Its benign test set was leaky: pruned and quantized
members were copies of the base itself, so 13 of the 20 test models were
byte-identical to calibration models. The population is now fixed and
the harness refuses any overlap.

## 5.2 Pilot B: adaptive attacker (`securemodelgate adaptive`)

The attacker trains poisoned cross-entropy plus λ × (differentiable
envelope score). The envelope threshold is 0.1087 (n = 20, α = 0.05).

| λ | 0 | 0.5 | 1.0 | 2.0 | 5.0 |
|---|---|---|---|---|---|
| Detected | 10/10 | 8/10 | 2/10 | 3/10 | 6/10 |
| Attack success retained | 99.8% | 100% | 100% | 100% | 100% |

The rebound at higher λ is most likely optimisation noise at n = 10. The
robust finding is the collapse between λ = 0 and λ ≈ 1.

## 5.3 Pilot C: permutation robustness (`securemodelgate permutation-check`)

| Scenario | S_w | Max output difference vs unpermuted |
|---|---|---|
| Unpermuted descendant | 1.0000 | — |
| One layer permuted | 1.0000 | 4.8 × 10⁻⁷ |
| Every layer permuted (cascading) | 1.0000 | 1.4 × 10⁻⁶ |

Unrelated models score 0.159–0.203. Both permutations are verified to
compute the same function.

## 5.4 Pilot D: lineage scope check (`securemodelgate lineage-scope-check`)

**Splice depth.** A genuine fine-tune has its last k weight matrices
replaced by an unrelated model's, across 5 donors. The table counts
splices **admitted**.

| k foreign matrices | Median-only rule | Full rule | Full rule, replacements declared |
|---|---|---|---|
| 1 | 5/5 | 0/5 | 5/5 (by design: legitimate head replacement) |
| 2 | 5/5 | 0/5 | 0/5 (cap) |
| 3 | 5/5 | 0/5 | 0/5 (cap) |
| 4 | 4/5 | 0/5 | 0/5 (cap) |

**Spectral ablation.** Scored against 13 genuine derivatives and 20
held-out unrelated models.

| Per-layer score | Worst genuine min-layer | Best unrelated min-layer | Genuine admitted | Unrelated admitted | Splices admitted |
|---|---|---|---|---|---|
| Cosine only (default) | 0.848 | 0.148 | 13/13 | 0/20 | 0/5 |
| 50/50 blend (old) | 0.917 | 0.573 | 13/13 | 0/20 | 0/5 |
| Spectra only | 0.986 | 0.986 | 9/13 | 0/20 | 0/5 |

## 5.5 Real-model evidence (`experiments/smg_bert_lineage.py`)

The script tests public weights:

- **Base:** `bert-base-uncased`.
- **Descendants:** seven public fine-tunes, plus INT8, 30%-pruned and
  FFN-permuted copies.
- **Negatives:** 25 MultiBERTs seeds, which share BERT-base's
  architecture and training data but not its random seed, the hardest
  honest negative available. Twenty calibrate the thresholds and five
  are held out.
- **Attack:** a layer splice.

Two fixes were made after auditing it:

1. **It refuses infinite thresholds.** Its originally recommended quick
   run (`--n-seeds 12 --n-cal 9` with the default α) made both
   thresholds infinite. Every genuine model was rejected while the
   summary still looked healthy.
2. **It aligns every model,** calibration included, so the thresholds
   account for alignment inflation.

`--selftest` passes offline. The author ran it on a Mac (about 12 GB
downloaded), cosine-only and with a 0.5 spectral blend
([`pilot_results_bert/`](pilot_results_bert/README.md), transcribed from
the console logs):

| Check | Cosine-only (default) | 0.5 spectral blend |
|---|---|---|
| T_median / T_floor | 0.1115 / 0.1282 | 0.5549 / 0.5636 |
| Descendants accepted | 10/10 | 10/10 |
| Held-out MultiBERTs rejected | 5/5 | 5/5 |
| Splice: median-only / full rule | admitted / blocked | admitted / blocked |
| Weakest genuine matrix | 0.930 (Yelp) | 0.965 |
| Margin to the floor | ≈ 0.80 | ≈ 0.40 |

Without alignment, the FFN-permuted fine-tune is rejected (worst matrix
0.055); with alignment it scores 1.000. The blend halves the margin and
puts unrelated medians on the median threshold, which confirms the
cosine-only default on real weights.

Limits:
- lineage only, on honest models;
- 7 public fine-tunes, 6 of them from one publisher;
- 5 held-out negatives (95% CI for 5/5: 48–100%);
- no adaptive forger and no same-initialisation null.

# 6. Attacks tested, and where each stands

| Attack | Before the fix | After | What stops it |
|---|---|---|---|
| Lineage forgery: independent model claims a signed base | Blocked (S_w 0.17 vs threshold 0.22) | Blocked | Median and floor |
| Permuted independent model claims a base | — | Blocked (S_w 0.173, per-layer minimum 0.139) | Alignment does not manufacture ancestry |
| Splice: verbatim early layers plus foreign tail | **Admitted** by median-only (S_w 0.76–0.81) | Blocked at every depth | Per-layer floor |
| Splice plus adaptive envelope training | **Tier 1 ADMIT, 100% attack success** (S_w 0.78, envelope 0.065 vs 0.122) | Blocked | Per-layer floor |
| Splice plus reshaped foreign layers plus adaptive training | **Tier 1 ADMIT, 100% attack success** on 2 of 3 seeds; only 3 of 6 matrices scored | Blocked on 3/3 (6/6 scored, per-layer minimum 0.150–0.159 vs floor 0.490) | Every matrix scored; unscored capped |
| Declare 3 foreign layers "replaced" to dodge scoring | — | Blocked | Exemption cap (max 1) |
| Add extra foreign modules beside copied ones | Silently ignored | Counted as unscored; 2 added → blocked | Exemption cap |
| Widen a copied layer with extra foreign channels | Ignored by matching | Unmatched channels score 0 | Rectangular scoring rule |
| Hand the gate the wrong base model | **Admitted** against a decoy | Tier 2 | Digest binding |
| Corrupt one calibration score with NaN | **Envelope test silently disabled** | Raises an error | Finite-value guard |
| Replay or alter an attestation | Signature-only check | Rejected | `verify_attestation` |
| Non-adaptive backdoor (any training intensity tried) | Escalated | Escalated | Envelope test |
| Adaptive backdoor in a genuine fine-tune (A2 + A4) | Detected 20% at λ ≈ 1 | **Not stopped** (4/4 admitted at λ ≥ 1) | — (see Section 8) |
| S_w-adaptive forgery: unrelated model pulled toward the base | — | **Passes lineage** 4/4 (μ ≥ 3); escalated by the envelope | Envelope only |
| Joint attack: S_w pull plus envelope penalty | — | **Not stopped** (4/4 admitted, ASR 100%) | — |
| Every one of 63 whole-matrix splice patterns, undeclared | — | 0/285 multi-matrix splices pass | Per-layer floor |
| Edit BatchNorm statistics after admission | **Old attestation still verified** | Rejected | Full state-dict digest |
| Head weights of ±1e37 so logits overflow | **Admitted** (NaN JS ignored) | Denied | Non-finite envelope components raise |
| Re-sign a predicate claiming ADMIT | Verified | Rejected | Verifier re-derives thresholds, verdict, scores |
| Deep-scan result for different bytes | Accepted (unbound boolean) | Ignored, stays ESCALATE | `deep_scan_report` subject digest |
| Allowed signer name over a digest it never signed | Tier 1 | Tier 2 with the trusted-base registry | `trusted_bases` |
| Malicious pickle model file | Code runs under plain `torch.load` | Refused, not executed | `loader.py` |

# 7. Bugs and false claims found and fixed

Each item was reproduced with runnable code before being called a bug,
and verified against the same reproduction after the fix.

| # | Problem | Consequence if shipped | Fix |
|---|---|---|---|
| 1 | 1D BatchNorm parameters included in S_w | Genuine and unrelated models indistinguishable (gap 0.002) | Score weight matrices only |
| 2 | Flattened cosine not permutation-invariant | Benign permuted derivatives rejected | Channel-matched cosine |
| 3 | Base model not bound to declared digest | Signed "verified" lineage against the wrong model | Digest check, fail closed |
| 4 | Median-only lineage | Full pipeline bypass (splice) | Per-layer floor |
| 5 | NaN calibration score | Envelope test silently fail-open | Reject non-finite scores |
| 6 | Shape-mismatch fallback in the numpy scorer | Scattered channel pruning scored 0.13 instead of 1.0 | Rectangular assignment |
| 7 | Shape-mismatched layers silently skipped in the pipeline | **Full pipeline bypass re-opened** (reshape route); claims about items 4 and 6 overstated | Sequential alignment, unscored accounting, cap |
| 8 | Cascading permutation only partly recovered | First admitted silently, then falsely blocked once the floor landed | Sequential alignment |
| 9 | Spectral term in S_w | Margin halved; 5 genuine derivatives falsely rejected | Cosine-only default |
| 10 | `gate_version` ignored; two calibration sets unbound | Unverifiable thresholds; wrong version in every record | Thread through, digest all |
| 11 | BERT script's recommended quick run | All-reject result that looks like success | Refuse infinite thresholds |
| 12 | Reference [26] wrong title; [23]–[26] incomplete | Desk-reject risk under Tech Con's reference rule | Corrected (confirm bylines) |
| 13 | Benign population of deterministic copies of the base | Leaky test set; the 10% false-block figure and 30/30 were measured on it | Distinct seeded derivatives; overlap refused |
| 14 | `max(1 − CKA, NaN)` returns the CKA term | Logit-overflow model admitted (fail-open) | Non-finite components raise |
| 15 | Subject digest over parameters only | BatchNorm edits after admission undetected | Digest the full state dict |
| 16 | DSSE stand-in signed the raw payload | `payloadType` unauthenticated | Sign PAE |
| 17 | Verifier checked digests only | A re-signed false predicate verified | Re-derive thresholds and verdict; recompute scores |
| 18 | Unbounded candidate width; vocabulary-size matching | Memory denial of service on the gate | 2× row cap; index alignment above 8,192 rows |
| 19 | Seed-commitment test named "does not leak seed" | Overclaim: the hash is brute-forced in seconds | Renamed; documented as a record only |

Item 7 is the most important lesson. The fix for item 4 was declared
complete while an attacker could still choose which layers got scored.
The correction is recorded in the pilot documents rather than rewritten.

# 8. What SecureModelGate does not do

These limits are stated in the paper as well. A reviewer should not
have to discover them.

1. **A backdoored *genuine* fine-tune trained against the envelope is
   not stopped.** Its lineage is true, and 4/4 are admitted at λ ≥ 1.
   A backdoor in a declared, replaced head is escalated only because
   every head replacement is. Against that attacker the gate contributes
   a signed, attributable record of who published what claiming which
   base, not detection.
2. **Lineage is not forgery-proof.** An unrelated model trained toward
   the public base passes the lineage test, and with an envelope penalty
   as well it is admitted (4/4). Models sharing the base's random
   initialisation score S_w ≈ 0.70. About a quarter of every layer's
   channels can be foreign.
3. **The error guarantees are narrower than they look.** The conformal
   bound covers only derivatives like the self-made calibration ones
   (deeper fine-tunes are escalated 7/7). The per-layer floor has no
   formal bound. One non-adaptive backdoor in 30 is admitted.
4. **Legitimate head replacement goes to deep scan.** This is a
   deliberate availability cost, chosen to avoid giving attackers a
   switch that disables the output check.
5. **Alignment is exact only for sequential networks.** Attention-head
   permutation is not handled.
6. **Signing and signature verification are stand-ins.** HMAC replaces
   Sigstore, and the OMS check is structural only.
7. **Real-model evidence is lineage-only and small.** The BERT run
   covers 10 honest descendants and 5 held-out unrelated models. Every
   envelope, backdoor and adaptive-attack number is synthetic, and
   E1–E9 are pending.
8. **No real Kubernetes deployment has been tested,** and nothing binds
   what a container loads after admission to the attested digest.
9. **Out of scope:** compromised publisher keys, backdoors already in
   the signed base, distillation (not a weight descendant, so it goes to
   deep scan), and runtime monitoring.

# 9. Test suite

129 tests pass.

| File | Tests | What it protects |
|---|---|---|
| `test_lineage.py` | 33 | Scoring maths, alignment, rectangular and widened matching, CKA/JS, conformal quantiles, NaN guards, routing, verdict constants, signing |
| `test_pipeline.py` | 20 | End-to-end admission: tiers, forgery, poisoning, digest binding, splice, reshape, added modules, declared head and cap, permutation, attestation contents |
| `test_verify_and_webhook.py` | 14 (5 functions, parametrised) | Tamper detection on every bound input; only ADMIT is allowed |
| `test_stats.py` | 8 | Clopper–Pearson, AUROC, TPR at FPR |
| `test_evaluation.py` | 6 | Pilot A harness |
| `test_adaptive_attack.py` | 6 | Differentiable CKA/JS match the scoring code; adaptive training |
| `test_permutation_check.py` | 3 | Permutations are function-preserving and fully recovered |
| `test_cli.py` | 3 | CLI subcommands run and write reports |
| `test_hardening.py` | 25 | Re-derivation, freshness, revocation, policy, deep-scan binding, signer substitution, scorer guards, safe loader |
| `test_end_to_end_attacks.py` | 11 | Each attack the gate should stop, through the deployment entry points; one pinned known gap |

# 10. Reproducing everything

```bash
pip install -e ".[dev]"
pytest tests/                                   # 129 tests
securemodelgate demo                            # six admission scenarios, signed attestations
securemodelgate evaluate                        # Pilot A
securemodelgate adaptive --lambdas 0 0.5 1 2 5 --n-per-lambda 10   # Pilot B
securemodelgate permutation-check               # Pilot C
securemodelgate lineage-scope-check             # Pilot D
securemodelgate defensibility-check             # Pilot E (developer checklist)
python experiments/smg_bert_lineage.py --selftest
python experiments/smg_bert_lineage.py --out results_cosine.json                       # needs Hugging Face access
python experiments/smg_bert_lineage.py --spectral-weight 0.5 --out results_blend.json
```

# 11. Remaining work, in order of value

1. **Commit the BERT results JSON** (`results_cosine.json`,
   `results_blend.json`) into `pilot_results_bert/`. Then extend the run:
   - an adaptive forger (an unrelated MultiBERT fine-tuned toward the
     base's weights);
   - a same-initialisation null (MultiBERTs intermediate checkpoints);
   - more publishers' fine-tunes.
2. **Run E1–E9** (PreAct-ResNet-18 / CIFAR-10 / BackdoorBench with
   Neural Cleanse, MM-BD and STRIP re-run) on GPU. That fills the
   paper's Table 1.
3. **Replace the stand-ins.** Real Sigstore verification in
   `verify_oms_signature`; keyless or KMS signing in place of
   `LocalHmacSigner`.
4. **Measure the false-block rate on real third-party fine-tunes**, and
   size calibration for the claimed α (n ≥ 99 for 1%).
5. **Build alignment maps for residual and transformer architectures**,
   including attention heads.
6. **Confirm the bylines of references [23]–[26]** on their landing
   pages, and resolve [39] (Cisco AI Defense) or drop it.
