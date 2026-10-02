# SecureModelGate

Signed, calibrated **behavioral-lineage attestation** for admitting
third-party AI models. SecureModelGate answers two questions a plain
model signature can't:

1. **Lineage** — does this model really descend from the OMS-signed
   base it claims (`base_model` in its ML-BOM), or is that claim
   forged?
2. **Envelope** — has it drifted from that base further than benign
   fine-tunes, pruning or quantization normally do?

It issues the verdict as a signed in-toto predicate alongside the
model's OpenSSF Model Signing (OMS) bundle, and enforces it with
risk-tiered Kubernetes admission (trusted publisher → fast path; signed
base, unverified derivative → lineage + behavioral check; unsigned →
deep scan). See
[`paper/techcon2027_revision/02_paper.md`](paper/techcon2027_revision/02_paper.md)
for the full design and evaluation plan.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python ≥3.10. Pulls in `numpy`, `scipy`, `torch`, `scikit-learn`
and `matplotlib` (CPU is fine — nothing here needs a GPU).

## Quickstart

```bash
securemodelgate demo --out-dir ./attestations
securemodelgate verify ./attestations/benign_fine_tune.attestation.json
```

`demo` builds a small base model and six candidates end to end (real
torch models, real forward hooks, real signing — see
[`securemodelgate/demo_models.py`](securemodelgate/demo_models.py)) and
runs each through the full admission pipeline:

```
scenario               tier  verdict    reason
----------------------------------------------------------------------------------------------------
trusted_publisher      0     ADMIT      trusted publisher signature verified
unsigned_import        2     ADMIT      deep scan found nothing; admitted with attestation
benign_fine_tune       1     ADMIT      lineage verified (S_w=1.0000) and within the benign-derivative envelope (score=0.0591 <= 0.0988)
pruned_derivative      1     ADMIT      lineage verified (S_w=0.9984) and within the benign-derivative envelope (score=0.0027 <= 0.0988)
poisoned_derivative    2     BLOCK      escalated candidate: deep scan flagged the model
lineage_forged         1     BLOCK      S_w=0.5681 did not exceed the independent-model quantile (0.5882 at alpha_L=0.05); candidate does not verify as a descendant of the declared base
```

The poisoned derivative (a genuine fine-tune of the base carrying a
backdoor with ~99.5% attack success rate) is caught by the **behavioral
envelope test**, not the lineage test — it really is a descendant, it
just behaves anomalously. The `lineage_forged` model (independently
trained, falsely claiming the base as its ancestor) is caught by the
**lineage test** instead. That's the two-question split the design is
built around.

Each scenario writes a signed DSSE-wrapped in-toto attestation JSON;
`verify` checks the signature and prints the predicate.

### Statistically evaluate it

```bash
securemodelgate evaluate --n-independent 40 --n-benign 40 --n-backdoored-per-trigger 10
```

Runs the same statistical protocol the paper's Table 1 uses —
split-conformal calibration, Clopper-Pearson 95% CIs, TPR at a target
FPR, per-trigger-type breakdown, lineage accuracy against held-out
negatives — end to end against the synthetic corpus, and writes
`report.json` / `report.md` plus an ROC curve, score-distribution and
per-trigger-breakdown figure. A real run (n=40/40/30, ~30s on 4 CPU
cores):

| Metric | Value |
|---|---|
| False-block rate (benign derivatives) | 0/20 = 0.0% (95% CI 0.0%–16.8%) |
| Backdoor TPR @ conformal threshold | 29/30 = 96.7% (95% CI 82.8%–99.9%); the one miss (patch, ASR 100%) is admitted |
| Envelope-score AUROC | 0.9967 |
| Lineage accuracy | 69/70 = 98.6% (95% CI 92.3%–100.0%) |
| Median pipeline latency (scoring only) | ~120 ms |

It also writes `evaluation.json`: one record per model (all 110), with
every per-layer score, threshold and decision. An earlier version of this
table reported 2/20 false blocks on a test set that was mostly
byte-identical to the calibration set; see the correction in the pilot
write-up.

**This is an engineering-validation pilot, not the paper's evaluation.**
It proves the conformal/lineage/envelope machinery works correctly end
to end on real (if small and synthetic) models — it is not a substitute
for Experiments E1–E9
([`paper/techcon2027_revision/04_experiments_plan.md`](paper/techcon2027_revision/04_experiments_plan.md)),
which run the paper's actual PreAct-ResNet-18/CIFAR-10/BackdoorBench
protocol. See
[`paper/techcon2027_revision/06_engineering_validation_pilot.md`](paper/techcon2027_revision/06_engineering_validation_pilot.md)
for the full pilot write-up, including what it does and doesn't
demonstrate.

### Attack it back: the adaptive adversary (A4)

```bash
securemodelgate adaptive --lambdas 0.0 0.5 1.0 2.0 5.0 --n-per-lambda 10
```

An attacker who knows the exact envelope score and trains directly
against it (`securemodelgate/adaptive_attack.py`: the score
reimplemented in torch, numerically checked against the real scoring
code to 1e-8, added as a penalty term during poisoning). A real run:

| λ (penalty weight) | Detection rate | Mean ASR retained |
|---|---|---|
| 0.0 (non-adaptive) | 100.0% | 99.8% |
| 0.5 | 80.0% | 100.0% |
| 1.0 | 20.0% | 100.0% |
| 5.0 | 60.0% | 100.0% |

Detection collapses well before the backdoor loses any effectiveness.
This is reported as a limitation, not hidden — see
[`paper/techcon2027_revision/07_adaptive_attack_pilot.md`](paper/techcon2027_revision/07_adaptive_attack_pilot.md)
for the honest read of what this does and doesn't prove.

### Does S_w actually survive the permutation it claims to resist?

```bash
securemodelgate permutation-check
```

Section 3.2 claims S_w "resists simple permutation ... obfuscation."
This command tests that directly against an exact, function-preserving
channel permutation (verified with a forward-pass diff, not assumed) —
and found the claim was wrong as originally stated. A real run:

| Scenario | S_w | Functional diff |
|---|---|---|
| Unpermuted descendant | 1.0000 | — |
| Single layer permuted | 1.0000 | 4.77e-07 |
| Every layer permuted (cascading) | 1.0000 | 1.43e-06 |
| Independent (unrelated) models, for context | 0.16–0.20 | — |

Both are fully recovered on this sequential architecture: output
channels are matched by optimal assignment, and each layer's matching is
carried into the next layer's input columns. Residual, grouped and
attention layers would need a model-specific alignment map — see
[`paper/techcon2027_revision/09_permutation_robustness_pilot.md`](paper/techcon2027_revision/09_permutation_robustness_pilot.md).

## Architecture

```
securemodelgate/
  lineage/
    reference_resolver.py   — resolve + verify the declared OMS-signed base (§3.1)
    lineage_score.py        — S_w: channel-matched weight agreement, sequentially aligned (§3.2)
    behavioral_delta.py     — D_b: min-layer linear CKA + output JS divergence (§3.2)
    conformal.py            — split-conformal thresholds, replacing a hand-picked τ (§3.3)
    intoto_attestation.py   — signed in-toto predicate, replacing a JWT MAT (§3.4)
    admission_policy.py     — Tier 0 / 1 / 2 routing (§3.5)
  digest.py                 — SHA-256 digests over weights / JSON / probe pools
  oms.py                    — demo OMS bundle + ML-BOM construction
  demo_models.py            — small, fast-to-train torch models + 3 trigger types for the CLI demo, evaluation & tests
  pipeline.py                — wires the above into one `run_admission(...)` call
  stats.py                   — Clopper-Pearson CIs, AUROC, TPR@FPR
  evaluation.py               — the statistical evaluation harness (`run_evaluation`)
  adaptive_attack.py           — differentiable envelope score for the A4 adaptive adversary
  adaptive_evaluation.py       — the adaptive-attack experiment harness
  permutation_check.py          — does S_w survive an exact, function-preserving channel permutation?
  lineage_scope_check.py        — splice-depth sweep and spectral ablation
  defensibility_checks.py       — the developer-checklist measurements (splices, S_w-adaptive and joint attacks, sweeps, determinism, scaling)
  verify.py                     — admission-time attestation verification (digests, thresholds, verdict, recomputed scores)
  webhook.py                    — AdmissionReview mapping; fail-closed gate and attestation review paths
  loader.py                     — untrusted model-file loading (safetensors / weights_only, no code execution)
  figures.py                    — ROC / score-distribution / per-trigger-breakdown / adaptive-attack plots
  cli.py                         — `securemodelgate demo` / `evaluate` / `adaptive` / `permutation-check` / `lineage-scope-check` / `defensibility-check` / `verify`
```

Section numbers refer to
[`paper/techcon2027_revision/02_paper.md`](paper/techcon2027_revision/02_paper.md).
Each `lineage/` module's docstring explains the reasoning (why spectral
correlation, why linear CKA, why median-over-layers, etc.) and is a
better starting point than this README for the *why*. (Spectral correlation is still implemented but weighted 0 by default: it was measured to hurt separation.)

**What's real vs. a stand-in.** The scoring math (`lineage_score.py`,
`behavioral_delta.py`, `conformal.py`), the routing logic
(`admission_policy.py`), and the DSSE envelope shape
(`intoto_attestation.py`) are the real design. `LocalHmacSigner` and
`oms.py`'s bundle-building are structural stand-ins for real
Sigstore/KMS signing and real OMS verification (see their docstrings) —
swapping those in is the only change needed to move from this demo to
production signing.

## Testing

```bash
pytest tests/ -v
```

- `tests/test_lineage.py` — pure-math unit tests (spectral/directional
  agreement, linear CKA, JS divergence, conformal quantiles, attestation
  sign/verify + tamper detection, admission routing). No torch
  dependency, runs in well under a second.
- `tests/test_stats.py` — Clopper-Pearson CIs checked against the
  paper's own cited intervals (3/20 → 3–38%, 20/20 → 83–100%), plus
  AUROC/TPR@FPR sanity checks.
- `tests/test_pipeline.py` — end-to-end tests of `pipeline.run_admission`
  against real torch models: a trusted-publisher fast path, an unsigned
  import routed to deep scan, a benign fine-tune and pruned derivative
  admitted at Tier 1, a lineage-forged model blocked by the lineage
  test, and a poisoned derivative escalated by the envelope test.
- `tests/test_evaluation.py` — the evaluation harness at a small
  config: well-formed metrics, all three trigger types covered,
  backdoors score higher than benign derivatives, the report round-trips
  through JSON, figures actually get written.
- `tests/test_adaptive_attack.py` — the torch envelope-score
  reimplementation matches the numpy scoring code to 1e-8, gradients
  flow into the candidate only (never the frozen base), a higher penalty
  measurably lowers the envelope score while attack success stays high,
  the adaptive-evaluation report is well-formed.
- `tests/test_permutation_check.py` and permutation-specific tests in
  `test_lineage.py`/`test_pipeline.py` — every permutation is verified
  function-preserving (forward-pass diff) before checking S_w at all;
  single-layer and full cascading permutation both recover exactly.
- `tests/test_verify_and_webhook.py`, `tests/test_hardening.py` —
  attestation tampering, re-signed inconsistent predicates, freshness,
  revocation, policy changes, deep-scan binding, signer substitution,
  scorer input guards, and the safe loader (including a live malicious
  pickle that must not execute).
- `tests/test_end_to_end_attacks.py` — each attack the gate should stop,
  run through the deployment entry points, after first showing the
  attack works; plus one test that pins a known gap (an adaptive poisoned
  fine-tune is admitted).
- `tests/test_cli.py` — subprocess smoke tests of the `demo`/`verify`
  commands, including signature-tamper rejection.

CI (`.github/workflows/ci.yml`) runs the full suite against Python
3.10–3.12 on every push.

## Repository layout

| Path | What it is |
|---|---|
| `securemodelgate/` | This package — the revised lineage-attestation methodology, installable and tested. |
| `tests/` | Its test suite. |
| `paper/techcon2027_revision/` | The revised Tech Con 2027 abstract, paper, change log, and the E1–E9 experiment plan that fills in the paper's real-number placeholders. |
| `paper/` (other files), `paper_and_readme.txt` | The original Parasparam 2026 submission this revision responds to. |
| `from_Downloads_alt_code/` | The Parasparam 2026 evaluation pipeline (`static.py`/`fingerprint.py`/`attestation.py`/`evaluator.py`) that produced the retained 20/20-detection, 3/20-false-block numbers cited in the revised paper. Left untouched. |
| `figures/`, `run_all.py`, `paper_metrics.txt` | Earlier prototype artifacts from the same evaluation. |

## Status

This package implements, tests, statistically evaluates, and
adversarially pressure-tests the *design* end to end on small synthetic
models — including a real conformal-calibration/AUROC/lineage-accuracy
pilot
([`paper/techcon2027_revision/06_engineering_validation_pilot.md`](paper/techcon2027_revision/06_engineering_validation_pilot.md))
and a real adaptive-attacker pilot that finds and reports the design's
actual breaking point
([`paper/techcon2027_revision/07_adaptive_attack_pilot.md`](paper/techcon2027_revision/07_adaptive_attack_pilot.md)),
and a real permutation-robustness pilot that found and fixed an
overclaimed invariance property rather than just asserting it
([`paper/techcon2027_revision/09_permutation_robustness_pilot.md`](paper/techcon2027_revision/09_permutation_robustness_pilot.md)).
It is not the paper's 4-day, 90-model BackdoorBench evaluation
(Experiments E1–E9,
[`paper/techcon2027_revision/04_experiments_plan.md`](paper/techcon2027_revision/04_experiments_plan.md))
— that needs real GPU time, real CIFAR-10 images, and PreAct-ResNet-18,
and is the next step toward filling in the paper's evaluation table.
That step is currently blocked: as of 2026-09-29 this working
environment has no GPU and its network policy denies `huggingface.co`,
`download.pytorch.org` and the CIFAR-10 mirror, confirmed still denied
after requesting a network-access change. See
[`paper/techcon2027_revision/08_presubmission_checklist.md`](paper/techcon2027_revision/08_presubmission_checklist.md)
§5 for the current status and what actually unblocks it.
