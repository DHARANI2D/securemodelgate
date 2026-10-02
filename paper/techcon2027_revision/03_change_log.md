# Change Log: Original → Revision, and Why

- **Claim.** Before: "static + behavioral backdoor detection, 117×
  faster". After: a signed, calibrated behavioral-lineage attestation.
  Why: this answers the novelty criticism (Innovation 12.5/20), and
  signature-only admission already exists.
- **Reference.** An undefined clean F_ref is replaced by the
  OMS-verified base from ML-BOM `pedigree.ancestors`. This fixes the
  reference problem for arbitrary imports.
- **Fingerprint.** The ad-hoc 128-dim [μ, σ, H] vector is replaced by
  S_w (spectral + directional weight agreement) and D_b (min-layer
  linear CKA + JS on a randomized probe set), with invariance-based
  justification. This answers the "fingerprint construction / theory"
  criticism.
- **Threshold.** τ = 0.05/0.02 is replaced by split-conformal thresholds
  with an explicit α and n. τ = 0.02 could not separate a backdoored
  range starting at 0.001, and conformal calibration gives a stated
  false-block bound.
- **Results.** Kept: 20/20, 3/20 (now with CIs), the stage timings and
  the KL range. Removed: AUC = 1.0 (unnamed score), the τ = 0.02
  separability claim, the "Combined" row (it equalled Static, so the
  behavioral stage added nothing), and the ~1,348/hr throughput claim
  for the combined pipeline.
- **Baselines.** Copied figures (NC 83.3%/7.2%/312 s, STRIP
  78.1%/11.5%/28 s) and "117×" are removed. NC, MM-BD and STRIP are
  re-run on identical models, with STRIP labelled input-level.
- **Adaptive attacks.** "Excluded (obscurity)" becomes an explicit A4
  adversary under Kerckhoffs' principle, with seed commitment and a
  fingerprint-regularised attack experiment.
- **Formats.** The RSA-4096 JWT "MAT" and bespoke MBOM are replaced by
  an in-toto predicate in a Sigstore/DSSE bundle, alongside the OMS
  signature, plus CycloneDX 1.6/1.7 ML-BOM (SPDX 3.0 AI equivalent).
- **Pipeline.** A single pipeline becomes three risk tiers, with the
  fast check as triage in front of MM-BD/BAIT.
- **Products.** "HPE Ezmeral Container Platform" is replaced by HPE
  Private Cloud AI, HPE AI Essentials Software, HPE Private Cloud
  (Morpheus), GreenLake Intelligence and HPE OpsRamp.
- **Regulation.** The EU AI Act Art. 13 citation is replaced by Art.
  11/Annex IV, 15(5) and 53, plus the Digital Omnibus 2 Dec 2027 date,
  the NSA CSI (Mar 2026), CISA (May 2025) and NIST AI 100-2e2025.
- **References.** The fabricated "L. Dex" is corrected to
  Newman/Meyers/Torres-Arias. BadNets, Goldblum and the others are fixed
  with DOI/arXiv, and 2025–2026 lineage, LLM-backdoor and TrojAI work is
  added.
- **Scope.** Runtime behavior is explicitly excluded (left to WITNESS),
  and the roadmap is re-dated from Sept 2026.
- **Name.** "Dhaaranidharan" is corrected to "Dharanidharan".

## Second revision pass (2026-09-29), responding to an external "v2" edit

A "v2" edit of the docx proposed several changes; each was independently
verified before being adopted, corrected, or rejected — not applied on
trust, per this project's own standard set by the "L. Dex" fabrication
above.

- **Adopted as verified.** References [35]–[38] (Model Atlas, NVIDIA
  blog, JFrog, Palo Alto/Protect AI) filled in with the primary sources
  v2 supplied, confirmed independently. Three new references added:
  [41] SIG (resolved to Barni/Kallas/Tondi ICIP 2019 via BackdoorBench's
  own bibtex, not just a secondary index), [42] Blended (Chen et al.
  arXiv:1712.05526), [43] Git Re-Basin (Ainsworth et al., ICLR 2023).
  Abstract tightened (~430 → ~366 words) and Section 4.4's infra-diagnostic
  detail condensed in the main body (full detail stays in
  `08_presubmission_checklist.md`).
- **Adopted, corrected.** v2's reference [40] claimed EO 14028 "as
  amended by EO 14306" — checked against whitehouse.gov/the Federal
  Register and found wrong: EO 14306 amends EO 14144 (itself built on
  EO 14028) and, separately, EO 13694; it does not amend EO 14028
  directly. Cited as the corrected three-order chain instead.
- **Rejected as factually wrong, with evidence.** v2 claimed the
  permutation weakness "fails safe": a permuted genuine derivative goes
  to deep scan and isn't admitted. Tested directly against
  `pipeline.run_admission()`: a fully cascading-permuted genuine
  derivative is currently ADMITTED at Tier 1 (S_w=0.6039 clears the
  0.5915 threshold) — the opposite of "fails safe". What IS true and now
  documented precisely (`09_permutation_robustness_pilot.md`):
  permutation cannot forge ancestry for an unrelated model (0.5756 vs.
  0.5743, both blocked), and has zero effect on the envelope/behavioral
  test (identical to 9 decimal places, since permutation is exactly
  function-preserving) — so it cannot be used to evade backdoor
  detection either way. The honest scope is a one-directional,
  currently-open lineage false-negative on genuine derivatives, not a
  security hole.
- **Rejected as a misreading, prose clarified instead of code changed.**
  v2 read Section 2/3.5's two failure paths (no verifiable base at all →
  Tier 2 deep scan; verified base but failed S_w → Tier 1 block) as a
  contradiction and proposed routing both through Tier 2. Declined:
  these are two different, deliberately different-routed failure modes
  — a positively-scored failed lineage claim is stronger evidence than
  an unscoreable one, and deep scan checks for malicious payloads, not
  ancestry, so rerouting a known-false lineage claim through it would
  let forged provenance reach admission on a clean scan. Section 2 (A1)
  and 3.5 were reworded instead to state the distinction explicitly so
  it can't be misread as a contradiction again.
- **Answered, not just noted.** v2 asked whether Pilot B's adaptive
  attacker trained against the per-admission probe subset or the full
  pool. Checked against `adaptive_attack.py`: it trains against a fresh
  resampled batch every step (not the literal fixed eval-time pool), but
  on this synthetic corpus (i.i.d. Gaussian noise, no natural-image
  structure) that distinction is close to moot — the pilot shows what a
  distribution-aware attacker achieves, not whether the per-admission
  seed commitment adds protection beyond that. `04_experiments_plan.md`'s
  E7 now calls for testing both attacker variants separately on real
  image data, where the distinction may actually matter.
- **Regression tests added** (`tests/test_pipeline.py`) pinning down the
  three permutation findings above precisely, so a future change to any
  of them is a deliberate, visible update rather than a silent
  regression.

## Third revision pass (2026-09-30): rigorous adversarial critique

A direct request to "fix all flaws and loopholes ... run rigorous
critical analysis and perform various scope tests" — not a v2-style
document edit this time, but an adversarial pass over the implemented
pipeline itself. Full writeup:
[`10_critical_review_pilot.md`](10_critical_review_pilot.md). Five scope
tests, each reproduced with runnable code before being called a finding:

- **Fixed — reference-digest binding gap.** `resolve_reference` verified
  only the candidate's OWN claim about its ancestor; nothing checked that
  the `base_model` object actually scored against matched that claim.
  `pipeline.run_admission` now binds the two explicitly and fails closed
  (Tier 2) on a mismatch.
- **Fixed — a full pipeline bypass.** S_w's median-over-6-layers
  aggregation is gameable: verbatim-copy just over half the matched
  layers from the real base, swap the rest for an independently-trained
  block, and the median clears the lineage threshold alone (~0.76-0.81
  vs ~0.59). Combined with adaptive training against the envelope score
  (reusing the existing A4 machinery), this evaded BOTH tests at once —
  100% attack success, clean envelope score, Tier 1 ADMIT. Fixed by
  requiring every matched layer to ALSO individually clear a floor
  calibrated on independent models' pooled per-layer scores
  (`pipeline.combined_lineage_verdict`), verified safe against genuine
  fine-tuned/pruned/quantized derivatives (per-layer minimums never
  below ~0.92 there, vs ~0.65 max for independent models). Side effect,
  reported not hidden: a fully permuted GENUINE derivative that used to
  be silently admitted is now correctly (if conservatively) blocked —
  fails closed, at a real availability cost until Git Re-Basin-style
  weight matching closes the permutation gap itself.
- **Fixed — silent NaN corruption.** A NaN calibration score could
  become the reported conformal threshold (np.sort places NaN last
  regardless of value), silently making the envelope test fail-open
  (never escalates). `conformal_quantile`/`envelope_test`/`lineage_test`
  now reject non-finite scores loudly.
- **Documented, not code-fixed — ESCALATE at a synchronous webhook.** A
  Kubernetes admission webhook call is synchronous with a bounded
  timeout; deep scan cannot finish inside it. Made explicit (module
  docstring + `02_paper.md` Section 3.5) that ESCALATE must map to deny
  at the webhook, not "not BLOCK so let it through" — the latter would
  silently defeat Tier 2 for every candidate.
- **Checked, came back clean.** A sweep of non-adaptive backdoor
  training intensity found no stealthy-but-effective middle ground at
  this synthetic-corpus scale — attack success and envelope-score
  anomaly rise together, not independently, across the whole grid
  tried. Reported as a real (if scale-limited) result, not assumed.

67 → 73 tests. Every fix verified against the same adversarial
reproduction that found the problem.

**Second pass, same day:** continued the adversarial testing further
(no GPU means E1–E9 stay blocked regardless, but this kind of testing
needs none). Found and fixed a sixth issue, on the other side of the
false-admit/false-reject ledger from item 2 above:

- **Fixed — structured channel pruning mis-scored on scattered drops.**
  `channel_aligned_directional_similarity` fell back to plain
  flattened-cosine similarity whenever channel counts differed (real
  structured pruning, as opposed to the magnitude pruning every existing
  pilot actually exercises) — reasoning the assignment problem "needs
  equal-sized sets." It doesn't; `scipy.optimize.linear_sum_assignment`
  handles rectangular matrices directly. The fallback was actively wrong
  whenever pruning dropped a SCATTERED subset of channels rather than a
  tidy trailing block (the realistic case for real saliency-based
  pruning): two structurally-pruned variants that both kept 6 of 8
  channels VERBATIM from the base scored 1.0 (prefix dropped) vs. ~0.13
  (scattered drop) — nearly as low as an unrelated model's ~-0.08 at the
  same shape. This is a false-rejection bug, not a security hole: it
  would have wrongly hurt a legitimate publisher who prunes the
  structurally-correct way. Fixed by running the rectangular assignment
  directly instead of falling back; both pruning variants now score 1.0.

67 → 74 tests (net +1: one obsolete fallback-behavior test replaced by
two new ones).

## Fourth revision pass (2026-09-30): an external review checked against the code

A separate session (one without this repository) reformatted the paper
into the Tech Con template and wrote a review, a fix plan and a
real-model evidence script. Each claim was checked here before being
applied; full detail is in
[`11_review_and_fix_plan_status.md`](11_review_and_fix_plan_status.md)
and [`10_critical_review_pilot.md`](10_critical_review_pilot.md), items
7–12.

- **Corrects this change log's third pass.** "Fixed a full pipeline
  bypass" was incomplete. The scorer skipped any layer whose shape
  differed from the base, so reshaping the spliced foreign layers left
  them unscored and re-opened the bypass: Tier 1 ADMIT at 100% attack
  success on 2 of 3 seeds. The channel-pruning fix likewise never reached
  the torch pipeline.
- **Fixed.** Sequential alignment scores every layer. Every unscored
  matrix counts against `max_exempt` (default 1) and is recorded in the
  attestation. Declared head replacement is supported via an ML-BOM
  property. Cascading permutation is now fully recovered on sequential
  networks. `verify.py` recomputes every bound digest and fails closed.
  `webhook.py` maps only ADMIT to `allowed: true`. The dead
  `gate_version` parameter and the unbound calibration sets are fixed.
- **Changed method.** An ablation showed singular spectra hurt
  separation, so S_w is now cosine-only by default. Pilot A lineage
  accuracy went 98.6% → 92.9% (blend, after the floor) → 98.6%
  (cosine-only). This was chosen on the synthetic corpus, so the
  real-model run must confirm it.
- **Now stated in the abstract and paper:**
  - the 10% benign false-block rate that was always in the pilot report;
  - that the gate does not stop a backdoored genuine fine-tune trained
    against the envelope;
  - that the per-layer floor has no formal guarantee;
  - that the conformal guarantee covers only self-made calibration
    derivatives;
  - that the probe-seed commitment is a reproducibility record, not a
    defence.
- **References.** [26]'s title corrected; [24]'s venue completed;
  authors added to [23], [25] and [26]. These came from search-index
  results, so confirm each byline before submitting.
- **External evidence script.** Added in `experiments/`. Its
  recommended quick run made both thresholds infinite and rejected every
  genuine model. It now refuses that configuration, and it aligns every
  model, calibration included. It has not been run on real weights:
  Hugging Face is blocked here.

## Fifth revision pass (2026-10-01): a 44-question developer checklist answered from the code

Every question was answered from the implementation, a test or a measured
run, in [`13_developer_checklist_answers.md`](13_developer_checklist_answers.md)
(also `SecureModelGate_Developer_Checklist_Answers.docx`). Doing that,
rather than answering from the paper, changed the paper.

- **Corrected headline numbers.** Pilot A's benign test set was leaky:
  pruned and quantized members were copies of the base, so 13 of 20 test
  models were byte-identical to calibration models. On a distinct
  population: 0/20 false blocks (CI 0–16.8%), 29/30 backdoors detected
  (one admitted, ASR 100%), AUROC 0.997, lineage 69/70. The 10% figure
  is withdrawn from the abstract and paper, with the reason stated.
- **New limitations, measured:**
  - an unrelated model trained toward the public base passes lineage
    4/4, and with an envelope penalty is admitted 4/4. Lineage is
    proximity, not provenance;
  - models sharing the base's random init score S_w ≈ 0.70;
  - ~25% foreign channels per layer still pass lineage;
  - benign fine-tunes 5× longer than calibrated are escalated 7/7;
  - a secret probe draw catches an attacker who overfits the exact
    pool (7/12) but not one who trains on the distribution (0/16).
- **Bugs fixed, each with a regression test:**
  - a fail-open where a NaN JS divergence was dropped by `max()`;
  - BatchNorm statistics outside the subject digest;
  - an unauthenticated DSSE payload type;
  - a verifier that did not re-derive thresholds, verdicts or scores;
  - attacker-controlled matching cost;
  - a stale default gate version;
  - an overclaiming seed-commitment test name.
- **Added:**
  - a per-model `evaluation.json`;
  - `securemodelgate defensibility-check` (all 63 splice patterns,
    S_w-adaptive and joint attacks, depth/pruning/quantization sweeps,
    siblings, probe variation, determinism, scaling);
  - `loader.py` (safetensors / weights_only; a live malicious pickle is
    refused and not executed);
  - deep-scan digest binding, a trusted-base registry, freshness and
    policy checks, a `replaceable` policy;
  - `tests/test_end_to_end_attacks.py`.

  129 tests pass.

## Real-model lineage run (2026-10-01)

The author ran `experiments/smg_bert_lineage.py` on a Mac, cosine-only
and with a 0.5 spectral blend. Results are in
[`pilot_results_bert/`](pilot_results_bert/README.md).

| Check | Cosine-only | 0.5 blend |
|---|---|---|
| Descendants accepted | 10/10 | 10/10 |
| Held-out MultiBERTs rejected | 5/5 | 5/5 |
| Splice, median-only rule | admitted | admitted |
| Splice, full rule | blocked (worst matrix 0.056) | blocked |
| FFN-permuted, unaligned / aligned | rejected / accepted | rejected / accepted |
| Weakest genuine matrix vs floor | 0.930 vs 0.128 (margin ≈ 0.80) | 0.965 vs 0.564 (margin ≈ 0.40) |

The blend also puts unrelated medians on the median threshold. This
confirms the cosine-only default on real weights; it had been chosen on
synthetic data.

Paper changes:
- Section 4 gains a real-model lineage subsection.
- The abstract gains one sentence.
- The "MultiBERTs may shrink the margin" caveat is replaced by the
  measured margin.

Stated limits: lineage only, honest models, small n, and descent taken
from model cards.

## Code-level counterpart of this change log

`securemodelgate/` (repo root) — install with `pip install -e ".[dev]"`,
run with `securemodelgate demo`, test with `pytest tests/`:

| Paper change | Code |
|---|---|
| Undefined F_ref → declared OMS-signed base | `securemodelgate/lineage/reference_resolver.py` |
| Ad-hoc 128-dim fingerprint → S_w + D_b | `securemodelgate/lineage/lineage_score.py`, `securemodelgate/lineage/behavioral_delta.py` |
| Hand-picked τ → split-conformal thresholds | `securemodelgate/lineage/conformal.py` |
| RSA-4096 JWT MAT → signed in-toto predicate | `securemodelgate/lineage/intoto_attestation.py` |
| Single pipeline → Tier 0/1/2 routing | `securemodelgate/lineage/admission_policy.py` |
| (new) end-to-end orchestration | `securemodelgate/pipeline.py` |

The original `from_Downloads_alt_code/src/static.py`,
`fingerprint.py`, `attestation.py` and `evaluator.py` are unchanged —
they are what produced the retained Parasparam 2026 numbers cited as
prior results in `02_paper.md`, Section 4.
