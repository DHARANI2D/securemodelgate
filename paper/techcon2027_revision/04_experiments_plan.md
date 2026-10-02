# Experiments to Run Before Submission (E1–E9)

4 days, M-series or 1 GPU. None of these have run yet — as of
2026-09-29, torch installs fine in this environment but
`torch.cuda.is_available()` is `False` and its network policy denies
`huggingface.co`, `download.pytorch.org` and the CIFAR-10 mirror at
`www.cs.toronto.edu`, both confirmed on retest after requesting a
network-access change. See
[`08_presubmission_checklist.md`](08_presubmission_checklist.md) §5 for
the current status and what actually unblocks this. This is the exact
plan to execute once it is: fills the `[__]` placeholders in
[`02_paper.md`](02_paper.md).

1. **E1 – Model population (Day 1).** Train b0 (PreAct-ResNet-18,
   BackdoorBench, CIFAR-10). Generate 40 benign derivatives (20
   fine-tunes, 10 pruned, 10 INT8) and 50 backdoored derivatives (10 ×
   BadNets/Blended/WaNet/SIG/Input-aware). Log clean accuracy and ASR,
   and apply the inclusion rule (ASR ≥ 90%, CA drop ≤ 3 pts). Reuse the
   20 original clean models as lineage negatives. *Gate for all metrics
   below; report the model count actually admitted.*
2. **E2 – Calibration and false blocks (Day 2).** Split the benign
   derivatives 20/20. Set conformal τ at α = 0.05 (at n = 20 this is the
   20th of 20 sorted scores, i.e. the maximum — see
   `conformal.conformal_quantile` in the code scaffold) and apply it to
   the 20 held-out benign derivatives. Fills **[false-block rate = __
   (95% Clopper–Pearson CI __)]**.
3. **E3 – Backdoor escalation (Day 2).** Apply τ to the 50 backdoored
   derivatives. Fills **[TPR@5%FPR = __]**, the Tier 1 TPR cell (overall
   and per attack), and **AUROC of the named envelope score = __**.
4. **E4 – Lineage (Day 2).** Compute S_w for the 90 derivatives against
   the 20 independent models, with the threshold calibrated on
   independent models at α_L = 0.05. Fills **[lineage verification
   accuracy = __]** and the lineage false-accept rate.
5. **E5 – Baselines (Day 3).** Run Neural Cleanse, MM-BD and STRIP
   (aggregated per model, rule stated) on the same 90 + 20 models. Fills
   each baseline's **TPR, false-block rate (95% CI), median time/model**.
6. **E6 – Tiered pipeline (Day 3).** Run MM-BD only on Tier 1
   escalations. Fills the tiered row: **end-to-end TPR, false-block
   rate, fraction escalated, median time/model**.
7. **E7 – Adaptive attack (Day 4).** Train 10 backdoored derivatives
   with the loss + λ·(JS + 1 − CKA) on the public probe pool (λ ∈ {0.1,
   1}), then evaluate with per-admission random probe subsets. Fills
   **[TPR under adaptive attack = __]** and **[ASR retained = __]**. Run
   TWO attacker variants and report both, since the synthetic-corpus
   pilot below could not actually distinguish them (see
   [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md)'s "What
   probe pool did the attacker actually train against?"): (a) trained
   against the literal fixed evaluation-time probe pool (upper bound on
   attacker knowledge, violates the paper's committed-seed assumption on
   purpose to measure the ceiling) vs. (b) trained only against the
   public probe DISTRIBUTION with per-step resampling, never seeing the
   exact eval-time images (matches the paper's actual A4 threat model).
   The gap between (a) and (b), if any, is the first real evidence for
   or against the per-admission seed-commitment defense — real CIFAR-10
   images have exploitable per-image structure the synthetic corpus
   lacks, so this distinction may finally matter at this scale.
   A synthetic-corpus pilot of this exact attack already ran —
   `securemodelgate adaptive` /
   [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md) — using
   λ·max(1−CKA, JS) (the literal deployed envelope score, not the sum
   above) as the penalty, over a wider λ sweep (0, 0.5, 1, 2, 5). It
   found detection collapsing from 100% to 20% by λ≈1 with ASR retained
   ≥99.8% throughout. E7 should reuse that penalty formula (matching the
   real score, not an approximation of it) and expect a similar
   qualitative collapse, though not necessarily at the same λ.
8. **E8 – Latency (Day 4).** Time reference resolution, OMS verify, S_w,
   D_b and attestation sign/verify separately on M-series (and on the
   GPU if available). Fills **[median admission latency = __ s]** and
   the Tier 1 time cell.
9. **E9 – Optional LLM lineage demo (Day 4 if time allows).** Score a
   small public base against 3 public fine-tunes/LoRA adapters and a
   GGUF quantization, and against a same-architecture model from another
   family. Fills **[LLM lineage score gap = __]**. If it does not run,
   remove the phrase from the abstract rather than leaving a
   placeholder.

## What already exists to build E1–E9 on

- `from_Downloads_alt_code/src/models.py`, `static.py`, `fingerprint.py`,
  `attestation.py`, `evaluator.py`, `reporter.py` — the Parasparam 2026
  pipeline that trains the 40-model corpus and produced the retained
  numbers. E1 extends `models.py`'s generation logic to add benign
  fine-tuned/pruned/INT8 derivatives and the additional BackdoorBench
  trigger types.
- `securemodelgate/` (repo root; `pip install -e ".[dev]"`) — the new
  math (S_w, D_b, conformal thresholds, in-toto attestation, tier
  routing), wired end to end by `securemodelgate/pipeline.py` and
  already exercised by `securemodelgate demo` / `pytest tests/` against
  small synthetic models (35 passing tests). E2–E4, E6 and E8 wire that
  same `pipeline.run_admission(...)` call — the torch-dependent
  adapters it uses (`weight_lineage_score_from_models`,
  `compute_behavioral_delta_from_models`) already work against real
  `nn.Module`s — to real PreAct-ResNet-18 checkpoints and a CIFAR-10
  probe-set `DataLoader` instead of `securemodelgate/demo_models.py`'s
  synthetic ones.
- Neural Cleanse, MM-BD and STRIP (E5) are not vendored in this repo;
  BackdoorBench ships reference implementations of all three and is the
  recommended source rather than reimplementing them.

## If the four-day budget slips

Submit the abstract with only E2–E4 filled and state that the other
results are pending (see `02_paper.md`'s Caveats). Never leave invented
values in place of placeholders — an unfilled `[__]` is honest; a guessed
number is not.
