# SecureModelGate — Tech Con 2027 Revision

This folder holds the revised abstract/paper for the Tech Con 2027
resubmission, and the source review material that drove the revision. It
sits alongside (not replacing) the original Parasparam 2026 materials in
`paper/` and `paper_and_readme.txt`, and the evaluation code in
`from_Downloads_alt_code/`. The revised methodology's *implementation*
lives in a separate, installable package at the repo root,
`securemodelgate/` — see "Code for the new methodology" below.

## What changed, in one line

The claim changed from "static + behavioral backdoor detector" (which the
Parasparam reviewers found unconvincing — 12.5/20 on technical
innovation) to **a signed, calibrated behavioral-lineage attestation**:
prove a derivative model really descends from its declared, OMS-signed
base, and that its behavior stays inside a conformally calibrated
benign-derivative envelope, then enforce that verdict at Kubernetes
admission.

## Files

| File | Contents |
|---|---|
| [`SecureModelGate_TechCon2027.docx`](SecureModelGate_TechCon2027.docx) | The paper below, formatted to the HPE Tech Con / Parasparam submission convention (title block, numbered sections, evidence figures, numbered references) — the file to actually hand to reviewers. Regenerate with `npm install && node build_docx.js` after editing `02_paper.md` (the script is hand-authored, not an auto-converter — mirror any content edit in both places). XSD-schema-validated (`validate.py`: 212 paragraphs, all checks pass); this session's sandboxed LibreOffice couldn't render a PDF preview to eyeball (confirmed environment-wide, not file-specific — even a blank `.txt` fails the same conversion here), so give it a look in Word before presenting it. |
| [`01_abstract.md`](01_abstract.md) | Revised ~320-word Tech Con abstract |
| [`02_paper.md`](02_paper.md) | Revised ~3-page paper (problem, threat model, contribution, evaluation table with placeholders, related work, business impact, roadmap, references) |
| [`03_change_log.md`](03_change_log.md) | Original → revision change log, with the reason for each change |
| [`04_experiments_plan.md`](04_experiments_plan.md) | E1–E9: the exact experiments that fill the paper's `[__]` placeholders |
| [`05_critical_review_source.md`](05_critical_review_source.md) | The critical review of the original Parasparam PDF that this revision responds to (reference/citation corrections, eligibility findings, product/regulatory updates) |
| [`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md) | A real, statistically-reported (Clopper-Pearson CIs, AUROC, TPR@FPR) run of the pipeline's own protocol against a small synthetic corpus — proof the design works end to end, **not** a substitute for E1–E9's numbers |
| [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md) | The A4 adaptive-attacker experiment (Experiment E7's shape, synthetic-corpus scale): an attacker who trains directly against the envelope score. Detection collapses from 100% to 20% by λ≈1 while attack success stays ≥99.8% — a real, honestly-reported limitation, not a claimed strength |
| [`08_presubmission_checklist.md`](08_presubmission_checklist.md) | **Read this before submitting anything.** Every citation added in the "no flaws" pass that couldn't be verified offline, itemized with what to check; Tech Con process questions; internal approvals needed; the now-3-days-out Oct 2 deadline; a mechanical sanity-check script |
| [`09_permutation_robustness_pilot.md`](09_permutation_robustness_pilot.md) | Tests Section 3.2's "resists permutation ... obfuscation" claim against a real, exact, function-preserving channel permutation. Found the claim was wrong as stated (a genuine derivative was wrongly rejected); fixed for the single-layer case (verified exact recovery); full-network cascading permutation is honestly reported as only partially fixed |
| [`10_critical_review_pilot.md`](10_critical_review_pilot.md) | Three rounds of adversarial testing of the implemented pipeline, each finding reproduced with runnable code. Found and fixed a **full pipeline bypass** by two routes: splicing, then reshaping the spliced layers so the scorer skipped them. Also fixed a reference-digest binding gap, a silent NaN-corruption path and a broken quick-run configuration in the external evidence script. Corrects two of its own earlier claims. Records the costs too: alignment inflation at the classifier, and envelope escalation of declared head replacements |
| [`12_implementation_report.md`](12_implementation_report.md) / [`SecureModelGate_Implementation_Report.docx`](SecureModelGate_Implementation_Report.docx) | Detailed implementation report: every module, what it does and why, the attack or failure it answers, the evidence, all pilot results, an attack-by-attack status table, every bug found and fixed, and what the gate does not do. Rebuild the .docx with `node build_report_docx.js` (a real markdown converter; edit only the .md) |
| [`13_developer_checklist_answers.md`](13_developer_checklist_answers.md) / [`SecureModelGate_Developer_Checklist_Answers.docx`](SecureModelGate_Developer_Checklist_Answers.docx) | A 44-question developer checklist answered from the code: each answer cites the function, the test and the measured number, or says plainly that it is not done. It also includes the 38-item short list and an end-to-end attack table with the exact test for each. Answering it found and fixed a leaky Pilot A test set, a NaN fail-open, BatchNorm statistics outside the digest and more, and measured that adaptive lineage forgery is admitted. Rebuild the .docx with `node build_report_docx.js 13_developer_checklist_answers.md SecureModelGate_Developer_Checklist_Answers.docx` |
| [`pilot_results_defensibility/`](pilot_results_defensibility/report.md) | `securemodelgate defensibility-check` output: splice patterns, adaptive and joint attacks, benign sweeps, siblings, determinism, scaling, calibration depth, probe variation |
| [`11_review_and_fix_plan_status.md`](11_review_and_fix_plan_status.md) | An external review's claims checked against the code, and its 9-item fix plan with each item's status: what's done here and what needs your Mac or a GPU |
| [`../../experiments/smg_bert_lineage.py`](../../experiments/smg_bert_lineage.py) | Real-model lineage evidence: `bert-base-uncased` fine-tunes against MultiBERTs. Audited and fixed here (it now refuses infinite thresholds and aligns every model). `--selftest` runs offline. **Run by the author on a Mac**; results in [`pilot_results_bert/`](pilot_results_bert/README.md) |
| [`pilot_results_bert/`](pilot_results_bert/README.md) | Real-model lineage results, transcribed from the author's two runs. Cosine-only: 10/10 BERT descendants accepted, 5/5 held-out MultiBERTs and a layer splice rejected, margin ≈ 0.80. The 0.5 spectral blend gives the same verdicts but halves the margin. Lineage only, small n |

## Status of the placeholders

Every `[__]` in `02_paper.md`'s evaluation table is a real placeholder,
not an invented number. This session did **not** run the E1–E9
experiment suite. Torch installs fine here, but as of 2026-09-29 this
environment has no GPU (`torch.cuda.is_available()` is `False`, and
neither of this account's two available Claude Code Remote environments
is GPU-tagged) and its network policy denies `huggingface.co`,
`download.pytorch.org` and the CIFAR-10 mirror — confirmed still
denied on retest after requesting a network-access widening. See
`04_experiments_plan.md` and
[`08_presubmission_checklist.md`](08_presubmission_checklist.md) §5 for
the exact status and what actually unblocks it, and the caveats section
of `02_paper.md` for what to do if the four-day budget slips regardless.

## Code for the new methodology

`securemodelgate/` (repo root) is an installable, tested Python package
implementing Section 3 of `02_paper.md` end to end: reference
resolution against a declared OMS-signed base, the S_w weight lineage
score, the D_b behavioral delta (linear CKA + JS divergence),
split-conformal thresholds, a signed in-toto predicate, and risk-tiered
admission routing — wired into one `pipeline.run_admission(...)` call
and a `securemodelgate demo` / `securemodelgate verify` CLI. Run it:

```bash
pip install -e ".[dev]"   # from the repo root
securemodelgate demo
securemodelgate evaluate --n-independent 40 --n-benign 40 --n-backdoored-per-trigger 10
securemodelgate adaptive --lambdas 0.0 0.5 1.0 2.0 5.0 --n-per-lambda 10
securemodelgate permutation-check
securemodelgate lineage-scope-check   # splice-depth sweep + spectral ablation
securemodelgate defensibility-check   # the developer-checklist measurements
python experiments/smg_bert_lineage.py --selftest
pytest tests/              # 129 tests: core maths, pipeline, CLI, stats, evaluation, adaptive attack, permutation, verification/webhook, hardening, end-to-end attacks
```

`evaluate` (which also writes a per-model `evaluation.json`) runs the paper's statistical protocol (split-conformal
calibration, Clopper-Pearson CIs, TPR@FPR, per-trigger breakdown,
lineage accuracy) against the same synthetic corpus and writes a report
+ figures — see
[`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md)
for a run's actual output. **It is not E1–E9** and its numbers do not
belong in `02_paper.md`'s Table 1.

See the repo root `README.md` for the architecture and what's real vs.
a signing/verification stand-in. All of the above runs against small,
fast-to-train synthetic torch models (`securemodelgate/demo_models.py`),
not the paper's PreAct-ResNet-18/CIFAR-10 BackdoorBench setup — that's
E1–E9 below, still to be run.

The original Parasparam 2026 pipeline (`from_Downloads_alt_code/src/`:
`static.py` / `fingerprint.py` / `attestation.py` / `evaluator.py`)
that produced the retained 20/20-detection, 3/20-false-block numbers is
untouched.
