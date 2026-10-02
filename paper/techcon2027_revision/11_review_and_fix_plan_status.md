# External review and fix plan: status in this repo

A separate session, without access to this repository, reformatted the
paper into the Tech Con 2027 template (in that session's
`SecureModelGate_TechCon2027_submission.docx`, which is not in this
repo). It also wrote a review, a nine-item fix plan and
`experiments/smg_bert_lineage.py`. This file tracks that plan against
what was verified and done here. Findings from checking it are in
[`10_critical_review_pilot.md`](10_critical_review_pilot.md), items 7–11.

## The review's claims, checked

| Review claim | Verdict here |
|---|---|
| [26]'s title is wrong: it is "Weight Space Detection of Backdoors in LoRA Adapters" | **Correct.** Fixed, authors added. |
| Tan et al. [24] is LLMSC 2026, not "FSE '26 Companion" | **Partly right.** It is an LLMSC 2026 workshop paper, published in the FSE Companion '26 proceedings. The draft was incomplete rather than invented. Full venue and DOI added. |
| Several entries lacked authors | **Correct.** [23], [25] and [26] authors added. All of these came from search-index results (the landing pages were blocked here), so check each byline before submitting. |
| Pilot A numbers predate the per-layer floor | **Correct.** Re-run with the current code; see `06_engineering_validation_pilot.md`. |
| The floor rejects fine-tunes with a replaced head | **Correct** (the head scores 0.05 under cosine-only scoring). Declared replacement fixes lineage, **but the envelope test still escalates** those models (output JS 0.32). See item 10. |
| A fully permuted network is fixable by layer-by-layer alignment | **Correct.** Implemented as sequential alignment: S_w 0.60 → 1.00, and the model is admitted again. |
| The floor has no formal error guarantee | **Correct.** Pooled per-matrix scores are not exchangeable. Stated in the paper and the script. |
| The spectral component probably does little | **Correct, and understated.** It actively hurt: blended, the margin halves (worst genuine vs best unrelated per-layer minimum 0.917/0.573, against 0.848/0.148 cosine-only); spectra alone can't separate at all (0.986/0.986). Pilot A lineage accuracy is 65/70 blended vs 69/70 cosine-only. **Default switched to cosine-only**, a choice made on the synthetic corpus, so the BERT run should report both (`--spectral-weight`). |
| The script's self-test passes | **Correct.** But its recommended quick run makes both thresholds infinite and rejects every genuine model (item 11). Fixed. |

## Fix plan status

| # | Item | Status |
|---|---|---|
| 1 | Real-model evidence (`smg_bert_lineage.py`) | Script added and audited, with two fixes (item 11). **Run by the author on a Mac (2 runs).** Cosine-only: 10/10 descendants accepted, 5/5 held-out MultiBERTs rejected, splice blocked by the floor, margin ≈ 0.80. The 0.5 spectral blend halves the margin. See [`pilot_results_bert/`](pilot_results_bert/README.md); commit the JSON outputs there. |
| 2 | Exempt declared-replaced layers, with a cap | **Done.** Every unscored matrix (declared, missing, added, incomparable) counts against `max_exempt` (default 1), and the list is recorded in the predicate. |
| 3 | Propagate permutations across layers | **Done**, for sequential nets. Residual and transformer nets need a model-specific map; attention heads are not handled. |
| 4 | Re-run Pilot A with current code | **Done.** Lineage accuracy fell to 92.9% under the blend (5 warped-trigger backdoored fine-tunes whose classifier layer drifted just under the floor), then went back to 98.6% under cosine-only. Envelope side unchanged: 100% TPR, AUROC 0.998, **10% benign false-block rate** at α=0.05 |
| 5 | Confirm splice blocking and find where the floor stops working | **Done** (`securemodelgate lineage-scope-check`): the full rule admits 0/5 splices at every depth, k = 1–4 foreign matrices; median-only admits 4–5/5. Declared single-matrix replacement is admitted by design (that is legitimate transfer learning), and 2 or more declared hits the cap |
| 6 | Webhook: ESCALATE maps to deny | **Done**: `securemodelgate/webhook.py`. Only ADMIT is `allowed: true`. |
| 7 | Bind all digests; verification fails closed | **Done**: `securemodelgate/verify.py`. Also found and fixed a dead `gate_version` parameter, and unbound envelope and floor calibration sets. |
| 8 | Spectral ablation | **Done** (in `lineage-scope-check`). |
| 9 | BackdoorBench evaluation with re-run baselines | **Not possible here**: no GPU, CIFAR-10 blocked. |

## What still needs you

1. Run `python experiments/smg_bert_lineage.py --out results.json` on
   your Mac (about 12 GB). For a smaller run, loosen the alphas as the
   script's docstring says, or it will now refuse to start. Report the
   measured margin between genuine and unrelated models. Alignment
   raises unrelated models' classifier scores, and MultiBERTs share
   training data, so the margin may be tighter than in the synthetic
   pilot.
2. Copy the corrected references, re-run pilot numbers and new limits
   into the template `.docx` the other session produced.
3. Open the [23]–[26] landing pages and confirm each byline.
