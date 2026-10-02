# Real-model lineage evidence: bert-base-uncased vs MultiBERTs

`experiments/smg_bert_lineage.py`, run by the author on an Apple-silicon
Mac:

| Component | Version |
|---|---|
| Python | 3.14 |
| torch | 2.14.1 |
| transformers | 5.18.0 |
| scipy | 1.18.1 |
| numpy | 2.5.3 |

The script ran twice with default settings (20 calibration + 5 held-out
MultiBERTs, α = 0.05 for the median, α_floor = 0.001 for the floor,
max_exempt = 2): once cosine-only (the default) and once with
`--spectral-weight 0.5`.

**Source of these numbers.** They are transcribed from the console logs
of those two runs. The machine-readable outputs, `results_cosine.json`
and `results_blend.json`, are on the author's machine; commit them to
this folder so every number here has a file behind it.

## What was scored

- **Matrices:** the 72 encoder weight matrices of each model (12 blocks
  × Q, K, V, attention-output, FFN-in, FFN-out). Embeddings, the pooler
  and task heads are not scored, because the regex `KEEP` drops them.
- **Matching:** matrices wider than 512 rows (the 3,072-row FFN-in) are
  matched on 512 sampled rows.
- **Alignment:** every model, calibration included, is FFN-aligned
  before scoring.
- **Median threshold:** the conformal quantile of the 20 calibration
  medians. At n = 20 and α = 0.05 this is their maximum.
- **Floor threshold:** the conformal quantile of the 20 × 72 = 1,440
  pooled per-matrix scores. At α_floor = 0.001 this is their maximum.
- **Lineage only.** No behavioural envelope, no backdoors and no
  adaptive attacker were run on BERT.

## Results, cosine-only (the default)

T_median = **0.1115**, T_floor = **0.1282**. Runtime 21,392 s, most of it
downloading about 25 × 440 MB of weights.

| Model | Kind | Median | Min matrix | Decision |
|---|---|---|---|---|
| 20 MultiBERTs, seeds 0–19 | calibration | 0.111–0.112 | 0.055–0.056 | — |
| textattack/bert-base-uncased-SST-2 | descendant | 1.000 | 1.000 | accept |
| textattack/bert-base-uncased-MNLI | descendant | 0.999 | 0.998 | accept |
| textattack/bert-base-uncased-imdb | descendant | 1.000 | 0.999 | accept |
| textattack/bert-base-uncased-ag-news | descendant | 0.997 | 0.993 | accept |
| textattack/bert-base-uncased-yelp-polarity | descendant | 0.966 | **0.930** | accept |
| textattack/bert-base-uncased-QQP | descendant | 0.999 | 0.998 | accept |
| bhadresh-savani/bert-base-uncased-emotion | descendant | 1.000 | 1.000 | accept |
| SST-2 + per-row INT8 | descendant | 1.000 | 1.000 | accept |
| SST-2 + 30% magnitude pruning | descendant | 0.993 | 0.992 | accept |
| SST-2 + FFN-permuted, **not aligned** | descendant | 1.000 | 0.055 | **reject** (floor) |
| SST-2 + FFN-permuted, aligned | descendant | 1.000 | 1.000 | accept |
| MultiBERTs seeds 20–24 | held-out independent | 0.111 | 0.055–0.056 | reject |
| Splice: base blocks 0–6 + seed-20 blocks 7–11 | attack | 1.000 | 0.056 | median-only: **admit**; full rule: **reject** |

Summary as printed by the script:

| Check | Count | 95% Clopper–Pearson CI |
|---|---|---|
| Descendants accepted (with floor, aligned) | 10/10 | 69.2–100% |
| Held-out independents rejected | 5/5 | 47.8–100% |
| Splice admitted by the median-only rule | 1/1 | — |
| Splice blocked by the full rule | 1/1 | — |
| FFN-permuted fine-tune accepted without alignment | no | — |
| FFN-permuted fine-tune accepted with alignment | yes | — |

## Results, 0.5 spectral blend

T_median = **0.5549**, T_floor = **0.5636**. Runtime 833 s, from cache.

| Population | Median | Min matrix |
|---|---|---|
| Calibration MultiBERTs | 0.554–0.555 | 0.493–0.521 |
| Descendants (lowest: yelp-polarity) | ≥ 0.983 | ≥ 0.965 |
| Held-out MultiBERTs | 0.554–0.555 | 0.512–0.522 |
| Splice | 1.000 | 0.512 |

The verdicts are the same as cosine-only (10/10, 5/5, splice blocked).

## Reading

1. **Separation on real weights is wide.** With cosine-only scoring, the
   weakest genuine descendant's worst matrix (yelp-polarity, 0.930)
   sits 0.80 above the floor (0.128). The best unrelated matrix in 1,440
   scores 0.128. This is a much larger gap than the synthetic corpus,
   because no small classifier layer is scored. On the synthetic corpus
   a 4×64 head set the floor (0.508) through alignment inflation.
2. **The spectral blend hurts on real models too.** Blended, that margin
   halves (0.965 − 0.564 = 0.40). The unrelated models' medians
   (0.554–0.555) sit essentially *on* the median threshold (0.5549), so
   the median test alone barely separates; the floor decides. This
   confirms the cosine-only default, which had been chosen on the
   synthetic corpus only.
3. **The floor is what stops the splice.** A model that copies 7 of 12
   blocks verbatim scores median 1.000 and passes a median-only rule. Its
   foreign blocks score 0.056 and fail the floor.
4. **Alignment is necessary for BERT.** An exactly function-preserving
   FFN permutation drops the worst matrix to 0.055 (rejected) without
   alignment, and leaves it at 1.000 with it.

## What this does not show

- **Small n.** 7 public fine-tunes (6 from one publisher, textattack),
  3 local transforms of one of them, 5 held-out negatives and 1 splice.
  The CIs above are wide. These are counts, not rates.
- **Ground truth is the model card.** Descent from `bert-base-uncased`
  is what each publisher states; nothing here verifies it
  independently.
- **Honest models only.** No adaptive attacker was run. The synthetic
  corpus shows an unrelated model trained toward the public base passes
  lineage (`13_developer_checklist_answers.md`, Q7); nothing here
  suggests BERT is different.
- **No same-initialisation null.** The synthetic corpus shows models
  sharing the base's init score high. MultiBERTs' intermediate
  checkpoints of one seed would test that on real weights; not run.
- **No behaviour.** No envelope test, no backdoored BERT, no attention
  heads, embeddings or task heads scored.
- **Margins near the median threshold are thin by construction.** The
  calibration medians span only 0.111–0.112, so the median threshold is
  their maximum and held-out unrelated models sit just under it. The
  floor carries the decision for unrelated models.
- **Scoring time is not separated from download time** in the reported
  runtimes.
