# SecureModelGate — lineage-rule scope check

Synthetic corpus (`demo_models.TinyConvNet`), not E1–E9. Every decision uses `pipeline.lineage_verdict_for` (median S_w AND per-layer floor AND at most 1 unscored matrix), alpha=0.05.

## 1. Splice depth

A genuine fine-tune with its last k weight matrices (head first) replaced by an unrelated model's. Cells count splices ADMITTED by lineage (lower is better).

| k foreign matrices | n | median-only rule | full rule | full rule, replacements declared |
|---|---|---|---|---|
| 1 | 5 | 5 | 0 | 5 |
| 2 | 5 | 5 | 0 | 0 |
| 3 | 5 | 5 | 0 | 0 |
| 4 | 5 | 4 | 0 | 0 |

Declaring k <= max_exempt matrices replaced is admitted by design: that is exactly ordinary transfer learning with a new head, and lineage alone cannot tell it from a backdoored head. The envelope test is the only remaining check on that case.

## 2. Spectral ablation

| spectral weight | AUROC (median S_w) | AUROC (min layer) | worst genuine min-layer | best unrelated min-layer | genuine admitted | unrelated admitted | k=3 splices admitted |
|---|---|---|---|---|---|---|---|
| 0.0 | 1.000 | 1.000 | 0.848 | 0.148 | 13/13 | 0/20 | 0/5 |
| 0.5 | 1.000 | 1.000 | 0.917 | 0.573 | 13/13 | 0/20 | 0/5 |
| 1.0 | 0.954 | 0.996 | 0.986 | 0.986 | 9/13 | 0/20 | 0/5 |
