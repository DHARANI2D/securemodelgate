# SecureModelGate — defensibility checks

Synthetic corpus (`demo_models.TinyConvNet`), not E1–E9. Same calibration as `evaluation.py`; every decision is the gate's own rule. Thresholds: median S_w > 0.2144, every layer > 0.5080 (n=120 pooled), envelope ≤ 0.1108; alpha=0.05.

## 1. Splice patterns (lineage only)

| foreign matrices | n | admitted, undeclared | admitted, declared |
|---|---|---|---|
| 1 | 30 | 0 | 30 |
| 2 | 75 | 0 | 0 |
| 3 | 100 | 0 | 0 |
| 4 | 75 | 0 | 0 |
| 5 | 30 | 0 | 0 |
| 6 | 5 | 0 | 0 |

Partial-channel splice (a fraction of every matrix's output channels copied from an unrelated model):

| fraction foreign | n | lineage admitted | mean S_w | mean min-layer |
|---|---|---|---|---|
| 0.1 | 5 | 5 | 0.862 | 0.715 |
| 0.25 | 5 | 5 | 0.655 | 0.565 |
| 0.5 | 5 | 0 | 0.367 | 0.308 |
| 0.75 | 5 | 0 | 0.206 | 0.148 |

## 2. Adaptive attacks (end-to-end gate decision)

| start | mu (S_w pull) | lambda (envelope) | n | lineage pass | envelope pass | ADMIT | ADMIT with ASR≥0.9 | mean ASR | clean acc | mean S_w | mean min-layer | mean envelope |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unrelated | 0.0 | 0.0 | 4 | 0 | 0 | 0 | 0 | 1.00 | 0.77 | 0.184 | 0.129 | 0.191 |
| unrelated | 1.0 | 0.0 | 4 | 1 | 0 | 0 | 0 | 0.99 | 0.76 | 0.500 | 0.394 | 0.256 |
| unrelated | 3.0 | 0.0 | 4 | 4 | 0 | 0 | 0 | 1.00 | 0.73 | 0.867 | 0.865 | 0.207 |
| unrelated | 10.0 | 0.0 | 4 | 4 | 0 | 0 | 0 | 0.99 | 0.69 | 0.981 | 0.979 | 0.172 |
| unrelated | 3.0 | 1.0 | 4 | 4 | 1 | 1 | 1 | 1.00 | 0.63 | 0.832 | 0.831 | 0.120 |
| unrelated | 10.0 | 1.0 | 4 | 4 | 4 | 4 | 4 | 1.00 | 0.61 | 0.976 | 0.973 | 0.085 |
| base | 0.0 | 0.0 | 4 | 4 | 0 | 0 | 0 | 1.00 | 0.79 | 0.943 | 0.650 | 0.196 |
| base | 0.0 | 1.0 | 4 | 4 | 4 | 4 | 4 | 1.00 | 0.67 | 0.923 | 0.630 | 0.091 |
| base | 0.0 | 2.0 | 4 | 4 | 4 | 4 | 4 | 1.00 | 0.65 | 0.897 | 0.616 | 0.076 |

## 3. Declared head replacement

| head | n | lineage pass (declared) | lineage pass (undeclared) | decisions | mean ASR | clean acc |
|---|---|---|---|---|---|---|
| benign | 5 | 5 | 5 | ESCALATE, ESCALATE, ESCALATE, ESCALATE, ESCALATE | 0.00 | 0.71 |
| backdoored | 5 | 5 | 0 | ESCALATE, ESCALATE, ESCALATE, ESCALATE, ESCALATE | 0.94 | 0.62 |

## 4. Benign-derivative sweeps

| variant | n | decisions | min S_w | min min-layer | max envelope | clean acc |
|---|---|---|---|---|---|---|
| fine-tune 8 steps lr=0.005 | 3 | ADMIT, ADMIT, ADMIT | 1.000 | 1.000 | 0.086 | 0.65 |
| fine-tune 40 steps lr=0.005 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.999 | 0.990 | 0.219 | 0.72 |
| fine-tune 200 steps lr=0.005 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.993 | 0.964 | 0.194 | 0.82 |
| fine-tune 800 steps lr=0.005 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.955 | 0.927 | 0.219 | 0.88 |
| fine-tune 40 steps lr=0.05 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.958 | 0.923 | 0.328 | 0.79 |
| fine-tune 200 steps lr=0.05 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.855 | 0.744 | 0.445 | 0.75 |
| fine-tune 800 steps lr=0.05 | 3 | ESCALATE, ESCALATE, ESCALATE | 0.727 | 0.608 | 0.307 | 0.88 |
| magnitude prune 10% | 1 | ADMIT | 1.000 | 1.000 | 0.000 | 0.54 |
| magnitude prune 30% | 1 | ADMIT | 0.990 | 0.988 | 0.005 | 0.54 |
| magnitude prune 50% | 1 | ADMIT | 0.953 | 0.945 | 0.028 | 0.55 |
| magnitude prune 70% | 1 | ESCALATE | 0.859 | 0.838 | 0.112 | 0.49 |
| magnitude prune 90% | 1 | ESCALATE | 0.610 | 0.551 | 0.480 | 0.47 |
| quantize 256 levels/tensor | 1 | ADMIT | 1.000 | 1.000 | 0.000 | 0.54 |
| quantize 16 levels/tensor | 1 | ADMIT | 0.995 | 0.991 | 0.003 | 0.54 |
| quantize 8 levels/tensor | 1 | ADMIT | 0.977 | 0.961 | 0.018 | 0.54 |
| quantize 4 levels/tensor | 1 | ADMIT | 0.880 | 0.848 | 0.085 | 0.61 |
| quantize 2 levels/tensor | 1 | ESCALATE | 0.825 | 0.809 | 0.785 | 0.41 |
| Gaussian weight noise 0.01 x std | 1 | ADMIT | 1.000 | 1.000 | 0.000 | 0.54 |
| Gaussian weight noise 0.05 x std | 1 | ADMIT | 0.999 | 0.999 | 0.001 | 0.55 |
| Gaussian weight noise 0.1 x std | 1 | ADMIT | 0.995 | 0.995 | 0.004 | 0.55 |
| Gaussian weight noise 0.3 x std | 1 | ESCALATE | 0.958 | 0.955 | 0.198 | 0.68 |

## 5. Same-init siblings

| data seed | S_w | min-layer | envelope | decision |
|---|---|---|---|---|
| 500 | 0.707 | 0.527 | 0.178 | ESCALATE |
| 501 | 0.703 | 0.560 | 0.330 | ESCALATE |
| 502 | 0.720 | 0.491 | 0.482 | BLOCK |

## 6. Determinism (max abs difference over S_w, min-layer, min CKA, JS)

| comparison | max abs diff |
|---|---|
| same inputs, repeated | 0 |
| 1 thread vs 4 | 0 |
| probe batch 64 vs 16 | 0 |
| weights + 1e-6·std noise | 8.56e-08 |

## 7. Scaling (CPU)

| matched rows (x768 cols) | seconds | cost matrix MB |
|---|---|---|
| 64 | 0.001 | 0.0 |
| 256 | 0.006 | 0.5 |
| 768 | 0.157 | 4.7 |
| 1024 | 0.130 | 8.4 |
| 2048 | 0.626 | 33.6 |
| 3072 | 1.354 | 75.5 |

| TinyConvNet width | params | S_w s | D_b s (64 probes) |
|---|---|---|---|
| 8 | 25500 | 0.003 | 0.127 |
| 16 | 100532 | 0.019 | 0.129 |
| 32 | 399204 | 0.036 | 0.136 |
| 64 | 1590980 | 0.130 | 0.143 |

| probes | D_b seconds |
|---|---|
| 64 | 0.096 |
| 256 | 0.045 |
| 1024 | 0.265 |

## 8. Calibration-population depth (envelope test only)

| calibration | threshold | false blocks: 8 / 40 / 200-step fine-tunes | backdoors detected |
|---|---|---|---|
| evaluation calibration (8-step) | 0.1108 | 0/7 / 7/7 / 7/7 | 29/30 |
| mixed-depth calibration (8/40/200) | 0.2387 | 0/7 / 0/7 / 0/7 | 27/30 |

## 9. Genuine derivative under each benign transformation (`securemodelgate defensibility-check`)

| case | per-layer scores (stem, l1, l2, l3, l4, fc) | median S_w | scored | unscored | envelope | decision | clean acc |
|---|---|---|---|---|---|---|---|
| genuine fine-tune | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0591 | ADMIT | 0.642 |
| single-layer permuted | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0591 | ADMIT | 0.642 |
| all layers permuted | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0591 | ADMIT | 0.642 |
| magnitude-pruned 30% | 0.993, 0.991, 0.991, 0.989, 0.989, 0.988 | 0.9901 | 6 | 0 | 0.0554 | ADMIT | 0.637 |
| structured: 25% channels zeroed | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.1126 | ESCALATE | 0.675 |
| structured: 50% channels zeroed | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.6680 | ESCALATE | 0.403 |
| 16-level quantized | 0.996, 0.994, 0.995, 0.994, 0.991, 0.996 | 0.9946 | 6 | 0 | 0.0263 | ADMIT | 0.610 |
| FP16 round-trip | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0592 | ADMIT | 0.642 |
| BF16 round-trip | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0605 | ADMIT | 0.642 |
| INT8 symmetric per-tensor | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 0.9999 | 6 | 0 | 0.0611 | ADMIT | 0.650 |
| INT8 symmetric per-channel | 1.000, 1.000, 1.000, 1.000, 1.000, 1.000 | 1.0000 | 6 | 0 | 0.0590 | ADMIT | 0.642 |
| INT4 symmetric per-channel | 0.996, 0.995, 0.995, 0.995, 0.995, 0.996 | 0.9949 | 6 | 0 | 0.0454 | ADMIT | 0.623 |

Function preservation of the permutations, on 2000 fresh inputs:

| case | max abs logit diff | mean abs logit diff | prediction agreement | accuracy diff | envelope-score diff | CKA vs unpermuted | JS vs unpermuted |
|---|---|---|---|---|---|---|---|
| single-layer permuted | 9.54e-07 | 1.06e-07 | 1.0000 | +0.0000 | 1.07e-09 | 1.000000 | 1.53e-15 |
| all layers permuted | 2.86e-06 | 2.10e-07 | 1.0000 | +0.0000 | 3.12e-09 | 1.000000 | 3.65e-15 |

## 10. Unrelated models under other training recipes (lineage only)

| recipe | n | lineage admitted | S_w min / mean / max | highest min-layer |
|---|---|---|---|---|
| SGD lr=0.05, 40 steps (calibration recipe) | 5 | 0 | 0.158 / 0.173 / 0.192 | 0.132 |
| SGD lr=0.05, 200 steps | 5 | 0 | 0.179 / 0.210 / 0.235 | 0.160 |
| SGD lr=0.01, 150 steps | 5 | 0 | 0.181 / 0.200 / 0.214 | 0.137 |
| Adam lr=1e-3, 100 steps | 5 | 0 | 0.171 / 0.182 / 0.194 | 0.137 |
| Adam lr=1e-2, 100 steps | 5 | 0 | 0.185 / 0.203 / 0.212 | 0.176 |

## 11. Probe-set variation (envelope test, each probe draw calibrated separately)

| probe seed | threshold | benign false blocks | non-adaptive detected | adaptive (public draw) detected | adaptive (exact pool) detected |
|---|---|---|---|---|---|
| 20260101 | 0.1108 | 0/10 | 15/15 | 0/4 | 0/4 |
| 11 | 0.1344 | 0/10 | 15/15 | 0/4 | 3/4 |
| 12 | 0.1098 | 0/10 | 15/15 | 0/4 | 3/4 |
| 13 | 0.1084 | 0/10 | 15/15 | 0/4 | 1/4 |

Total runtime: 1485 s.
