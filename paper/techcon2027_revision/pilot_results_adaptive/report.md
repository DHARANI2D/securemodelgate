# SecureModelGate — adaptive attacker (A4) pilot

Attacker knows the exact envelope score (`max(1 - min-layer CKA, JS)`) and trains a poisoned fine-tune with an added penalty `lambda * envelope_score` on a public probe pool (Section 2's A4; `securemodelgate.adaptive_attack`). lambda=0 is an ordinary, non-adaptive poisoned fine-tune.

Envelope conformal threshold used for detection: **0.1087** (alpha=0.05, n=20 benign calibration models).

| lambda | n | Detected | Mean ASR retained | Mean envelope score |
|---|---|---|---|---|
| 0.0 | 10 | 10/10 = 100.0% (95% CI 69.2%-100.0%) | 99.8% | 0.2914 |
| 0.5 | 10 | 8/10 = 80.0% (95% CI 44.4%-97.5%) | 100.0% | 0.1627 |
| 1.0 | 10 | 2/10 = 20.0% (95% CI 2.5%-55.6%) | 100.0% | 0.1082 |
| 2.0 | 10 | 3/10 = 30.0% (95% CI 6.7%-65.2%) | 100.0% | 0.1104 |
| 5.0 | 10 | 6/10 = 60.0% (95% CI 26.2%-87.8%) | 100.0% | 0.1389 |
