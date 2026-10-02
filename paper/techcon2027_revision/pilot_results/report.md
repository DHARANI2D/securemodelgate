# SecureModelGate — engineering validation pilot

**This is not Experiments E1-E9.** It is the same statistical protocol (split-conformal calibration, Clopper-Pearson CIs, TPR@FPR) run against `securemodelgate/demo_models.py`'s small synthetic corpus (a 4-stage CNN, synthetic Gaussian-noise images), not the paper's PreAct-ResNet-18/CIFAR-10/BackdoorBench protocol. Do not paste these numbers into the paper's Table 1 — see `paper/techcon2027_revision/04_experiments_plan.md` for what has to run instead.

Config: `EvaluationConfig(base_seed=1, n_independent_calibration=20, n_independent_test=20, n_benign_calibration=20, n_benign_test=20, n_backdoored_per_trigger=10, alpha_lineage=0.05, alpha_envelope=0.05, target_fpr=0.05, probe_seed=20260101)`

## Results

| Metric | Value |
|---|---|
| False-block rate (benign derivatives) | 0/20 = 0.0% (95% CI 0.0%-16.8%) |
| Backdoor TPR @ conformal threshold (α=0.05) | 29/30 = 96.7% (95% CI 82.8%-99.9%) |
| Backdoor TPR @ target FPR=0.05 | 96.7% |
| Envelope-score AUROC | 0.9967 |
| Lineage accuracy (positives vs. held-out negatives) | 69/70 = 98.6% (95% CI 92.3%-100.0%) |
| Median pipeline latency (S_w + D_b only) | 130.40 ms |

### Per-trigger-type breakdown

| Trigger | n | TP | TPR | Mean ASR |
|---|---|---|---|---|
| patch | 10 | 9 | 90.0% | 99.6% |
| blended | 10 | 10 | 100.0% | 98.5% |
| warped | 10 | 10 | 100.0% | 99.9% |
