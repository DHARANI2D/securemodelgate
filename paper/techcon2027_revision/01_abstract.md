# Tech Con 2027 Abstract (revised, ≈406 words — target is 250–350; over by ~55 to state what the gate does not stop and the real-model result; trim before submitting)

## SecureModelGate: Signed Behavioral-Lineage Attestation for Admitting Third-Party AI Models

Dharanidharan Senthilkumar, OLAA, CDRM – Cyber Defense Center, HPE, Bengaluru
(dharanidharan.senthilkumar@hpe.com)

Enterprises increasingly deploy models derived from a trusted base
rather than published by a trusted vendor: fine-tunes, LoRA adapters,
pruned and quantized variants. Model Atlas (Horwitz et al., NeurIPS 2025)
found that more than 60% of Hub models carry no documented parentage;
where a `base_model` field exists, it is optional and unverified.
OpenSSF Model Signing (OMS) and Kubernetes admission tools prove a
model's bytes are unchanged since signing — they say nothing about
whether a derivative really descends from the claimed base, or how far
its behavior has moved.

SecureModelGate closes this gap with a signed behavioral-lineage
attestation. At admission, the gate reads the declared ancestor from the
model's CycloneDX ML-BOM, verifies its OMS signature, and uses it as the
reference fingerprint for two calibrated tests: weight-space lineage
agreement with the base, and layer-wise representation similarity
(linear CKA) plus output divergence on a probe set drawn per admission.
Thresholds come from split-conformal calibration on benign derivatives, a
stated false-block bound instead of a hand-picked cut-off. The verdict
and evidence ship as a signed in-toto predicate driving risk-tiered
admission: trusted publisher (fast path), signed base (lineage + envelope
check), unsigned (deep scan).

Our earlier prototype caught all 20 of 40 CIFAR-10 ResNet-18 backdoored
models with a static weight check, but also blocked 3 of 20 clean models
(95% CI 3–38%), and its behavioral stage added nothing. The revised
evaluation re-runs Neural Cleanse, STRIP and MM-BD on identical
BackdoorBench models and adds an adaptive attacker; that run
([TPR@5%FPR/false-block/lineage-accuracy/latency = __]) is still pending
GPU time. A smaller
synthetic-corpus pilot shows the machinery working end to end: 29/30
non-adaptive backdoors escalated, 0.997 envelope AUROC, 98.6% lineage
accuracy and 0/20 benign false blocks. On real weights, lineage accepts
10/10 bert-base-uncased descendants and rejects 5/5 held-out MultiBERTs
and a layer splice. The pilot also shows where the gate breaks. One backdoor is admitted. Benign fine-tunes deeper than the
calibration set are escalated. An attacker who trains against the
envelope score, or who also pulls an unrelated model's weights toward
the public base, is admitted. Adversarial testing found and closed a
splice-and-reshape bypass, a fail-open path and a post-admission
tampering gap. The envelope test is triage and the lineage test is not
forgery-proof. What the gate guarantees is a signed, recomputable,
fail-closed verdict bound to the exact model bytes.
The talk demonstrates the gate for HPE Private Cloud AI, with a public
LLM/LoRA example, and proposes the predicate as an open format.

> **Note:** the bracketed values are unfilled. See
> [`04_experiments_plan.md`](04_experiments_plan.md) (E1–E9) for the
> experiments that produce them, and `02_paper.md`'s Caveats for what to
> submit if the four-day budget slips before Oct 2.
