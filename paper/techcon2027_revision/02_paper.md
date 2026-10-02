# SecureModelGate: Signed Behavioral-Lineage Attestation for Admitting Third-Party AI Models

Dharanidharan Senthilkumar — OLAA, CDRM – Cyber Defense Center, HPE, Bengaluru — dharanidharan.senthilkumar@hpe.com

*(~3-page revised paper, Tech Con 2027 resubmission)*

## Abstract

See [`01_abstract.md`](01_abstract.md).

## 1. Problem

Model supply chains are now mostly derivative. A trusted publisher
releases a base model, and third parties fine-tune, adapt (LoRA), prune,
quantize (INT8/GGUF) and re-publish it. Two recent developments make the
base trustworthy:

- **Signed bases.** OMS v1.0 (April 2025) gives publishers a detached
  Sigstore-bundle signature over every model file [29], and according to
  NVIDIA's Developer Blog, NVIDIA has signed all NVIDIA-published NGC
  Catalog models with OMS since March 2025, making NGC the first major
  model hub to do so [36].
- **Enforceable signatures.** Admission tools such as the Sigstore
  model-validation-operator [30] and the proposed KServe
  storage-initializer verification [31] can enforce those signatures.

A signature, however, covers only the exact bytes the publisher signed.
A derivative is different bytes, so it is either unsigned or signed by
an unknown party. Its claimed parentage sits in a self-reported, optional
`base_model` field, and Model Atlas (Horwitz et al. [35]) measured that
more than 60% of Hub models carry no documented parentage.

That leaves admission teams with two unanswered questions:

- **Q1 — lineage:** Is this model really a descendant of the signed base
  it claims?
- **Q2 — behavioral envelope:** Has it drifted from that base in ways
  benign derivatives do not?

Existing controls cover neighbouring risks. Pickle and code-execution
scanners address serialization attacks. PickleBall shows 44.9% of
popular models still ship pickle [27], and Zhao et al.'s MalHug scanner
found 91 malicious models among more than 705K monitored [28]. JFrog
says its scanner eliminates more than 96% of the false positives other
scanners produce on current Hugging Face models [37]. Deep backdoor
detectors (Neural Cleanse [8], MM-BD [10], BAIT [17]) are too slow or
too specialised to run on every import. Our Parasparam 2026 prototype also showed the cost
of lacking a reference: the behavioral stage compared models against an
undefined "clean reference" and contributed nothing.

## 2. Threat Model

We follow the general poisoning/backdoor taxonomy of Goldblum et al.
[3] and NIST AI 100-2e2025 [4], narrowed to the four adversaries below.

- **A1 — lineage forgery.** The adversary publishes a model claiming a
  popular OMS-signed base, e.g. to inherit its trust tier, but the model
  is independently trained or substituted. The gate's goal is to reject
  the lineage claim. Two distinct failure modes both defeat this
  adversary but route differently (Section 3.5), deliberately: a
  candidate with **no verifiable declared base at all** cannot even have
  S_w computed, so it drops to Tier 2 deep scan as the only applicable
  check; a candidate with a **verified base whose measured S_w fails the
  lineage threshold** has already been scored and positively found not
  to descend from what it claims, so it is blocked directly at Tier 1 —
  deep scan (pickle/code/backdoor scanners) checks for malicious
  payloads, not ancestry, so routing a known-false lineage claim through
  it instead of blocking would let a forged-provenance model reach
  admission on a clean scan result, defeating Q1 entirely.
- **A2 — poisoned true derivative.** The adversary fine-tunes the genuine
  base with a backdoor (BadNets [1], Blended [42], WaNet [21], SIG [41],
  Input-aware [21]; LLM trigger-response backdoors). The goal is to
  escalate it to deep scanning with a stated benign false-block rate. We
  treat this as calibrated triage, not a detection guarantee.
- **A3 — serialization/code payloads** (pickle, `trust_remote_code`).
  These are routed to existing scanners, and the gate records their
  results in the attestation.
- **A4 — adaptive adversary.** The adversary knows the full algorithm,
  features, public probe pool and calibration procedure (Kerckhoffs'
  principle, with no reliance on obscurity). An earlier draft also
  withheld the per-admission probe seed from the attacker. As
  implemented, that seed is not a secret (Section 3.4), so this model
  assumes the attacker may know it. The adversary may add fingerprint-matching regularisers
  during backdoor training. A synthetic-corpus pilot of exactly this
  attack (train directly against the envelope score) shows detection
  collapsing from 100% to 20% by penalty weight λ≈1, with attack success
  staying ≥99.8% throughout — see
  [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md). That
  pilot's attacker trained against probe batches freshly resampled every
  step from the public probe distribution, not the literal fixed
  evaluation-time pool — but on this synthetic corpus (i.i.d. Gaussian
  images with no natural-image structure to overfit) that distinction is
  close to moot, so the pilot demonstrates what a distribution-aware
  attacker achieves, not whether the per-admission committed seed adds
  protection beyond that; Experiment E7 is planned to test the two
  separately on real image data. We treat the collapse itself as
  confirming the envelope test is triage, not a guarantee, per Section
  3's framing, and expect E7 to find a similar qualitative result at the
  paper's real scale even though the specific λ won't transfer.
- **Trusted:** the base publisher's OMS signing identity, the gate's
  signing identity, and the cluster admission controller.
- **What the gate does not stop (stated up front).** Measured on the
  synthetic corpus
  ([`13_developer_checklist_answers.md`](13_developer_checklist_answers.md),
  Q7–Q8):
  - An adversary who backdoors a genuine fine-tune of the real base
    (A2) and trains against the envelope score (A4) is admitted (4/4
    at λ ≥ 1, ASR 100%). Lineage is true.
  - **An adaptive lineage forger is admitted too.** An unrelated,
    backdoored model trained with a penalty pulling its weights toward
    the public base passes lineage 4/4. With the envelope penalty
    added as well, 4/4 are admitted with 100% attack success. S_w
    measures weight-space proximity to the declared base, not
    provenance, and the base is public by construction.
  - A backdoor in a declared, replaced head is escalated, not
    admitted, but only because every head replacement escalates.

  Against these adversaries the gate contributes a signed, attributable
  audit record (who published what, claiming which base), not
  detection. What it does stop is:
  - non-adaptive forged lineage, including the splice and reshape
    variants (0/285 undeclared multi-matrix splices pass);
  - non-adaptive backdoors, at the triage stage (29/30; one is
    admitted);
  - tampered, replayed or re-signed attestations, and models changed
    after admission;
  - unsigned or unknown models, which are routed to deep scan.
- **Out of scope:** compromised publisher keys, backdoors already present
  in the signed base (inherited, so lineage-consistent), distillation
  (not a weight descendant; flagged "lineage unverified" and sent to deep
  scan), and all runtime or post-deployment monitoring. Runtime
  monitoring is the scope of the separate WITNESS proposal.

## 3. Contribution / Solution

**Innovation claim (single):** *SecureModelGate is, to our knowledge, the
first admission control that issues a signed, calibrated
behavioral-lineage attestation. It proves that a derivative model is a
weight-level descendant of its declared, OMS-signed base, and that the
model's behavioral deviation from that base falls within a conformally
calibrated benign-derivative envelope. It binds both results, with their
evidence, to the model's digests in an in-toto predicate that Kubernetes
admission policy enforces.* Signing, lineage scoring and backdoor
detection each exist separately. The contribution is making lineage a
verifiable, policy-enforceable supply-chain claim, with a stated error
bound.

> **Implementation.** `securemodelgate/lineage/` (repo root; `pip
> install -e ".[dev]"`, `securemodelgate demo`) implements the pieces
> below: `reference_resolver.py` (3.1), `lineage_score.py` and
> `behavioral_delta.py` (3.2), `conformal.py` (3.3),
> `intoto_attestation.py` (3.4), `admission_policy.py` (3.5), wired
> together by `securemodelgate/pipeline.py`. It runs and is tested end
> to end (`tests/`, 129 passing tests) against small, fast-to-train
> synthetic torch models, not the paper's PreAct-ResNet-18/CIFAR-10
> BackdoorBench evaluation (E1–E9, still to run). On that synthetic
> corpus, re-run with the current code
> ([`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md)):
> - envelope test: 29/30 non-adaptive backdoors detected (the miss, a
>   patch trigger with ASR 100%, is **admitted**), AUROC 0.997, and 0/20
>   benign false blocks at α = 0.05 (95% CI 0–16.8%);
> - lineage accuracy: 98.6% (69/70; the one miss is a genuine derivative
>   blocked, not an unrelated model admitted).
>
> On real weights (`bert-base-uncased` against 25 MultiBERTs; lineage
> only), the lineage rule accepts 10/10 descendants and rejects 5/5
> held-out unrelated models and a 7-of-12-block splice
> ([`pilot_results_bert/`](pilot_results_bert/README.md)).
>
> An earlier version of this pilot reported a 10% false-block rate
> (2/20). Its test set was leaky: 13 of the 20 "held-out" benign models
> were byte-identical to calibration models. Every number above comes
> from one per-model file (`pilot_results/evaluation.json`).
>
> The pilots found real problems as well as confirming the design:
> - a bug in S_w's parameter matching;
> - an overclaimed permutation-invariance property, now fully recovered
>   on sequential networks
>   ([`09_permutation_robustness_pilot.md`](09_permutation_robustness_pilot.md));
> - a **full pipeline bypass**: splicing layers to game S_w's median,
>   combined with adaptive training against the envelope score, reached
>   100% attack success at Tier 1 ADMIT. There were two routes to it, and
>   the first fix closed only one; the second (reshape the foreign layers
>   so they go unscored) was found a round later;
> - a reference-digest binding gap, a silent NaN-corruption path, two
>   unbound calibration sets and a dead `gate_version` parameter;
> - a spectral component that was measured to hurt separation;
> - answering a 44-question developer checklist from the code
>   ([`13_developer_checklist_answers.md`](13_developer_checklist_answers.md))
>   found more:
>   - the leaky benign test set above;
>   - a fail-open path (a NaN JS divergence was silently ignored);
>   - BatchNorm statistics outside the attestation's digest, so
>     post-admission edits went unnoticed;
>   - an unauthenticated DSSE payload type;
>   - a verifier that never re-derived thresholds or verdicts;
>   - an attacker-controlled matching cost.
>
> All are fixed ([`10_critical_review_pilot.md`](10_critical_review_pilot.md)).
> What is not fixed is stated in Section 2's "does not stop" list.

**3.1 Reference resolution (fixes the undefined F_ref).**

1. The candidate model *m* must ship a CycloneDX (1.6/1.7) ML-BOM whose
   machine-learning-model component lists the base *b* under
   `pedigree.ancestors`, identified by purl (e.g.,
   `pkg:huggingface/...`) and OMS bundle digest. The SPDX 3.0 equivalent
   is an AI-profile package with a `descendantOf` relationship.
2. The gate verifies *b*'s OMS signature against an allow-list of
   publisher identities.
3. The fingerprint of the verified *b* is F_ref.
4. If no signed ancestor is declared, the model drops to the deep-scan
   tier.

**3.2 Two-part fingerprint with justification.**

- **Weight lineage score S_w (answers Q1).** For each architecturally
  matched layer *l* — a weight **matrix** (conv kernels, linear
  weights); 1D parameters (biases, BatchNorm affine weight/bias) are
  excluded, since a 1D tensor has no singular-value spectrum to
  correlate and, worse, framework-default-initialized BatchNorm affine
  parameters stay near-identical across *unrelated* models trained only
  briefly, which is actively misleading rather than merely
  uninformative — compute the correlation between the singular-value
  spectra of W_l(m) and W_l(b), plus the cosine similarity between
  OPTIMALLY MATCHED output channels of W_l(m) and W_l(b) (Hungarian
  assignment on pairwise channel similarity; HuRef-style [15] in
  spirit). Layers are walked in order, and each layer's channel
  correspondence is carried into the next layer's input columns before
  that layer is matched (sequential alignment, see below). Lineage
  verifies only if three conditions hold:
  1. the median over layers clears its conformal threshold;
  2. **every** scored layer clears a per-layer floor, calibrated on
     unrelated models' pooled per-layer scores;
  3. at most `max_exempt` (default 1) weight matrices go unscored.

  Every matrix is either scored or recorded as unscored, with a reason:
  declared replaced in the ML-BOM, missing, added, or incomparable.
  Each condition closes a demonstrated attack:
  - **Median only:** a candidate that verbatim-copies just over half the
    layers and swaps the rest for a foreign block cleared it (S_w
    0.76–0.81 against a threshold of about 0.59, measured under the
    earlier 50/50 spectral blend).
  - **Median plus floor, without a cap:** the earlier scorer skipped any
    layer whose shape differed from the base, so widening each foreign
    block by one channel left it unscored. With adaptive training
    against the envelope score added, that got a 100%-success backdoor
    **admitted at Tier 1** on 2 of 3 seeds.

  Both routes are now blocked (all three reshaped attacks scored 6/6
  layers and were rejected). See
  [`10_critical_review_pilot.md`](10_critical_review_pilot.md), items 2
  and 7. The floor has **no formal error guarantee**: per-layer scores
  from one model are correlated, so pooling them breaks exchangeability.
  It only ever adds rejections, so false accepts stay bounded by the
  median test; its false-reject rate is measured, not guaranteed.
  - Why no spectra (a change from earlier drafts): the score was
    originally a 50/50 blend of channel-matched cosine and the
    correlation between singular-value spectra, justified by spectra's
    invariance to rotation. An ablation dropped them.
    - **Spectra don't discriminate.** Same-architecture models trained
      on the same data have nearly identical spectra.
    - **Spectra alone can't separate at all.** The worst genuine and
      best unrelated per-layer minimums were both 0.986, and 4 of 13
      genuine derivatives were rejected.
    - **The blend is worse than cosine alone.** Blending halves the
      margin cosine gives alone (0.917 vs 0.573, against 0.848 vs 0.148)
      and lowers Pilot A lineage accuracy (65/70 vs 69/70).

    The default is now cosine-only. That choice was made on the
    synthetic corpus, Pilot A's own population included, so the
    real-model run reports both settings
    ([`pilot_results_lineage_scope/report.md`](pilot_results_lineage_scope/report.md)).
  - Why channel-matched cosine, not flattened cosine: permuting a
    conv/linear layer's output channels — and correspondingly its
    BatchNorm affine params and the next layer's input channels — is an
    EXACT, function-preserving symmetry of any ReLU/BatchNorm network,
    not a hypothetical. An earlier version of this score used plain
    cosine similarity of the flattened tensor, which does NOT survive
    that permutation: measured on a real model, it collapsed a genuine
    derivative's per-layer score from ~1.0 to ~0.5 (statistically
    indistinguishable from an unrelated model) under a single layer's
    permutation. Matching channels via optimal assignment fixes a
    single layer's permutation exactly. A full-network CASCADING
    permutation (every layer, each forcing the next layer's input order
    to shift) at first recovered only 0.60, because each layer was
    matched against inputs its predecessor had scrambled; the per-layer
    floor then turned that into a false rejection of a benign model.
    Sequential alignment, a Git Re-Basin-style [43] weight matching,
    removes the cause: the fully permuted derivative now scores 1.00 on
    every layer and is admitted.

    Scope and cost:
    - **Scope.** Chaining follows parameter-registration order, which is
      data-flow order only for sequential networks. Residual and
      transformer blocks need a model-specific alignment map (the BERT
      evidence script has one for FFN blocks), and attention-head
      permutation is not handled.
    - **Cost.** Alignment also makes unrelated models look more alike at
      the classifier layer (null maximum 0.30 → 0.61 with cosine-only
      scoring), the Re-Basin effect itself. That layer alone sets the
      pooled floor (0.49 on the scope-check calibration, 0.508 on Pilot
      A's; the conv layers' null maximum is 0.34). The
      margin to the weakest genuine derivative is 0.848 − 0.49 ≈ 0.36.
      On real BERT weights, where no small head is scored, the margin is
      wider, not narrower: 0.930 − 0.128 ≈ 0.80 (Section 4.1, real-model
      lineage evidence).

    "Resists rotation obfuscation" is dropped: a general rotation is not
    a function-preserving symmetry of a ReLU network the way a
    permutation is ([`09_permutation_robustness_pilot.md`](09_permutation_robustness_pilot.md)).
  - Why rectangular assignment, not a same-shape restriction: real
    structured/channel pruning changes a layer's output-channel COUNT
    (unlike the magnitude pruning used elsewhere in this evaluation,
    which zeroes weights but keeps every tensor's shape). An earlier
    version of the channel-matched comparison fell back to plain
    flattened cosine whenever channel counts differed, on the reasoning
    that optimal assignment "needs equal-sized sets" — it doesn't;
    `scipy.optimize.linear_sum_assignment` matches min(m, n) channels
    directly on a rectangular cost matrix. In the scoring function the
    fallback was wrong for a scattered channel drop. Two variants that
    both kept 6 of 8 channels verbatim scored 1.0 when the dropped pair
    was a trailing block, but about 0.13 when it was scattered. Fixing
    that function did not reach the gate, though: the torch adapter
    skipped any shape-mismatched layer before scoring it, so at
    admission a structurally pruned layer went silently unscored. That
    was the loophole behind the reshaping attack above, not a false
    rejection. Now every layer is matched rectangularly.
    - **Extra channels** that no base channel explains count as zero
      similarity, so foreign capacity bolted onto a copied layer lowers
      its score.
    - **Pruned layers:** the next layer compares only the base input
      columns its surviving channels map to.
  - What permutation does NOT touch: it has no effect on D_b at all.
    Channel permutation is exactly function-preserving, so a candidate's
    activations and logits — and therefore its CKA/JS envelope score —
    are identical whether or not its weights are permuted (measured
    directly: a backdoored derivative's envelope score changed by
    <4e-9 under a full cascading permutation, i.e. numerically zero).
    Permutation is a pressure test on the WEIGHT-side ancestry claim
    only; an attacker who permutes a backdoored model to evade S_w gains
    nothing on the behavioral side, because the same envelope test still
    scores the identical function. Nor does permutation let an unrelated
    model forge ancestry: a fully permuted independent model is still
    blocked under sequential alignment (S_w 0.173, per-layer minimum
    0.139).
  - Why direction: HuRef [15] shows that base-model parameter directions
    barely move under SFT/RLHF, while independently trained models
    share no such alignment.
  - What S_w does not establish:
    - **Provenance against an adaptive attacker.** Training an
      unrelated model toward the public base's weights passes the
      lineage test (Section 2).
    - **Training lineage rather than shared initialisation.** Models
      sharing the base's random init but trained separately score
      S_w ≈ 0.70, and two of three pass.
    - **Freedom from foreign channels.** About a quarter of every
      layer's channels can be foreign and still pass.

    The lineage test separates honest derivatives from honest unrelated
    models; it is not a forgery-proof check
    ([`13_developer_checklist_answers.md`](13_developer_checklist_answers.md),
    Q4, Q7, Q9).
- **Behavioral delta D_b (answers Q2).** On a probe set P of k inputs,
  sampled for each admission from a public pool with seed *s*
  (commitment H(s) recorded first), compute:
  - per-layer linear CKA between m's and b's activations, and take the
    minimum over layers;
  - the Jensen–Shannon divergence between the output distributions. This
    replaces the original unanchored KL: it keeps the same idea but
    anchors it to the true base.
  - Why CKA: linear CKA [19] is invariant to orthogonal transforms and
    isotropic scaling of representations, so benign re-scaling and
    quantization noise should move it little. A backdoor must add a new
    feature-to-target mapping, which we hypothesise lowers late-layer
    CKA and raises output divergence on trigger-bearing directions more
    than benign fine-tuning does. This is a hypothesis the evaluation
    tests, not an assumption — and on the synthetic-corpus pilot it
    mostly holds against a non-adaptive attacker (AUROC 0.997, 29/30
    detected, one admitted;
    [`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md))
    but breaks down against the A4 adaptive attacker who optimizes
    directly against it (detection 100%→20% by λ≈1;
    [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md)). Both
    outcomes are pilot-scale, not Experiment E1/E7-scale, but both are
    measured, not asserted.
  - We also record the stable rank of ΔW_l = W_l(m) − W_l(b) in the
    final layers as a diagnostic, not a decision feature.

**3.3 Principled thresholds (replaces τ).** Two split-conformal [20]
tests with two calibration populations:

- **Lineage test.** Null hypothesis: *m* is unrelated to *b*. Calibrate
  S_w on independently trained models of the same architecture, and
  verify lineage if S_w exceeds their (1−α_L) conformal quantile.
- **Envelope test.** Null hypothesis: *m* is a benign derivative.
  Calibrate on *n* benign derivatives (fine-tuned, pruned, INT8), set τ
  to the ⌈(n+1)(1−α)⌉-th smallest calibration score, and escalate if the
  score exceeds τ. Under exchangeability, P(benign derivative escalated)
  ≤ α.
- **Consequence 1:** with n = 19 and α = 0.05, τ equals the maximum
  calibration score. The prior review's advice to "calibrate against the
  maximum clean score" is therefore this rule, now with a stated
  guarantee.
- **Consequence 2:** a 1% false-block claim needs n ≥ 99 calibration
  models. We report α together with the *n* it was calibrated on.
- **What the guarantee covers.** "≤ α" holds only for candidates
  exchangeable with the calibration set. The calibration derivatives
  are ones we made ourselves (fine-tuned, pruned and quantized under
  chosen settings), and real third-party fine-tunes are not
  exchangeable with them. So the claim is "calibrated on N self-made
  derivatives, α = …"; the real-world false-block rate is a separately
  measured number, not a consequence of the theorem. The synthetic
  pilot shows how narrow the coverage is. Held-out derivatives made
  like the calibration ones: 0/20 escalated. Benign fine-tunes 5×
  longer than the calibration recipe: 7/7 escalated. Calibrating on
  mixed depths removes those false blocks but detects 27/30 backdoors
  instead of 29/30. The per-layer floor has no such guarantee at all:
  it pools per-layer scores whose distributions differ (unrelated
  classifier layers score up to 0.61, conv layers ≤ 0.38).

**3.4 Attestation.** The attestation is an in-toto Statement v1 [32]
whose `subject` lists the same file digests as the model's OMS
manifest.

- **predicateType:** `https://securemodelgate.hpe.com/lineage/v0.1`
  (versioned; the domain must be approved internally). in-toto predicate
  types are not registered, and custom types are explicitly allowed [32].
- **Predicate fields:**
  - `declaredBase` {purl, omsBundleDigest, signerIdentity};
  - `mlbomDigest`;
  - `lineage` {S_w, threshold, minLayerScore, layerFloorThreshold,
    exempt, maxExempt, α_L, calibrationSetDigest,
    layerFloorCalibrationSetDigest};
  - `envelope` {minCKA, JS, threshold, α, n, calibrationSetDigest};
  - `probe` {poolDigest, seedCommitment, seed};
  - `tier`; `deepScan` {tool, version, resultDigest}; `verdict`;
    `gateVersion`; `timestamp`.

  Every threshold is traceable to the exact calibration data behind
  it, and `exempt` lists every unscored weight matrix with its reason.
- **Verification at admission** (`securemodelgate/verify.py`). A valid
  signature only proves the gate said something. The verifier also
  recomputes the candidate's weight digest, the ML-BOM digest, all
  three calibration-set digests, the probe-pool digest and the seed
  commitment from the artifacts actually presented. It fails closed on
  any mismatch, or when a Tier 1 attestation is presented without the
  inputs needed to check it. It also re-derives what the predicate
  asserts:
  - each threshold must be the conformal quantile of its bound
    calibration set;
  - the tier and verdict must follow from the recorded scores;
  - given the base, every score is recomputed (within 1e-6).

  So even a correctly signed but false predicate is rejected. Optional
  freshness, revocation (a trusted-base registry) and policy checks
  bound replay. The subject digest covers the full state dict,
  BatchNorm statistics included, and signatures cover DSSE's
  pre-authentication encoding. Tamper tests change each input in turn
  (`tests/test_verify_and_webhook.py`, `tests/test_hardening.py`,
  `tests/test_end_to_end_attacks.py`). Building this surfaced two
  binding gaps, both fixed: the `gate_version` parameter was dropped
  (every predicate said 0.1.0), and the envelope and floor calibration
  sets weren't digested at all.
- **The probe-seed commitment is a reproducibility record, not a
  defence.** The seed and its hash are published together in the same
  attestation, and a small integer seed can be recovered from its hash
  by enumeration. So the commitment proves nothing about ordering. The
  adaptive pilot (Section 2, A4) also shows that an attacker who knows
  only the probe distribution already collapses detection. Making the
  seed a real defence would require publishing a salted commitment
  before the candidate is submitted, and E7 testing a fixed-pool
  attacker against it.
- **Signing:** the gate signs a DSSE envelope as a Sigstore bundle. It
  uses keyless signing where the cluster is connected, and a private
  Sigstore instance or KMS key in air-gapped HPE Private Cloud
  deployments. This replaces the original RSA-4096 JWT "MAT" and bespoke
  JSON MBOM with standard, verifiable formats.
- **Placement:** the current model-signing tooling (1.1.1) documents no
  custom-predicate option, so the attestation ships alongside the OMS
  bundle, not inside it.

**3.5 Risk-tiered admission.** A policy engine verifies the OMS signature
and the SecureModelGate attestation, both bound to the same digests. It
can run as a validating webhook, Kyverno, or the Sigstore
policy-controller, and can also hook into the KServe storage initializer
for non-OCI storage URIs.

- **Tier 0:** trusted signed publisher and unmodified bytes → admit
  (signature check only).
- **Tier 1:** OMS-signed declared base, derivative → lineage + envelope
  test. Pass both → admit with attestation; fail envelope only → escalate
  to Tier 2 (lineage held, behavior is merely uncertain — deep scan can
  still clear it); fail lineage (S_w scored, below threshold) → block
  directly, without deep scan (see A1 above for why).
- **Tier 2:** no verifiable declared base (lineage cannot even be
  scored), or escalated from a failed envelope test → deep scan
  (pickle/code scanners, then MM-BD for classifiers or BAIT/PEFTGuard-
  class tools for LLMs/adapters), with human review on a positive
  result.

**A synchronous-webhook gap this design must state, not leave implicit.**
A Kubernetes `ValidatingAdmissionWebhook` call is synchronous with a
bounded timeout (a handful of seconds by default); deep scan is
explicitly "too slow ... to run on every import" (Section 1) and will
not finish inside that window. So the ESCALATE verdict cannot mean
"wait and see" at the webhook itself — the integration must map it to
`allowed: false` (deny) at the synchronous call, with deep scan running
out-of-band and a later re-submission or explicit approval carrying the
resolved ADMIT/BLOCK once it completes. A naive integration that reads
"not BLOCK" as "let it through" would silently defeat Tier 2 for every
candidate, since at the moment of the webhook call deep scan has not
run for any of them yet. `securemodelgate/webhook.py` implements the
required mapping, where only ADMIT becomes `allowed: true`. ESCALATE,
BLOCK and any unrecognised verdict deny. Any other integration (Kyverno,
policy-controller) must reproduce that mapping.

**Architecture figure (single figure, description):** A left-to-right
pipeline.

1. **Intake** — receives model files, the OMS bundle and the ML-BOM
   from Hugging Face, NGC or an internal registry.
2. **Reference Resolver** — reads `pedigree.ancestors`, fetches the base
   and verifies its OMS signature against the publisher allow-list.
3. **Tier Router** — branches into three lanes: Tier 0 (signature only),
   Tier 1 (the **Lineage Engine**: S_w, then probe-set CKA/JS against the
   base, then conformal decision using calibration sets stored as
   digests), and Tier 2 (the **Deep Scan** lane: pickle/code scanners,
   then MM-BD/BAIT).
4. **Attestor** — signs the in-toto predicate (Sigstore/KMS).
5. **Admission Controller** — on HPE Private Cloud AI Kubernetes,
   verifies the OMS and SecureModelGate attestations and admits or
   blocks the model.
6. A telemetry arrow carries every decision and score to HPE OpsRamp.

## 4. Evaluation

**4.1 Pilot validation (not part of this evaluation).** Before
committing GPU time to the protocol below, this revision's engineering
work was pressure-tested five separate times against a small synthetic
corpus — not to produce preliminary numbers for the table below (none
of these substitute for it), but to catch real implementation problems
before spending real GPU time on a broken design. All five found
something:

| Pilot | What it tested | Finding |
|---|---|---|
| [`06_engineering_validation_pilot.md`](06_engineering_validation_pilot.md) | Does the conformal/lineage/envelope machinery work end to end? | Caught a real bug (1D BatchNorm matching collapsed lineage separation to ~0.002). Current run: 29/30 backdoors detected (one admitted), AUROC 0.997, 98.6% lineage accuracy, 0/20 benign false blocks (CI 0–16.8%). An earlier 10% figure came from a leaky test set. |
| [`13_developer_checklist_answers.md`](13_developer_checklist_answers.md) | 44 developer questions answered from the code, with new measurements (all 63 splice patterns, S_w-adaptive and joint attacks, fine-tune depth, pruning, quantization, probe variation, determinism, scaling) | An adaptive lineage forger and a joint-objective attacker are admitted 4/4. Benign fine-tunes 5× longer than calibrated are escalated 7/7. Six implementation bugs were found and fixed, including a fail-open on NaN JS and BatchNorm statistics outside the digest. |
| [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md) | Does the envelope test survive an attacker who trains directly against it (A4)? | Detection collapses from 100% to 20% by penalty weight λ≈1, attack success stays ≥99.8% throughout. Reported as a real limitation, not fixed away. |
| [`09_permutation_robustness_pilot.md`](09_permutation_robustness_pilot.md) | Does S_w survive an exact, function-preserving channel permutation, as Section 3.2 claims? | Wrong as first stated: a genuine derivative was wrongly rejected. Now fully recovered on sequential networks (S_w 1.00) via sequential alignment. Residual/transformer nets need a model-specific map. |
| [`10_critical_review_pilot.md`](10_critical_review_pilot.md) | Three rounds of adversarial scope tests, including checking an external review against the code | A **full pipeline bypass** with two routes (splice; splice + reshape so the foreign layers go unscored), each reaching 100% attack success at Tier 1 ADMIT — both closed, and the full rule now admits 0/5 splices at every depth. Also fixed: a reference-digest binding gap, a silent NaN-corruption path, unbound calibration sets, and a broken quick-run config in the external evidence script. Spectra measured to hurt, so the default is now cosine-only. Two of its own earlier claims corrected. |

None of these numbers belong in the table below — they're a 4-stage CNN
on synthetic Gaussian images, not PreAct-ResNet-18 on CIFAR-10. What
they establish is that the design in Section 3 is implemented
correctly (or, where it wasn't, was found and either fixed or honestly
scoped) before the real evaluation spends GPU time on it.

**Real-model lineage evidence (lineage test only).** This was run by
the author with `experiments/smg_bert_lineage.py`
([`pilot_results_bert/`](pilot_results_bert/README.md)).

- **Base:** `bert-base-uncased`.
- **Descendants (10):** 7 public fine-tunes (SST-2, MNLI, IMDB,
  AG-News, Yelp, QQP, emotion), plus INT8, 30%-pruned and
  FFN-permuted copies of one of them.
- **Unrelated models:** 25 MultiBERTs (same architecture and data,
  different seed). 20 calibrate the thresholds; 5 are held out.
- **What is scored:** the 72 encoder weight matrices, each model
  FFN-aligned first, cosine-only.

Results:

| Check | Result |
|---|---|
| Descendants accepted | 10/10 (95% CI 69–100%) |
| Held-out unrelated models rejected | 5/5 (95% CI 48–100%) |
| Splice (7 base blocks + 5 MultiBERT blocks) | Admitted by a median-only rule (median 1.000); blocked by the floor (worst matrix 0.056) |
| FFN-permuted fine-tune | Rejected without alignment (worst matrix 0.055); accepted with it (1.000) |
| Weakest genuine matrix (Yelp fine-tune) vs floor | 0.930 vs 0.128: margin ≈ 0.80 |
| Best of 1,440 unrelated per-matrix scores | 0.128 |
| Same run with a 0.5 spectral blend | Same verdicts, but the margin halves (0.965 vs 0.564), and unrelated medians (0.554–0.555) sit on the median threshold (0.5549) |

The blend result confirms on real weights the cosine-only default that
was chosen on the synthetic corpus.

What this does not establish:
- the counts are small and mostly one publisher's fine-tunes;
- descent is taken from the model cards;
- no adaptive forger, no same-initialisation null, and no envelope or
  backdoor test was run on BERT.

So it shows that lineage separates honest BERT descendants from honest
unrelated BERTs, not that it resists the adaptive forgery measured
synthetically in Section 2.

**4.2 Retained from Parasparam 2026:** 40 CIFAR-10 ResNet-18 models (20
clean, 20 backdoored with BadNets, blended and high-frequency triggers),
3 runs, Apple M-series.

- The static check detected 20/20 (95% Clopper–Pearson CI 83–100%) and
  blocked 3/20 clean models (15%, CI 3–38%).
- Per-stage times were 1.26 s (static) and 1.28 s (behavioral).
- The behavioral stage at τ = 0.05 detected 0/20, consistent with the
  observed backdoored KL range of 0.001–0.034 nats (clean mean 0.00018,
  backdoored mean 0.0156). Because the maximum clean KL was not
  recorded, the earlier separability and AUC claims are withdrawn.

**4.3 Revised protocol (identical models for all methods):**

- **Base:** one PreAct-ResNet-18 base b0 trained with BackdoorBench on
  CIFAR-10.
- **Benign derivatives (n = 40, split 20 calibration / 20 test):**
  fine-tunes of b0 on clean subsets with varied seeds and learning
  rates, magnitude-pruned variants (10–50%), and INT8
  post-training-quantized variants.
- **Backdoored derivatives (n = 50):** 10 each of BadNets, Blended,
  WaNet, SIG and Input-aware, implanted by fine-tuning b0. Inclusion
  requires attack success rate ≥ 90% and clean-accuracy drop ≤ 3 points.
- **Lineage negatives:** the 20 independently trained clean models from
  the original study.
- **Baselines:** Neural Cleanse [8] and MM-BD [10] (model-level),
  re-run on the same models. STRIP [9] is an input-level runtime
  detector, so it is reported separately, aggregated to a model-level
  score with the
  aggregation rule stated.
- **Adaptive attack (A4):** the backdoor is trained with an added
  penalty λ·(JS + (1 − CKA)) against b0 on the public probe pool.
- **Metrics:** TPR at the conformal threshold (α = 0.05); false-block
  rate on held-out benign derivatives with 95% CI; lineage accuracy;
  AUROC using the named envelope score (max of the calibrated component
  p-values); median time per model.

| Method (same models) | Backdoor TPR (95% CI) | False-block rate, benign derivatives (95% CI) | Lineage accuracy | Median time/model |
|---|---|---|---|---|
| Original static check (Parasparam 2026; independent clean models) | 20/20 = 100% (83–100%) | 3/20 = 15% (3–38%)* | n/a | 1.26 s (M-series) |
| SecureModelGate Tier 1 (lineage + envelope) | [__] | [__] | [__] | [__] |
| SecureModelGate tiered (Tier 1 → MM-BD on escalation) | [__] | [__] | [__] | [__] |
| Neural Cleanse (re-run) | [__] | [__] | n/a | [__] |
| MM-BD (re-run) | [__] | [__] | n/a | [__] |
| STRIP (re-run, input-level, aggregated) | [__] | [__] | n/a | [__] |
| SecureModelGate Tier 1 under adaptive attack | [__] (ASR retained [__]) | — | [__] | — |**

*The original 15% was measured on independently trained clean models,
not on derivatives.

**The closest evidence available before E7 runs is the synthetic-corpus
pilot in [`07_adaptive_attack_pilot.md`](07_adaptive_attack_pilot.md):
detection 100%→20% by penalty weight λ≈1, ASR retained ≥99.8%. Do not
copy those numbers into this row — they are a different model/dataset
scale, cited here only so the row isn't read as untested.

**4.4 Current execution status.** The real-model lineage run (BERT,
above) is done. As of this writing, E1–E9 have not run: this working environment has neither a GPU nor network access to
`huggingface.co`/`download.pytorch.org`/the CIFAR-10 mirror, confirmed
structural (not a stale cache) by an independent fresh-session retest.
Schedule: E1–E9 in Oct–Nov 2026, on GPU compute with that network access
— see
[`08_presubmission_checklist.md`](08_presubmission_checklist.md) §5 for
the itemized diagnostic detail and unblock path.

**Planned extension (talk):** a public LLM lineage demo: a small open
base (e.g., Qwen2.5-0.5B) against public fine-tunes, LoRA adapters and
GGUF quantizations, and against a same-architecture model from another
family. It reports [LLM lineage score gap = __]. It is followed by
TrojAI rounds, whose final report covers weight-analysis and
trigger-inversion detectors and "natural" Trojans.

## 5. Related Work & Differentiation

- **Signing and admission.** Sigstore (Newman et al., CCS 2022 [2]), OMS
  v1.0 [29], the Sigstore model-validation-operator [30], KServe RFC
  #5789 [31], and policy engines such as Kyverno and the Sigstore
  policy-controller all verify byte integrity and signer identity.
  SecureModelGate consumes these and adds a claim about descent and
  behavior.
- **Attestation gates and lifecycle provenance.**
  - Tan et al. (LLMSC 2026, FSE '26 Companion [24]) enforce signed training/release claims
    with in-toto and CycloneDX [32] but leave backdoor-level evidence to
    optional plug-ins.
  - Atlas [25] records transformation attestations, but only inside an
    instrumented pipeline.
  - SecureModelGate works post hoc on third-party derivatives, with no
    producer cooperation beyond the base's OMS signature.
- **Lineage and fingerprinting.** HuRef (NeurIPS 2024 [15]), REEF (ICLR
  2025 [16]), modelDNA (2026 [22]) and Centered Residual Signatures
  (2026 [23]) score lineage but do not sign or enforce the verdict.
  SecureModelGate reuses these signal families rather than claiming a
  new lineage metric.
- **Backdoor detection.** Neural Cleanse [8], ABS [13], MNTD [14],
  UNICORN [11], BTI-DBF [12], MM-BD [10], BAIT [17] and PEFTGuard (S&P
  2025 [18]), and weights-only LoRA detection (2026 preprint [26]) are
  deep detectors. SecureModelGate places them behind a calibrated
  triage tier. BackdoorBench [7] and the TrojAI final report [6] supply
  the evaluation methodology.
- **Commercial scanners.** Palo Alto Networks Prisma AIRS (Protect AI,
  acquisition completed July 2025 [38]), Cisco AI Defense [39] and
  JFrog's Hugging Face scanning [37] focus on malicious code,
  vulnerabilities and posture. In public documentation we found no
  product that issues a signed, calibrated lineage verdict.

## 6. Business Impact for HPE

- **Product fit.** HPE Private Cloud AI and HPE AI Essentials Software
  customers import open models and create their own fine-tunes.
  SecureModelGate gives them fast admission for NGC-signed bases (Tier
  0), a verifiable lineage record for derivatives (Tier 1), and a
  deep-scan lane for unknown models. It fits the Kubernetes management
  in HPE Private Cloud (Morpheus-based, announced May 2026) and
  air-gapped deployments (private Sigstore/KMS signing).
- **Operations.** Decisions and scores stream to HPE OpsRamp and can
  feed GreenLake Intelligence workflows, so model-admission posture
  becomes observable fleet-wide.
- **Compliance evidence.** The signed predicate is machine-readable
  evidence for, all under Regulation (EU) 2024/1689 [34]:
  - EU AI Act Art. 11/Annex IV technical documentation;
  - Art. 15(5) resilience against data and model poisoning;
  - Art. 53 GPAI downstream-provider information.

  The Digital Omnibus, Regulation (EU) 2026/1744 [34] (in force 27 July
  2026), moved Annex III high-risk obligations to 2 December 2027. The
  work also aligns with the NSA-led multinational "AI/ML Supply Chain
  Risks and Mitigations" CSI [33] (4 March 2026; AI-BOMs and
  cryptographic integrity checks), CISA's "AI Data Security" [33] (May
  2025), NIST AI RMF, AI 600-1 [5] and AI 100-2e2025 [4], and the EO
  14028/14144/14306 secure-software attestation regime [40].
- **Differentiation.** Competitors scan artifacts; HPE would attest
  lineage. The predicate format could also be contributed to the OpenSSF
  AI/ML Working Group.

## 7. Status & Roadmap

- **Status (as of 2026-09-30):** the Parasparam 2026 prototype (static +
  behavioral stages, CLI) remains as-is. The lineage engine, conformal
  calibration and in-toto attestor are implemented, installable, tested,
  and validated by four independent pilots end to end against small
  synthetic models: `securemodelgate/` at the repo root (`pip install
  -e ".[dev]"`, `securemodelgate demo`). What remains is wiring it to
  the real evaluation corpus (E1–E9) instead of the demo's synthetic
  models — currently blocked on GPU compute and network access to
  CIFAR-10/Hugging Face (Section 4.4;
  [`08_presubmission_checklist.md`](08_presubmission_checklist.md) §5
  has the current status and unblock path).
- **By Oct 2 2026:** the 4-day core evaluation (Section 4) fills the
  placeholders in the table and abstract — see
  [`04_experiments_plan.md`](04_experiments_plan.md) (E1–E9). **At risk**
  given the blocker above; if it slips, follow the Caveats section's
  "Time budget" guidance rather than the date silently passing.
- **Q4 2026:** LLM/LoRA/GGUF lineage demo on public models; admission
  policy (Kyverno/policy-controller) on an HPE Private Cloud AI test
  cluster; OpsRamp telemetry.
- **Q1 2027:** TrojAI-round evaluation and scaled calibration (n ≥ 99
  per architecture family, for 1% false-block claims); draft predicate
  spec shared with the OpenSSF AI/ML WG.
- **Tech Con 2027:** live demo of tiered admission and the lineage
  attestation.

## References

1. T. Gu, B. Dolan-Gavitt, S. Garg, "BadNets: Identifying
   Vulnerabilities in the Machine Learning Model Supply Chain,"
   arXiv:1708.06733, 2017. Journal version: T. Gu, K. Liu, B.
   Dolan-Gavitt, S. Garg, "BadNets: Evaluating Backdooring Attacks on
   Deep Neural Networks," IEEE Access 7:47230–47244, 2019,
   doi:10.1109/ACCESS.2019.2909068.
2. Z. Newman, J. S. Meyers, S. Torres-Arias, "Sigstore: Software
   Signing for Everybody," ACM CCS 2022, doi:10.1145/3548606.3560596.
3. M. Goldblum et al., "Dataset Security for Machine Learning: Data
   Poisoning, Backdoor Attacks, and Defenses," IEEE TPAMI
   45(2):1563–1580, 2023, doi:10.1109/TPAMI.2022.3162397;
   arXiv:2012.10544.
4. A. Vassilev et al., "Adversarial Machine Learning: A Taxonomy and
   Terminology of Attacks and Mitigations," NIST AI 100-2e2025, 2025,
   doi:10.6028/NIST.AI.100-2e2025.
5. NIST, "AI RMF: Generative AI Profile," NIST AI 600-1, 2024,
   doi:10.6028/NIST.AI.600-1.
6. K. W. Reese et al., "Trojans in Artificial Intelligence (TrojAI)
   Final Report," arXiv:2602.07152, 2026.
7. B. Wu et al., "BackdoorBench: A Comprehensive Benchmark of Backdoor
   Learning," NeurIPS 2022 Datasets & Benchmarks, arXiv:2206.12654.
8. B. Wang et al., "Neural Cleanse: Identifying and Mitigating Backdoor
   Attacks in Neural Networks," IEEE S&P 2019, doi:10.1109/SP.2019.00031.
9. Y. Gao et al., "STRIP: A Defence Against Trojan Attacks on Deep
   Neural Networks," ACSAC 2019, doi:10.1145/3359789.3359790;
   arXiv:1902.06531.
10. H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, "MM-BD: Post-Training
    Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types
    Using a Maximum Margin Statistic," IEEE S&P 2024,
    doi:10.1109/SP54263.2024.00015; arXiv:2205.06900.
11. Z. Wang, K. Mei, J. Zhai, S. Ma, "UNICORN: A Unified Backdoor
    Trigger Inversion Framework," ICLR 2023, arXiv:2304.02786.
12. X. Xu et al., "Towards Reliable and Efficient Backdoor Trigger
    Inversion via Decoupling Benign Features" (BTI-DBF), ICLR 2024,
    OpenReview id Tw9wemV6cb.
13. Y. Liu et al., "ABS: Scanning Neural Networks for Back-doors by
    Artificial Brain Stimulation," ACM CCS 2019,
    doi:10.1145/3319535.3363216.
14. X. Xu et al., "Detecting AI Trojans Using Meta Neural Analysis"
    (MNTD), IEEE S&P 2021, doi:10.1109/SP40001.2021.00034;
    arXiv:1910.03137.
15. B. Zeng et al., "HuRef: HUman-REadable Fingerprint for Large
    Language Models," NeurIPS 2024, arXiv:2312.04828.
16. J. Zhang et al., "REEF: Representation Encoding Fingerprints for
    Large Language Models," ICLR 2025, arXiv:2410.14273.
17. G. Shen et al., "BAIT: Large Language Model Backdoor Scanning by
    Inverting Attack Target," IEEE S&P 2025, pp. 1676–1694.
18. Z. Sun et al., "PEFTGuard: Detecting Backdoor Attacks Against
    Parameter-Efficient Fine-Tuning," IEEE S&P 2025, pp. 1713–1731.
19. S. Kornblith, M. Norouzi, H. Lee, G. Hinton, "Similarity of Neural
    Network Representations Revisited," ICML 2019, arXiv:1905.00414.
20. A. N. Angelopoulos, S. Bates, "A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification,"
    arXiv:2107.07511.
21. T. A. Nguyen, A. Tran, "WaNet – Imperceptible Warping-based
    Backdoor Attack," ICLR 2021, arXiv:2102.10369; "Input-Aware Dynamic
    Backdoor Attack," NeurIPS 2020, arXiv:2010.08138.
22. M. A. Bin Adil, S. Aamir, "modelDNA: Calibrated Lineage
    Verification and Merge Decomposition from Sampled Weight
    Fingerprints," arXiv:2607.10617, 2026.
23. A. S. Thakur, R. Khoury, "Training Leaves Traces: Centered Residual
    Signatures for Language Model Lineage Verification,"
    arXiv:2608.14929, 2026 (preprint).
24. Z. Tan, J. Singer, C. Anagnostopoulos, "Attesting LLM Pipelines:
    Enforcing Verifiable Training and Release Claims," 2nd Int'l
    Workshop on LLM Supply Chain Analysis (LLMSC 2026), co-located with
    FSE '26, in FSE Companion '26, ACM, doi:10.1145/3803437.3805530;
    arXiv:2603.28988.
25. M. Spoczynski, M. S. Melara, S. Szyller, "Atlas: A Framework for ML
    Lifecycle Provenance & Transparency," IEEE EuroS&PW 2025,
    pp. 448–461; arXiv:2502.19567.
26. D. Puertolas Merenciano, E. Vasyagina, K. Zhu, J. Ferrando,
    M. Chaudhary, "Weight Space Detection of Backdoors in LoRA
    Adapters," arXiv:2602.15195, 2026 (preprint). An earlier draft of
    this entry gave the title as "Detecting Backdoored LoRAs from
    Weights Alone," which is not this paper's title.
27. A. D. Kellas et al., "PickleBall: Secure Deserialization of
    Pickle-based Machine Learning Models," ACM CCS 2025,
    doi:10.1145/3719027.3765037; arXiv:2508.15987.
28. J. Zhao et al., "Models Are Codes: Towards Measuring Malicious Code
    Poisoning Attacks on Pre-trained Model Hubs," ASE 2024,
    doi:10.1145/3691620.3695271; arXiv:2409.09368.
29. OpenSSF, "OpenSSF Model Signing (OMS) Specification,"
    github.com/ossf/model-signing-spec; "Launch of Model Signing v1.0,"
    OpenSSF blog, 4 Apr 2025.
30. Sigstore, "model-validation-operator,"
    github.com/sigstore/model-validation-operator; Sigstore blog,
    "Trusting AI Models in Kubernetes."
31. KServe, "RFC: Model artifact integrity verification for non-OCI
    storage URIs," Issue #5789.
32. in-toto Attestation Framework v1 (Statement, Predicate specs),
    github.com/in-toto/attestation; CycloneDX v1.7 specification
    (ML-BOM, pedigree).
33. NSA AISC et al., "Artificial Intelligence and Machine Learning –
    Supply Chain Risks and Mitigations," CSI, 4 Mar 2026; NSA/CISA/FBI
    et al., "AI Data Security," 22 May 2025.
34. Regulation (EU) 2024/1689 (AI Act); Regulation (EU) 2026/1744
    (Digital Omnibus on AI).
35. E. Horwitz et al., "Model Atlas," NeurIPS 2025 (position paper track),
    arXiv:2503.10633; project page horwitz.ai/model-atlas. Verified
    against the primary source (previously cited only secondhand via
    modelDNA [22]).
36. NVIDIA Developer Blog, "Bringing Verifiable Trust to AI Models: Model
    Signing in NGC," developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc,
    28 Jul 2025. Confirms NGC Catalog models signed with OpenSSF Model
    Signing since March 2025.
37. JFrog and Hugging Face, "JFrog and Hugging Face Join Forces," 
    jfrog.com/blog/jfrog-and-hugging-face-join-forces (mirrored at
    huggingface.co/blog/jfrog); confirms the >96% false-positive-reduction
    figure for JFrog's Hugging Face Hub scanning/curation.
38. Palo Alto Networks, "Palo Alto Networks Completes Acquisition of
    Protect AI," press release, 22 Jul 2025,
    paloaltonetworks.com/company/press/2025/palo-alto-networks-completes-acquisition-of-protect-ai;
    now inside Prisma AIRS.
39. Cisco, "AI Defense" product documentation, on its AI Supply Chain
    Risk Management capability — exact URL/version still unconfirmed
    (no primary source was located for this pass). **Locate and cite
    before submission, or drop the Cisco AI Defense mention from
    Section 5 if it cannot be confirmed in time.**
40. Executive Order 14028, "Improving the Nation's Cybersecurity," 12 May
    2021 (establishes the CISA self-attestation regime this paper cites).
    Its provisions were built upon by Executive Order 14144,
    "Strengthening and Promoting Innovation in the Nation's
    Cybersecurity," 16 Jan 2025, which was in turn amended by Executive
    Order 14306, "Sustaining Select Efforts To Strengthen the Nation's
    Cybersecurity," 6 Jun 2025 (EO 14306 amends EO 14144 and, separately,
    EO 13694 — it does not amend EO 14028 directly; verified against
    whitehouse.gov and the Federal Register, correcting an earlier draft
    of this reference).
41. M. Barni, K. Kallas, B. Tondi, "A New Backdoor Attack in CNNs by
    Training Set Corruption Without Label Poisoning" (SIG), IEEE ICIP
    2019. Verified against the BackdoorBench benchmark's own citation
    for its `SIG` attack implementation (github.com/SCLBD/BackdoorBench,
    `attack/sig.py`), not merely a secondary index — resolves the
    previously-withdrawn placeholder for Section 2's A2 attack list.
42. X. Chen, C. Liu, B. Li, K. Lu, D. Song, "Targeted Backdoor Attacks on
    Deep Learning Systems Using Data Poisoning" (Blended),
    arXiv:1712.05526, 2017.
43. S. K. Ainsworth, J. Hayase, S. Srinivasa, "Git Re-Basin: Merging
    Models modulo Permutation Symmetries," ICLR 2023, arXiv:2209.04836.
    Cited in Section 3.2 as the concrete direction for fully solving
    cascading channel-permutation recovery in S_w.

## Caveats

- **Unreviewed sources.** Several 2026 sources are preprints without
  peer review: modelDNA, Centered Residual Signatures, weights-only LoRA
  detection, and the Tan et al. short paper. Cite them as preprints. The
  2.9M-repository figure comes from the modelDNA paper and the ">60% no
  parentage" figure from Model Atlas (Horwitz et al.), as cited by
  modelDNA, not from Hugging Face directly.
- **Unconfirmed metadata.** HuRef's and REEF's venues and author lists,
  and the NIST AI 600-1 DOI, were confirmed only through secondary
  indexes; check their landing pages before submitting. BTI-DBF has no
  arXiv version, so it is cited via OpenReview.
- **OMS tooling.** OMS custom-predicate embedding is described in the
  spec but not exposed by the current tooling (model-signing 1.1.1).
  Hence "alongside the OMS bundle"; do not claim the predicate sits
  inside it.
- **Internal approvals.** The predicate URI domain and any
  product-integration statements need internal approval. HPE
  integration is proposed, not shipped.
- **Tech Con context.** Tech Con is internal and competitive: 722
  abstracts in 2025 and 80 talks in 2026, per figures supplied to this
  revision. Confirm internally that a paper not selected at Parasparam is
  eligible for resubmission to Tech Con 2027 (see
  [`05_critical_review_source.md`](05_critical_review_source.md), part
  (a), for what is/isn't publicly documented about eligibility).
- **Time budget.** If the four-day budget slips, submit the abstract
  with only E2–E4 filled and state that the other results are pending.
  Never leave invented values in place of placeholders.
- **References [35]–[43].** [35] Model Atlas, [36] NVIDIA, [37] JFrog,
  [38] Palo Alto/Protect AI, [40] the EO 14028/14144/14306 chain, [41]
  SIG, [42] Blended and [43] Git Re-Basin have all been verified against
  primary or benchmark-code sources in this pass and corrected where an
  earlier draft mischaracterized them (see [40]'s entry for the EO
  amendment-chain correction, and [41]'s entry for how "SIG" was
  resolved to a specific paper rather than a secondary-index guess). [39]
  (Cisco AI Defense) remains unconfirmed — no primary source was located
  for it; locate one or drop the Cisco mention from Section 5 before
  submission.
- **References [23]–[26].** Corrected in a later pass: [26]'s title was
  wrong (the real title is "Weight Space Detection of Backdoors in LoRA
  Adapters"), [24]'s venue was incomplete (an LLMSC 2026 workshop paper
  published in the FSE '26 Companion), and [23], [25] and [26] had no
  authors. The corrected details came from search-index results because
  arXiv, dblp and the ACM DL were blocked from this environment — open
  each landing page and confirm the byline before submission.
