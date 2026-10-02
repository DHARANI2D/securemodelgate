# SecureModelGate for HPE Tech Con 2027: Critical Review and Revision Plan

*(Source review that drove this revision. Kept verbatim for traceability
— every fix listed here is either reflected in [`02_paper.md`](02_paper.md)
or explicitly called out as a caveat there.)*

**Bottom line: do not resubmit the Parasparam PDF as it is.** As written,
the paper's headline contribution, behavioral fingerprinting, detects 0%
of backdoors. All of the "Combined" result comes from static analysis.
At least one reference looks fabricated, and the novelty claim is out of
date now that OpenSSF model signing, the Sigstore
model-validation-operator and commercial model scanners exist. You can
still fix this in four days. Your own KL numbers suggest the behavioral
score may separate the classes perfectly at a correctly calibrated
threshold. If so, the problem is how the results are reported, not the
method. Reframe the work as a verifiable model-admission control plane
for HPE Private Cloud AI, and correct the numbers, the references and
the product names before Oct 2.

## TL;DR

- **Fix before submitting:** τ=0.02 cannot "separate all" backdoored
  models when their KL values start at 0.001. The Combined row (DR 100%,
  FPR 15%) matches Static-only exactly, so fingerprinting adds nothing
  but latency. AUC=1.0 does not name its score. The baselines are copied
  from papers with different setups, and STRIP measures a different
  thing. Reference [6] names a non-existent first author, "L. Dex". The
  correct citation is Newman, Meyers and Torres-Arias, CCS 2022.
- **Reframe the novelty:** Signing, ML-BOMs and Kubernetes admission for
  models are all prior art in 2026. OpenSSF Model Signing v1.0 shipped
  on April 4, 2025, per the OpenSSF blog. NVIDIA signs NGC models with
  OMS. The Sigstore model-validation-operator was announced on the
  Sigstore Blog on June 23, 2025 by Red Hat's Trusted Artifact Signer
  team. Palo Alto Networks (Protect AI), Cisco AI Defense and JFrog
  already sell model scanning. What is still defensible: a signed
  behavioral-integrity and lineage attestation that plugs into
  OMS/in-toto and is enforced per risk tier at admission on HPE Private
  Cloud AI.
- **Eligibility and positioning:** HPE does not publish Tech Con
  resubmission or multiple-submission rules, so confirm them with the
  Tech Con program office or your Distinguished Technologist/Fellow
  sponsor before Oct 2. Selection is highly competitive: 722 abstracts
  from 1,355 authors in 2025, and 80 talks in 2026 according to HPE Labs
  director Andrew Wheeler. Make SecureModelGate clearly different from
  WITNESS (pre-deployment admission versus whatever WITNESS covers), or
  submit only the stronger one.

## (a) Tech Con 2027 eligibility and format findings

**What is public:**

- Tech Con is HPE's internal, invitation-only technical conference, led
  by the CTO. Attendees "have to navigate a competitive and rigorous
  application process that begins by submitting an abstract aligned with
  HPE's innovation strategy."
- 2025 (the 23rd edition): HPE received **722 abstracts from 1,355
  authors**, covering areas "like AI and Edge, Hybrid Cloud and
  Security, Sustainability and Observability." It invited **350
  attendees**, 31 of them early-career technologists.
- 2026: Andrew Wheeler (HPE Labs) described **80 presentations "chosen
  as part of a rigorous review,"** with themes spanning "security to
  sovereignty to AI." HPE's recap says "customers are telling us
  security, sovereignty, and AI are among their top technology
  imperatives."
- If 2026 abstract volume was similar to 2025, roughly 1 in 9 abstracts
  became a talk. That is an inference, not a published rate.

**What is not public:** None of the following turned up in public
sources:

- page or abstract word limits
- scoring rubrics
- whether ideas not selected at Parasparam may be resubmitted
- per-author submission caps

These rules live on HPE's intranet. Before Oct 2, check the Tech Con
call-for-papers page or email the program committee, and ask two
questions:

1. Is a previously non-selected internal (Parasparam) idea eligible?
2. Is there a cap on primary-author submissions?

**General guidance from comparable internal conferences** (such as IBM,
Intel and Microsoft internal tech summits):

- Ideas that were not selected are usually eligible if materially
  revised. Reviewers do notice verbatim resubmissions.
- Multiple submissions are usually allowed, but overlapping ones split
  reviewer goodwill.
- Abstracts are typically 300–500 words plus a short "why HPE / why now"
  statement. Judges score novelty, technical depth, business relevance
  and clarity.

Your Parasparam scorecard shows exactly where to invest. Technical
innovation was your weakest area at 12.5/20 (62.5%). Business impact
was strong at 8/10, and clarity at 4/5.

**Timing:** Tech Con 2025 ended in mid-March 2025, and the 2026 recap
appeared in May 2026. Tech Con 2027 will probably take place in spring
2027. That gives you roughly six months between abstract acceptance and
the full talk, so the abstract can commit to larger-scale results you
will produce later, provided you label them as planned.

## (b) Critical errors and inconsistencies to fix

| # | Issue | Verdict | Fix |
|---|---|---|---|
| 1 | "Calibrated τ=0.02 separates all backdoored from clean models" | **False as stated.** The backdoored KL range is 0.001–0.034 nats, so every backdoored model with KL between 0.001 and 0.02 falls under τ and is missed. At τ=0.05, behavioral-only DR is 0%, which fits the backdoored maximum of 0.034. | Report the full clean KL distribution (max and 95th percentile). If the clean max is below 0.001, a threshold of about 0.0005–0.0009 would separate the classes, and your behavioral score may really reach AUC=1.0. Pick τ with a documented calibration procedure, not by eye. |
| 2 | Combined = Static-only (DR 100%, FPR 15%) | **The headline novelty adds no measured value.** Combined appears to be an OR rule. Behavioral flags nothing at τ=0.05, so Combined equals Static and roughly doubles latency (1.26 s → 2.67 s). | Re-run Combined at a properly calibrated τ. Report what each detector catches that the other misses (a Venn/UpSet chart). Show whether behavioral can remove static's 3 false positives (an AND rule or a learned fusion). |
| 3 | AUC=1.0 alongside 3 FPs | **Not strictly contradictory, but under-specified.** AUC is threshold-free, so AUC=1.0 can coexist with FPs at a badly chosen operating point. But then you are deliberately operating at a sub-optimal threshold, and a reviewer will ask why. | Name the score behind the ROC (static KS statistic, KL, or fused). Plot one ROC per score. Report TPR at a fixed FPR (for example 1% and 5%). |
| 4 | "Zero false positives at τ=0.02" vs combined FPR 15% | **Inconsistent framing.** The 3 FPs come from static analysis, so the statement is literally true only for the behavioral detector. | State per-detector FP counts explicitly. |
| 5 | Latency/ratio arithmetic | 312/2.67 = 116.9×, so "117×" is correct arithmetic. 3,600/2.67 = 1,348.3 models/hr, not 1,349 (a minor rounding slip). 0.0156/0.00018 = 86.7×, correct. However, 1.26 + 1.28 = 2.54 s, so **0.13 s of the Combined latency is unexplained** (token signing? I/O?). | Account for the 0.13 s. Say "~1,350 models/hr on Apple M-series MPS, single process." |
| 6 | 117× speed-up vs Neural Cleanse | **Invalid comparison.** 312 s comes from a different paper, model, hardware and dataset. A mean-KL ratio (86.7×) says nothing about separability. Only the tails matter, and the tails overlap your threshold. | Re-run Neural Cleanse on your 40 models on the same hardware, or drop the speed-up claim. Replace the mean ratio with the minimum backdoored KL divided by the maximum clean KL. |
| 7 | Baselines from published papers | **A reviewer will reject this on sight.** STRIP (Gao et al., ACSAC 2019) is a *run-time, input-level* trojan detector, so its reported rates are about inputs, not a population-level model DR/FPR. Neural Cleanse (Wang et al., IEEE S&P 2019) evaluated only a handful of models. | Run both on identical models. BackdoorBench implements both. If you cannot re-run them, remove Table 1's baseline rows and say so. |
| 8 | FPR 15% on 20 clean models | **Statistically weak.** An exact 95% CI for 3/20 is roughly 3%–38%. For DR 20/20 the lower bound is about 83%, which overlaps the copied Neural Cleanse figure of 83.3%. | Report Clopper-Pearson CIs. Grow the clean set, including fine-tuned, pruned and quantized benign variants, since real HuggingFace imports are rarely identical to a reference. |
| 9 | Reference fingerprint F_ref | **Biggest design gap, and the one the evaluators flagged.** For an arbitrary HuggingFace import there is no "clean twin" to compare against. It is unclear what F_ref is for a new architecture or fine-tune. | Define F_ref as the *declared base model* from the MBOM, plus a family-level clean population. This turns fingerprinting into lineage verification (see (d)). |
| 10 | Threat model excludes adaptive attacks because the corpus is secret | **Security through obscurity** (it violates Kerckhoffs's principle). The corpus can leak, and KS or layer-variance statistics can be regularized away during backdoor training. | Keep the holdout but assume the attacker knows the algorithm. Add at least a simple adaptive attack (below). |
| 11 | Stale roadmap (Q1–Q3 2026) | Those dates have passed. | Re-date to Q4 2026–Q2 2027 and mark what was actually done. |
| 12 | "3 runs" | It is unclear whether the 40 models are per run or pooled, and no variance is reported. | Report mean ± std across seeds. |
| 13 | Author name "Dhaaranidharan" | Possible mismatch with your HPE directory name. | Use exactly the spelling in your HPE employee record on every artifact. |

## (c) Reference corrections

- **[3] BadNets.** Correct form: T. Gu, B. Dolan-Gavitt, S. Garg,
  "BadNets: Identifying Vulnerabilities in the Machine Learning Model
  Supply Chain," arXiv:1708.06733, 2017. The title in your PDF is wrong,
  and there is no "et al." (three authors). If you want a 2019
  peer-reviewed version, cite T. Gu, K. Liu, B. Dolan-Gavitt, S. Garg,
  "BadNets: Evaluating Backdooring Attacks on Deep Neural Networks,"
  *IEEE Access*, vol. 7, pp. 47230–47244, 2019.
- **[6] Sigstore.** Correct form: Z. Newman, J. S. Meyers, S.
  Torres-Arias, "Sigstore: Software Signing for Everybody," Proc. ACM
  CCS 2022, DOI 10.1145/3548606.3560596. "L. Dex" does not exist as an
  author. That kind of error looks like LLM-generated citation
  hallucination and seriously damages reviewer trust, so **audit every
  reference against its DOI**.
- **[12] Goldblum.** Correct form: M. Goldblum, D. Tsipras, C. Xie, X.
  Chen, A. Schwarzschild, D. Song, A. Madry, B. Li, T. Goldstein,
  "Dataset Security for Machine Learning: Data Poisoning, Backdoor
  Attacks, and Defenses," *IEEE TPAMI*, vol. 45, no. 2, pp. 1563–1580,
  2023. It was published online in 2022 (DOI
  10.1109/TPAMI.2022.3162397), so "TPAMI 2022" is defensible as the
  online date, but cite 2023 with volume and issue. Your short title is
  also incomplete. Note that this is a *dataset* security survey. For a
  model-import threat model, add NIST AI 100-2e2025 (Vassilev et al.,
  March 2025) and the TrojAI Final Report (arXiv:2602.07152).
- **Baselines, verified:**
  - Neural Cleanse: B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H.
    Zheng, B. Y. Zhao, IEEE S&P 2019, pp. 707–723.
  - STRIP: Y. Gao, C. Xu, D. Wang, S. Chen, D. C. Ranasinghe, S. Nepal,
    ACSAC 2019, pp. 113–125.
- **Add these related-work citations, verified:**
  - ABS: Liu et al., ACM CCS 2019.
  - MNTD: Xu et al., "Detecting AI Trojans Using Meta Neural Analysis,"
    IEEE S&P 2021.
  - MM-BD: Wang, Xiang, Miller, Kesidis, IEEE S&P 2024.
  - UNICORN: Wang et al., ICLR 2023.
  - BTI-DBF: Xu et al., "Towards Reliable and Efficient Backdoor Trigger
    Inversion via Decoupling Benign Features," ICLR 2024.
  - BackdoorBench: Wu et al., NeurIPS 2022 Datasets & Benchmarks;
    extended version in IJCV.
  - HuRef: Zeng et al., NeurIPS 2024.
  - REEF: Zhang et al., ICLR 2025.

## (d) The novelty landscape and a recommended reframing

**Now established prior art. Do not claim these:**

- **Model signing.** OpenSSF Model Signing v1.0 was released on April
  4, 2025 by Google, NVIDIA and HiddenLayer under the OpenSSF AI/ML
  working group, per the OpenSSF launch blog. It supports Sigstore
  keyless signing, certificates, key pairs and PKCS#11. The OMS
  signature is a detached Sigstore bundle with a DSSE-wrapped in-toto
  statement that lists file hashes and optional metadata predicates.
  NVIDIA "signs all models published to the NGC Catalog" with OMS, and
  Google has prototyped OMS on Kaggle. Your RSA-4096 JWT MAT reinvents a
  weaker, non-interoperable version of this.
- **Kubernetes admission for models.** The Sigstore
  model-validation-operator was announced on the Sigstore Blog on June
  23, 2025 by Nina Bongartz of Red Hat's Trusted Artifact Signer team,
  and it is still alpha. It verifies OMS-signed models before workloads
  use them. Kyverno and Sigstore policy-controller already verify
  images at admission, and KServe has an open RFC for OMS verification
  inside its storage initializer. "Among the first systems to integrate
  … with Kubernetes enforcement" is therefore not defensible for
  signature enforcement.
- **AI/ML-BOM.** CycloneDX (1.5+) has ML-BOM. SPDX 3.0 (April 2024)
  added AI and Dataset profiles. The "AI/ML Supply Chain Risks and
  Mitigations" guidance, published on March 4, 2026 by the NSA AI
  Security Center with the cyber agencies of Canada, Australia, the UK,
  New Zealand, Japan, South Korea and Singapore, explicitly recommends
  AI BOMs and cryptographic integrity validation. A bespoke "MBOM" JSON
  should instead be a CycloneDX ML-BOM or SPDX 3.0 AI document.
- **Commercial model scanning:**
  - Palo Alto Networks completed its acquisition of Protect AI on July
    22, 2025. Its FY2025 10-K puts the total purchase consideration at
    $634.5M ($607.4M cash plus $27.1M in replacement awards). The
    technology now sits in Prisma AIRS, covering model scanning and red
    teaming.
  - Cisco AI Defense includes "AI Supply Chain Risk Management," which
    scans model files for backdoors and malicious code.
  - Per Hugging Face, JFrog scans every public model repository on the
    Hub automatically when files are pushed, and JFrog blocks risky
    downloads through JFrog Curation.

  Most of this is serialization and malicious-code scanning (pickle and
  similar), not behavioral backdoor detection. That distinction is your
  opening.
- **Backdoor detection research:** trigger inversion (Neural Cleanse,
  UNICORN, BTI-DBF), neuron stimulation (ABS), maximum-margin statistics
  (MM-BD, data-free), and meta-classifiers (MNTD). The IARPA TrojAI
  program's final report (Reese et al., arXiv:2602.07152, Feb 2026)
  describes a multi-year initiative and covers detection through both
  weight analysis and trigger inversion. TrojAI Round 3 alone has 1,584
  models across 22 architectures. For LLMs there are BAIT (IEEE S&P
  2025), PEFTGuard (S&P 2025, LoRA adapters), ConfGuard (2025) and the
  BackdoorLLM benchmark. Weight-statistic backdoor detection is not new.
  Activation-statistic fingerprints are also close to existing work.
- **Model fingerprinting and provenance:** HuRef (NeurIPS 2024) and REEF
  (ICLR 2025) fingerprint LLM lineage from weights and representations.

**What is still defensibly novel, and how to reframe:**

1. **Behavioral-integrity attestation as a first-class, signed
   predicate.** Existing signing proves *who* published *which bytes*.
   Scanners emit results into dashboards. What remains rare is binding
   "this model passed behavioral test suite X at threshold τ against
   reference R, run by attester Y" into an in-toto predicate. That
   predicate can sit inside the OMS bundle, travel with the model, and
   be re-verified by any admission controller. Build on OMS rather than
   competing with it.
2. **Lineage verification of ML-BOM claims.** Use the static and
   fingerprint machinery to check the ML-BOM's claim that a model is a
   fine-tune of base model B, in the style of HuRef and REEF. This also
   answers the F_ref problem: the reference is the declared base. It
   turns your weakest component (weight statistics as a backdoor
   detector) into a provenance check that fits it well.
3. **Risk-tiered admission policy.** Tier 1: an OMS-signed model from a
   trusted publisher (such as NGC) takes a fast path. Tier 2: a signed
   but unverified-lineage model gets a fingerprint and lineage check.
   Tier 3: an unsigned HuggingFace import gets a deep scan (pickle scan,
   then MM-BD or trigger inversion, then behavioral). Your 1–3 s static
   and behavioral stage then becomes a legitimate *triage* layer ahead
   of expensive detectors rather than a replacement for them. That is a
   credible systems contribution.
4. **HPE-specific integration.** Enforcement lives in HPE Private Cloud
   AI and HPE AI Essentials, and attestation telemetry flows to HPE
   OpsRamp.

New claim to use: "To our knowledge, SecureModelGate is the first system
to issue OMS-compatible behavioral-integrity and lineage attestations
and enforce them per risk tier at Kubernetes admission on an enterprise
private AI cloud."

*(This is the earlier framing. `02_paper.md`'s final claim narrows it
further — from "behavioral-integrity and lineage" to just "behavioral-
lineage" — after this review's own point 1–3 above and the numbers issue
in (b) made clear the behavioral stage needed to be anchored to the
declared base, not run as a second, separate integrity check.)*

## (e) Product and regulatory updates

**Products (as of September 2026):**

- **HPE Ezmeral Container Platform is a deprecated name.** HPE renamed
  it **HPE Ezmeral Runtime Enterprise** in 2021, and a third-party
  lifecycle database lists 5.7.x as the latest line. Do not headline it
  for 2027.
- **HPE Private Cloud AI** is the right primary target. Its software
  layer, **HPE AI Essentials Software** (docs at release 2026.07.1),
  ships Kubeflow, Ray and Spark Operator on Kubernetes. HPE markets
  Private Cloud AI under "Enterprise-Grade Data Security," and your
  evaluators already named it as the differentiator.
- **HPE Private Cloud (fourth generation, announced May 12, 2026)** adds
  Kubernetes management alongside VMs, with unified VM and container
  management GA scheduled for Q3 2026, under HPE Morpheus. This is your
  secondary target for the general Kubernetes admission path.
- **"GreenLake AI" is not a current product name.** Use **GreenLake**
  (HPE now brands it without the "HPE" prefix) and **GreenLake
  Intelligence**, the agentic AI framework launched at Discover Las
  Vegas on June 17, 2026.
- **HPE OpsRamp Software** is active. It is now part of **HPE CloudOps
  Software** (with Morpheus and Zerto), and "HPE OpsRamp Operations
  Copilot within GreenLake Intelligence" is available. Position OpsRamp
  as the sink for attestation events and denied-admission alerts. The
  ServiceNow integration is announced for rollout over 2026–2027, not
  yet delivered.
- **HPC:** name HPE Cray Supercomputing if you keep that path.

**Regulatory:**

- **EU AI Act.** Art. 13 covers transparency and instructions for use
  owed to deployers of high-risk systems. It is only tangential to an
  MBOM. Better anchors:
  - Art. 11 with Annex IV (technical documentation, where lineage and
    components belong).
  - Art. 15 (accuracy, robustness and cybersecurity). Art. 15(5)
    explicitly names data poisoning and model poisoning, which is your
    core threat.
  - Art. 10 (data governance) for dataset lineage.
  - Art. 53 with Annex XI (GPAI provider documentation), which has
    applied since 2 Aug 2025 and matters for imported foundation
    models.
  - **Timeline:** the Digital Omnibus (Regulation (EU) 2026/1744, in
    force 27 July 2026) moved Annex III high-risk obligations to **2
    Dec 2027** and Annex I obligations to **2 Aug 2028**. Frame
    compliance as "ahead of the December 2027 deadline," not as a
    current requirement.
- **EO 14028** remains in force. The June 6, 2025 Trump order struck EO
  14144's enhanced software-attestation provisions but left EO 14028's
  CISA self-attestation regime (OMB M-22-18/M-23-16) intact. It also
  ordered an update to NIST SP 800-218 (SSDF). Cite EO 14028 as the
  software supply-chain attestation basis, not as an AI mandate.
- **NIST:** cite AI RMF 1.0 (NIST AI 100-1) together with the
  Generative AI Profile (NIST AI 600-1), and NIST AI 100-2e2025 for the
  attack taxonomy. The 2025 edition added explicit AI supply-chain
  content.
- **CISA/NSA:**
  - "AI Data Security" CSI (May 22, 2025), which recommends digital
    signatures, data provenance tracking and trusted infrastructure.
  - The "AI/ML Supply Chain Risks and Mitigations" guidance of March 4,
    2026 (AI BOM, cryptographic integrity validation), issued by the
    NSA AI Security Center with the cyber agencies of Canada, Australia,
    the UK, New Zealand, Japan, South Korea and Singapore. This is your
    strongest single policy citation.

## (f) Revision checklist

**Before Oct 2 (four days; in priority order):**

1. **Recompute the behavioral results.** Dump the per-model KL values.
   Choose τ by leave-one-out or a held-out split (for example, the
   maximum clean KL plus a margin, or a target FPR). Re-run Combined
   with an explicit fusion rule. Rewrite Table 1 with per-detector
   TP/FP, 95% CIs and the ROC score source. *(About 4 hours.)*
2. **Remove or replace the copied baselines.** Use BackdoorBench's
   Neural Cleanse and STRIP implementations on your 40 models on the
   same machine. Forty Neural Cleanse runs at about 5 min each is
   roughly 3–4 hours; run it overnight. If that fails, delete the
   baseline rows and the 117× claim.
3. **Add two or three BackdoorBench attacks** (such as WaNet, SIG,
   Input-aware) and about 10 benign variants (fine-tuned, pruned, INT8)
   to stress FPR. *(About 1 day on MPS; doable in parallel.)*
4. **Fix every reference** against DOIs, especially [3], [6] and [12].
   Add NIST AI 100-2e2025, TrojAI, OMS, CycloneDX/SPDX 3.0, and one or
   two LLM detectors.
5. **Rewrite the novelty section** around the three defensible
   contributions in (d), and cite OMS, the model-validation-operator,
   Prisma AIRS, Cisco AI Defense and JFrog as related work.
6. **Swap the MAT for an OMS-bundle-plus-in-toto behavioral predicate**
   (a design change on paper; a prototype can follow). Map the MBOM to a
   CycloneDX 1.6 ML-BOM.
7. **Update product names** (Private Cloud AI / AI Essentials, HPE
   Private Cloud, GreenLake Intelligence, HPE OpsRamp) and the
   regulatory anchors (Art. 11/Annex IV, Art. 15, Art. 53; Omnibus
   dates; the 2026 NSA supply-chain guidance).
8. **Add a one-paragraph adaptive-attack statement,** plus even a toy
   experiment: train a backdoor with a loss term penalizing KS distance
   to the clean weight distribution and report whether static detection
   survives.
9. **Re-date the roadmap,** fix the author-name spelling, and confirm
   the resubmission and multiple-submission rules with the Tech Con
   committee.

**Longer term (for the talk and a full paper):**

- TrojAI rounds (image and NLP) for evaluation at scale across
  architectures.
- Transformer and LLM extension: BackdoorLLM benchmark, BAIT and
  PEFTGuard as baselines, GGUF/safetensors support, and LoRA-adapter
  scanning.
- A real adaptive-attacker study (a fingerprint-aware backdoor) and a
  formal threat model aligned with NIST AI 100-2e2025.
- Lineage verification against HuRef and REEF on real HuggingFace
  fine-tune families.
- A working integration with the Sigstore model-validation-operator or
  KServe on HPE Private Cloud AI, with OpsRamp dashboards.
- Operational metrics: false-block rate on real internal model imports,
  and triage savings versus running a deep scanner on everything.

**Differentiation from WITNESS:** no details of WITNESS were available
to this review, so the advice here is conditional:

- **If WITNESS is also attestation, provenance or monitoring,** split
  cleanly by lifecycle stage. SecureModelGate is *pre-deployment
  admission* (supply chain → cluster). WITNESS should own
  *runtime/in-operation* evidence. Cross-reference them as
  complementary.
- **If they overlap by more than about 30%,** submit the stronger one
  or merge them. Two thin abstracts from one author on the same theme
  compete with each other for the same reviewers.
- **For SecureModelGate specifically,** lead with the business hook the
  evaluators praised: CI/CD throughput, compliance evidence, and
  differentiation for Private Cloud AI. Then give one crisp technical
  novelty (signed behavioral and lineage attestation) with honest
  numbers.

## (g) Earlier draft abstract (superseded)

The abstract this review originally proposed is superseded by
[`01_abstract.md`](01_abstract.md), which reflects the further-narrowed
claim (Section 3 of `02_paper.md`) and the Section (b)/(d) fixes above.
It is kept here only as the intermediate step between the original
Parasparam PDF and the final revision:

> **SecureModelGate: Verifiable Behavioral and Lineage Attestation for
> AI Model Admission on HPE Private Cloud AI**
>
> Enterprises increasingly deploy open-weight models pulled from Hugging
> Face and GitHub, yet today's controls verify only *who* signed a
> model's bytes (OpenSSF Model Signing) or scan for malicious
> serialization. Neither tells a platform whether the model behaves like
> the base model it claims to derive from. SecureModelGate is a
> model-admission control plane for HPE Private Cloud AI that issues and
> enforces three evidence types: (1) an OMS-compatible signature; (2) a
> CycloneDX ML-BOM recording declared lineage and dependencies; (3) a
> signed in-toto *behavioral-integrity and lineage* predicate, produced
> by a lightweight weight-statistics and activation-fingerprint test
> against the declared base model. A Kubernetes admission policy applies
> risk tiers. Trusted-publisher models take a fast path,
> unverified-lineage models receive fingerprint checks, and unsigned
> imports are routed to deep backdoor scanning, so expensive detectors
> run only where needed. On [N] CIFAR-10 models covering [k] backdoor
> attacks and benign fine-tuned, pruned and quantized variants, the
> triage stage achieves [TPR]% at [FPR]% FPR (95% CI reported) in about
> 2.7 s per model on commodity hardware. Neural Cleanse and STRIP were
> re-run on identical models for comparison. We describe threshold
> calibration, limits under adaptive attacks, and the roadmap to
> transformer/LLM models (safetensors, GGUF, LoRA) and HPE OpsRamp
> telemetry. SecureModelGate gives HPE customers audit-ready evidence
> aligned with EU AI Act Articles 11 and 15 ahead of the December 2027
> high-risk deadline, EO 14028 supply-chain attestation, and the 2026
> multinational AI/ML supply-chain guidance.

## Caveats (from this review)

- Tech Con rules (word limits, resubmission, submission caps) are
  internal and could not be verified publicly. The advice in (a) is
  based on analogous internal conferences.
- The Ezmeral Runtime 5.7.x version detail comes from a third-party
  lifecycle site. Check HPE's support portal before citing it.
- This review could not check whether the specific Neural Cleanse and
  STRIP figures in the original Table 1 appear verbatim in the original
  papers. Regardless, they are not comparable to the paper's setup.
- The acceptance ratio (about 1 in 9) is an inference from two different
  years' figures, not an HPE statistic.
