/**
 * Regenerates SecureModelGate_TechCon2027.docx from this revision's
 * markdown source (01_abstract.md / 02_paper.md), formatted to match
 * the HPE Tech Con / Parasparam submission convention (title block,
 * numbered sections, in-toto-style figures, numbered references).
 *
 * Content is authored directly in this script rather than parsed from
 * markdown, so after editing 02_paper.md, mirror the change here too --
 * this is not an automatic converter.
 *
 * Usage (from this directory):
 *   npm install docx   # one-time; not committed (see .gitignore)
 *   node build_docx.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageOrientation, LevelFormat, UnderlineType,
} = require("docx");

const REPO = __dirname;
const FIG = (p) => fs.readFileSync(path.join(REPO, p));

// ---------- style helpers ----------
const FONT = "Calibri";
const COLOR_HEAD = "1F3864";
const COLOR_RULE = "8EA9C1";

function run(text, opts = {}) {
  return new TextRun({ text, font: FONT, ...opts });
}
function seg(parts, opts = {}) {
  // parts: array of [text, {bold,italic}] or plain strings
  return parts.map((p) =>
    typeof p === "string" ? run(p, opts) : run(p[0], { ...opts, ...(p[1] || {}) })
  );
}
function para(children, opts = {}) {
  const runs = Array.isArray(children) ? children : [run(children)];
  return new Paragraph({ children: runs, spacing: { after: 160 }, ...opts });
}
function bodyPara(parts, opts = {}) {
  return para(seg(parts, { size: 21 }), { alignment: AlignmentType.JUSTIFIED, ...opts });
}
function bullet(parts, opts = {}) {
  return para(seg(parts, { size: 21 }), {
    bullet: { level: 0 }, spacing: { after: 100 }, alignment: AlignmentType.JUSTIFIED, ...opts,
  });
}
function subBullet(parts, opts = {}) {
  return para(seg(parts, { size: 21 }), {
    bullet: { level: 1 }, spacing: { after: 100 }, alignment: AlignmentType.JUSTIFIED, ...opts,
  });
}
function h1(text) {
  return new Paragraph({
    text, heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    border: { bottom: { color: COLOR_RULE, space: 4, style: BorderStyle.SINGLE, size: 6 } },
  });
}
function h2(text) {
  return new Paragraph({
    children: [run(text, { bold: true, size: 24, color: COLOR_HEAD })],
    spacing: { before: 240, after: 120 },
  });
}
function h3(parts) {
  return new Paragraph({
    children: seg(parts, { bold: true, size: 21 }),
    spacing: { before: 160, after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [run(text, { italic: true, size: 18, color: "595959" })],
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
  });
}
function figure(imgPath, w, h, capText) {
  const scale = Math.min(1, 380 / w);
  return [
    new Paragraph({
      children: [new ImageRun({ data: FIG(imgPath), type: "png", transformation: { width: Math.round(w * scale), height: Math.round(h * scale) } })],
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
    }),
    caption(capText),
  ];
}

function cell(text, opts = {}) {
  const { bold = false, shade = null, width = null, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      children: [run(text, { bold, size: 18 })],
    })],
  });
}
function dataTable(headers, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: headers.map((htext, i) => cell(htext, { bold: true, shade: "D9E2F3", width: colWidths[i] })),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(c, { width: colWidths[i] })),
      })),
    ],
  });
}

// ---------- content ----------

const children = [];

// Title block
children.push(new Paragraph({
  children: [run("SecureModelGate: Signed Behavioral-Lineage Attestation for Admitting Third-Party AI Models", { bold: true, size: 34, color: COLOR_HEAD })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 120 },
}));
children.push(new Paragraph({
  children: [run("Dharanidharan Senthilkumar", { size: 22, bold: true })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 20 },
}));
children.push(new Paragraph({
  children: [run("OLAA, CDRM – Cyber Defense Center, HPE, Bengaluru, India", { size: 20, italic: true })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 20 },
}));
children.push(new Paragraph({
  children: [run("dharanidharan.senthilkumar@hpe.com", { size: 20 })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 60 },
}));
children.push(new Paragraph({
  children: [run("Tech Con 2027 — Revised Submission  ·  Revision date: 2026-09-29", { size: 18, italic: true, color: "595959" })],
  alignment: AlignmentType.CENTER,
  spacing: { after: 320 },
}));

// Abstract
children.push(h2("Abstract"));
children.push(bodyPara([
  "Enterprises increasingly deploy models derived from a trusted base rather than published by a trusted vendor: fine-tunes, LoRA adapters, pruned and quantized variants. Model Atlas (Horwitz et al., NeurIPS 2025) found that more than 60% of Hub models carry no documented parentage; where a ",
  ["base_model", { italic: true }],
  " field exists, it is optional and unverified. OpenSSF Model Signing (OMS) and Kubernetes admission tools prove a model's bytes are unchanged since signing — they say nothing about whether a derivative really descends from the base it claims, or how far its behavior has moved.",
]));
children.push(bodyPara([
  "SecureModelGate closes this gap with a signed behavioral-lineage attestation. At admission, the gate reads the declared ancestor from the model's CycloneDX ML-BOM, verifies that ancestor's OMS signature, and uses it as the reference fingerprint for two calibrated tests: weight-space lineage agreement with the base, and layer-wise representation similarity (linear CKA) plus output divergence on a probe set drawn per admission. Thresholds come from split-conformal calibration on benign derivatives, giving a stated false-block bound instead of a hand-picked cut-off. The verdict, scores and calibration evidence ship as a signed in-toto predicate and drive risk-tiered admission: trusted publisher (fast path), signed base with unverified derivative (lineage + envelope check), unsigned (deep scan).",
]));
children.push(bodyPara([
"Our earlier prototype caught all 20 of 40 CIFAR-10 ResNet-18 backdoored models with a static weight check, but also blocked 3 of 20 clean models (95% CI 3–38%), and its behavioral stage added nothing. The revised evaluation re-runs Neural Cleanse, STRIP and MM-BD on identical BackdoorBench models and adds an adaptive attacker; that run is still pending GPU time (Section 4.4). A smaller synthetic-corpus pilot shows the machinery working end to end: 29/30 non-adaptive backdoors escalated, 0.997 envelope AUROC, 98.6% lineage accuracy and 0/20 benign false blocks. On real weights, lineage accepts 10/10 bert-base-uncased descendants and rejects 5/5 held-out MultiBERTs and a layer splice. The pilot also shows where the gate breaks. One backdoor is admitted. Benign fine-tunes deeper than the calibration set are escalated. An attacker who trains against the envelope score, or who also pulls an unrelated model's weights toward the public base, is admitted. Adversarial testing found and closed a splice-and-reshape bypass, a fail-open path and a post-admission tampering gap. The envelope test is triage and the lineage test is not forgery-proof. What the gate guarantees is a signed, recomputable, fail-closed verdict bound to the exact model bytes. The talk demonstrates the gate for HPE Private Cloud AI, with a public LLM/LoRA example, and proposes the predicate as an open format.",
]));

// Section 1
children.push(h1("1. Problem"));
children.push(bodyPara([
  "Model supply chains are now mostly derivative. A trusted publisher releases a base model, and third parties fine-tune, adapt (LoRA), prune, quantize (INT8/GGUF) and re-publish it. Two recent developments make the base trustworthy:",
]));
children.push(bullet([["Signed bases. ", { bold: true }], "OMS v1.0 (April 2025) gives publishers a detached Sigstore-bundle signature over every model file [29], and according to NVIDIA's Developer Blog, NVIDIA has signed all NVIDIA-published NGC Catalog models with OMS since March 2025, making NGC the first major model hub to do so [36]."]));
children.push(bullet([["Enforceable signatures. ", { bold: true }], "Admission tools such as the Sigstore model-validation-operator [30] and the proposed KServe storage-initializer verification [31] can enforce those signatures."]));
children.push(bodyPara([
  "A signature, however, covers only the exact bytes the publisher signed. A derivative is different bytes, so it is either unsigned or signed by an unknown party. Its claimed parentage sits in a self-reported, optional ", ["base_model", { italic: true }], " field, and Model Atlas (Horwitz et al. [35]) measured that more than 60% of Hub models carry no documented parentage.",
]));
children.push(bodyPara(["That leaves admission teams with two unanswered questions:"]));
children.push(bullet([["Q1 — lineage: ", { bold: true }], "Is this model really a descendant of the signed base it claims?"]));
children.push(bullet([["Q2 — behavioral envelope: ", { bold: true }], "Has it drifted from that base in ways benign derivatives do not?"]));
children.push(bodyPara([
  "Existing controls cover neighbouring risks. Pickle and code-execution scanners address serialization attacks. PickleBall shows 44.9% of popular models still ship pickle [27], and Zhao et al.'s MalHug scanner found 91 malicious models among more than 705K monitored [28]. JFrog says its scanner eliminates more than 96% of the false positives other scanners produce on current Hugging Face models [37]. Deep backdoor detectors (Neural Cleanse [8], MM-BD [10], BAIT [17]) are too slow or too specialised to run on every import. Our Parasparam 2026 prototype also showed the cost of lacking a reference: the behavioral stage compared models against an undefined \"clean reference\" and contributed nothing.",
]));

// Section 2
children.push(h1("2. Threat Model"));
children.push(bodyPara(["We follow the general poisoning/backdoor taxonomy of Goldblum et al. [3] and NIST AI 100-2e2025 [4], narrowed to the four adversaries below."]));
children.push(bullet([["A1 — lineage forgery. ", { bold: true }], "The adversary publishes a model claiming a popular OMS-signed base, e.g. to inherit its trust tier, but the model is independently trained or substituted. The gate's goal is to reject the lineage claim. Two distinct failure modes both defeat this adversary but route differently (Section 3.5), deliberately: a candidate with no verifiable declared base at all cannot even have S_w computed, so it drops to Tier 2 deep scan as the only applicable check; a candidate with a verified base whose measured S_w fails the lineage threshold has already been scored and positively found not to descend from what it claims, so it is blocked directly at Tier 1 — deep scan checks for malicious payloads, not ancestry, so routing a known-false lineage claim through it instead of blocking would let a forged-provenance model reach admission on a clean scan result, defeating Q1 entirely."]));
children.push(bullet([["A2 — poisoned true derivative. ", { bold: true }], "The adversary fine-tunes the genuine base with a backdoor (BadNets [1], Blended [42], WaNet [21], SIG [41], Input-aware [21]; LLM trigger-response backdoors). The goal is to escalate it to deep scanning with a stated benign false-block rate. We treat this as calibrated triage, not a detection guarantee."]));
children.push(bullet([["A3 — serialization/code payloads ", { bold: true }], "(pickle, trust_remote_code). These are routed to existing scanners, and the gate records their results in the attestation."]));
children.push(bullet([["A4 — adaptive adversary. ", { bold: true }], "The adversary knows the full algorithm, features, public probe pool and calibration procedure (Kerckhoffs' principle, with no reliance on obscurity). An earlier draft also withheld the per-admission probe seed; as implemented that seed is not a secret (Section 3.4), so this model assumes the attacker may know it. The adversary may add fingerprint-matching regularisers during backdoor training. A synthetic-corpus pilot of exactly this attack shows detection collapsing from 100% to 20% by penalty weight λ≈1, with attack success staying ≥99.8% throughout (Supplementary Pilot B). That attacker trained against probe batches resampled from the public probe distribution, not the fixed evaluation pool; on this synthetic corpus that distinction is close to moot, and Experiment E7 tests the two separately on real image data."]));
children.push(bullet([["Trusted: ", { bold: true }], "the base publisher's OMS signing identity, the gate's signing identity, and the cluster admission controller."]));
children.push(bullet([["What the gate does not stop (stated up front). ", { bold: true }], "Measured on the synthetic corpus (Supplementary Pilot E, Q7–Q8). An adversary who backdoors a genuine fine-tune of the real base (A2) and trains against the envelope score (A4) is admitted (4/4 at λ ≥ 1, ASR 100%); lineage is true. An adaptive lineage forger is admitted too: an unrelated, backdoored model trained with a penalty pulling its weights toward the public base passes lineage 4/4, and with the envelope penalty added as well, 4/4 are admitted with 100% attack success. S_w measures weight-space proximity to the declared base, not provenance, and the base is public by construction. A backdoor in a declared, replaced head is escalated, not admitted, but only because every head replacement escalates. Against these adversaries the gate contributes a signed, attributable audit record (who published what, claiming which base), not detection. What it does stop: non-adaptive forged lineage, including the splice and reshape variants (0/285 undeclared multi-matrix splices pass); non-adaptive backdoors at triage (29/30; one is admitted); tampered, replayed or re-signed attestations, and models changed after admission; and unsigned or unknown models, which are routed to deep scan."]));
children.push(bullet([["Out of scope: ", { bold: true }], "compromised publisher keys, backdoors already present in the signed base (inherited, so lineage-consistent), distillation (not a weight descendant; flagged \"lineage unverified\" and sent to deep scan), and all runtime or post-deployment monitoring. Runtime monitoring is the scope of the separate WITNESS proposal."]));

// Section 3
children.push(h1("3. Contribution / Solution"));
children.push(bodyPara([
  ["Innovation claim (single): ", { bold: true }],
  ["SecureModelGate is, to our knowledge, the first admission control that issues a signed, calibrated behavioral-lineage attestation. It proves that a derivative model is a weight-level descendant of its declared, OMS-signed base, and that the model's behavioral deviation from that base falls within a conformally calibrated benign-derivative envelope. It binds both results, with their evidence, to the model's digests in an in-toto predicate that Kubernetes admission policy enforces.", { italic: true }],
  " Signing, lineage scoring and backdoor detection each exist separately. The contribution is making lineage a verifiable, policy-enforceable supply-chain claim, with a stated error bound.",
]));

children.push(new Paragraph({
  shading: { type: ShadingType.CLEAR, fill: "F2F5FA" },
  border: { left: { color: COLOR_RULE, size: 18, style: BorderStyle.SINGLE, space: 8 } },
  spacing: { before: 120, after: 240 },
  indent: { left: 200, right: 200 },
  children: seg([
    ["Implementation. ", { bold: true }],
"The securemodelgate Python package (this repository; pip install -e \".[dev]\", securemodelgate demo) implements every piece below, wired end to end into one admission pipeline, and is tested by 129 passing tests against small synthetic torch models, not the paper's PreAct-ResNet-18/CIFAR-10 BackdoorBench evaluation (E1–E9, still to run). On that synthetic corpus, re-run with the current code (Supplementary Pilot A): 29/30 non-adaptive backdoors detected (the miss, a patch trigger with ASR 100%, is admitted), envelope AUROC 0.997, 98.6% lineage accuracy (the one miss is a genuine derivative blocked, not an unrelated model admitted), and 0/20 benign false blocks at α = 0.05 (95% CI 0–16.8%). On real weights (bert-base-uncased against 25 MultiBERTs; lineage only), the lineage rule accepts 10/10 descendants and rejects 5/5 held-out unrelated models and a 7-of-12-block splice. An earlier version of this pilot reported a 10% false-block rate on a leaky test set: 13 of the 20 \"held-out\" benign models were byte-identical to calibration models. Every number comes from one per-model file (pilot_results/evaluation.json). The pilots found real problems as well as confirming the design: a bug in S_w's parameter matching; an overclaimed permutation-invariance property, now fully recovered on sequential networks (Supplementary Pilot C); a full pipeline bypass (splicing layers to game S_w's median, combined with adaptive training against the envelope score, reached 100% attack success at Tier 1 ADMIT; there were two routes to it, and the first fix closed only one); a reference-digest binding gap, a silent NaN-corruption path, two unbound calibration sets and a dead gate_version parameter; and a spectral component measured to hurt separation. Answering a 44-question developer checklist from the code (Supplementary Pilot E) found more: the leaky benign test set above; a fail-open path (a NaN JS divergence was silently ignored); BatchNorm statistics outside the attestation's digest, so post-admission edits went unnoticed; an unauthenticated DSSE payload type; a verifier that never re-derived thresholds or verdicts; and an attacker-controlled matching cost. All are fixed (Supplementary Pilots D and E). What is not fixed is stated in Section 2.",
  ], { size: 19, italic: false }),
}));

children.push(new Paragraph({ children: seg([["3.1  Reference resolution", { bold: true, size: 21 }], ["  (fixes the undefined F", {}], ["ref", { subScript: true }], [")."]], {size:21}), spacing: { before: 160, after: 80 } }));
children.push(bodyPara(["1. The candidate model m must ship a CycloneDX (1.6/1.7) ML-BOM whose machine-learning-model component lists the base b under pedigree.ancestors, identified by purl (e.g., pkg:huggingface/...) and OMS bundle digest. The SPDX 3.0 equivalent is an AI-profile package with a descendantOf relationship."]));
children.push(bodyPara(["2. The gate verifies b's OMS signature against an allow-list of publisher identities."]));
children.push(bodyPara(["3. The fingerprint of the verified b is F", ["ref", {subScript:true}], "."]));
children.push(bodyPara(["4. If no signed ancestor is declared, the model drops to the deep-scan tier."]));

children.push(h3(["3.2  Two-part fingerprint with justification."]));
children.push(bullet([["Weight lineage score S", {}], ["w", {subScript:true}], [" (answers Q1). ", { bold: true }], "For each weight matrix l of the verified base (conv kernels, linear weights; 1D parameters such as biases and BatchNorm affine weights are excluded, because framework-default initialisation makes them near-identical across unrelated models), compute the mean cosine similarity between optimally matched output channels of W_l(m) and W_l(b) (Hungarian assignment; HuRef-style [15] in spirit). Layers are walked in order, and each layer's channel correspondence is carried into the next layer's input columns before that layer is matched (sequential alignment). Lineage verifies only if (1) the median over layers clears its conformal threshold, (2) every scored layer clears a per-layer floor calibrated on unrelated models' pooled per-layer scores, and (3) at most max_exempt (default 1) weight matrices go unscored. Every matrix is either scored or recorded as unscored, with a reason: declared replaced in the ML-BOM, missing, added, or incomparable. Each condition closes a demonstrated attack. Median only: a candidate that verbatim-copies just over half the layers and swaps the rest for a foreign block cleared it (S_w 0.76–0.81 against a threshold of about 0.59, under the earlier 50/50 spectral blend). Median plus floor, without a cap: the earlier scorer skipped any layer whose shape differed from the base, so widening each foreign block by one channel left it unscored; with adaptive training against the envelope score added, that got a 100%-success backdoor admitted at Tier 1 on 2 of 3 seeds. Both routes are now blocked (Supplementary Pilot D). The floor has no formal error guarantee: per-layer scores from one model are correlated, so pooling them breaks exchangeability. It only ever adds rejections, so false accepts stay bounded by the median test; its false-reject rate is measured, not guaranteed."]));
children.push(subBullet(["Why no spectra (a change from earlier drafts): the score was originally a 50/50 blend of channel-matched cosine and singular-value-spectrum correlation. An ablation dropped the spectra. Same-architecture models trained on the same data have nearly identical spectra, so spectra alone cannot separate at all (worst genuine and best unrelated per-layer minimum both 0.986; 4 of 13 genuine derivatives rejected), and blending halves the margin cosine gives alone (0.917 vs 0.573, against 0.848 vs 0.148) and lowers Pilot A lineage accuracy (65/70 vs 69/70). The default is now cosine-only. That choice was made on the synthetic corpus, Pilot A's own population included, so the real-model run reports both settings."]));
children.push(subBullet(["Why channel-matched, aligned cosine: permuting a conv/linear layer's output channels (and correspondingly its BatchNorm parameters and the next layer's input channels) is an exact, function-preserving symmetry of any ReLU/BatchNorm network. Plain cosine of the flattened tensor collapsed a genuine derivative's per-layer score from ~1.0 to ~0.5 under a single layer's permutation. Per-layer channel matching fixed a single layer but recovered a full-network cascading permutation only to 0.60, because each layer was matched against inputs its predecessor had scrambled; the per-layer floor then turned that into a false rejection of a benign model. Sequential alignment, a Git Re-Basin-style [43] weight matching, removes the cause: the fully permuted derivative now scores 1.00 on every layer and is admitted. Scope: chaining follows parameter-registration order, which is data-flow order only for sequential networks; residual and transformer blocks need a model-specific alignment map, and attention-head permutation is not handled. Cost: alignment also makes unrelated models' classifier layers look more alike (null maximum 0.30 → 0.61 with cosine-only scoring, the Re-Basin effect itself), which alone sets the pooled floor (0.49 on the scope-check calibration, 0.508 on Pilot A's); the margin to the weakest genuine derivative is 0.848 − 0.49 ≈ 0.36. On real BERT weights, where no small head is scored, the margin is wider, not narrower: 0.930 − 0.128 ≈ 0.80 (Section 4). \"Resists rotation obfuscation\" is dropped: a general rotation is not a function-preserving symmetry of a ReLU network (Supplementary Pilot C)."]));
children.push(subBullet(["Why rectangular assignment: real structured pruning changes a layer's output-channel count. An earlier scoring function fell back to flattened cosine when counts differed and mis-scored a scattered channel drop (~0.13 for a layer keeping 6 of 8 channels verbatim). That fix never reached the gate, though: the torch adapter skipped any shape-mismatched layer before scoring it, so at admission a structurally pruned layer went silently unscored, which was the loophole behind the reshaping attack above, not a false rejection. Now every layer is matched rectangularly: extra channels that no base channel explains count as zero similarity, so foreign capacity bolted onto a copied layer lowers its score, and after pruning the next layer compares only the base input columns its surviving channels map to."]));
children.push(subBullet(["What permutation does not touch: it has no effect on D_b at all. Channel permutation is exactly function-preserving, so a candidate's activations and logits, and therefore its CKA/JS envelope score, are identical whether or not its weights are permuted (measured: a backdoored derivative's envelope score changed by <4e-9 under a full cascading permutation). An attacker who permutes a backdoored model gains nothing on the behavioral side. Nor does permutation let an unrelated model forge ancestry: a fully permuted independent model is still blocked under sequential alignment (S_w 0.173, per-layer minimum 0.139)."]));
children.push(subBullet(["Why direction: HuRef [15] shows that base-model parameter directions barely move under SFT/RLHF, while independently trained models share no such alignment."]));
children.push(subBullet(["What S_w does not establish. Provenance against an adaptive attacker: training an unrelated model toward the public base's weights passes the lineage test (Section 2). Training lineage rather than shared initialisation: models sharing the base's random init but trained separately score S_w ≈ 0.70, and two of three pass. Freedom from foreign channels: about a quarter of every layer's channels can be foreign and still pass. The lineage test separates honest derivatives from honest unrelated models; it is not a forgery-proof check (Supplementary Pilot E, Q4, Q7, Q9)."]));
children.push(bullet([["Behavioral delta D", {}], ["b", {subScript:true}], [" (answers Q2). ", { bold: true }], "On a probe set P of k inputs, sampled for each admission from a public pool with seed s (commitment H(s) recorded first), compute per-layer linear CKA between m's and b's activations (minimum over layers), and the Jensen–Shannon divergence between the output distributions — replacing the original unanchored KL with the same idea, anchored to the true base."]));
children.push(subBullet(["Why CKA: linear CKA [19] is invariant to orthogonal transforms and isotropic scaling of representations, so benign re-scaling and quantization noise should move it little. A backdoor must add a new feature-to-target mapping, which we hypothesise lowers late-layer CKA and raises output divergence on trigger-bearing directions more than benign fine-tuning does. On the synthetic-corpus pilot this mostly holds against a non-adaptive attacker (AUROC 0.997, 29/30 detected, one admitted; Supplementary Pilot A) but breaks down against the A4 adaptive attacker who optimizes directly against it (detection 100%→20% by λ≈1; Supplementary Pilot B). Both outcomes are pilot-scale, not Experiment E1/E7-scale, but both are measured, not asserted."]));
children.push(subBullet(["We also record the stable rank of ΔW_l = W_l(m) − W_l(b) in the final layers as a diagnostic, not a decision feature."]));

children.push(h3(["3.3  Principled thresholds (replaces τ)."]));
children.push(bodyPara(["Two split-conformal [20] tests with two calibration populations:"]));
children.push(bullet([["Lineage test. ", { bold: true }], "Null hypothesis: m is unrelated to b. Calibrate S_w on independently trained models of the same architecture, and verify lineage if S_w exceeds their (1−α_L) conformal quantile."]));
children.push(bullet([["Envelope test. ", { bold: true }], "Null hypothesis: m is a benign derivative. Calibrate on n benign derivatives (fine-tuned, pruned, INT8), set τ to the ⌈(n+1)(1−α)⌉-th smallest calibration score, and escalate if the score exceeds τ. Under exchangeability, P(benign derivative escalated) ≤ α."]));
children.push(bullet([["Consequence 1: ", { bold: true }], "with n = 19 and α = 0.05, τ equals the maximum calibration score — the earlier review's advice to \"calibrate against the maximum clean score\" is therefore this rule, now with a stated guarantee."]));
children.push(bullet([["Consequence 2: ", { bold: true }], "a 1% false-block claim needs n ≥ 99 calibration models. We report α together with the n it was calibrated on."]));
children.push(bullet([["What the guarantee covers. ", { bold: true }], "\"≤ α\" holds only for candidates exchangeable with the calibration set. The calibration derivatives are ones we made ourselves, and real third-party fine-tunes are not exchangeable with them, so the claim is \"calibrated on N self-made derivatives, α = …\"; the real-world false-block rate is a separately measured number. The synthetic pilot shows how narrow the coverage is. Held-out derivatives made like the calibration ones: 0/20 escalated. Benign fine-tunes 5× longer than the calibration recipe: 7/7 escalated. Calibrating on mixed depths removes those false blocks but detects 27/30 backdoors instead of 29/30. The per-layer floor has no such guarantee at all: it pools per-layer scores whose distributions differ (unrelated classifier layers score up to 0.61, conv layers ≤ 0.38)."]));

children.push(h3(["3.4  Attestation."]));
children.push(bodyPara(["The attestation is an in-toto Statement v1 [32] whose subject lists the same file digests as the model's OMS manifest."]));
children.push(bullet([["predicateType: ", { bold: true }], "https://securemodelgate.hpe.com/lineage/v0.1 (versioned; the domain must be approved internally). in-toto predicate types are not registered, and custom types are explicitly allowed [32]."]));
children.push(bullet([["Predicate fields: ", { bold: true }], "declaredBase {purl, omsBundleDigest, signerIdentity}; mlbomDigest; lineage {S_w, threshold, minLayerScore, layerFloorThreshold, exempt, maxExempt, α_L, calibrationSetDigest, layerFloorCalibrationSetDigest}; envelope {minCKA, JS, threshold, α, n, calibrationSetDigest}; probe {poolDigest, seedCommitment, seed}; tier; deepScan {tool, version, resultDigest}; verdict; gateVersion; timestamp. Every threshold is traceable to the exact calibration data behind it, and exempt lists every unscored weight matrix with its reason."]));
children.push(bullet([["Verification at admission. ", { bold: true }], "A valid signature only proves the gate said something. The verifier (securemodelgate/verify.py) also recomputes the candidate's weight digest, the ML-BOM digest, all three calibration-set digests, the probe-pool digest and the seed commitment from the artifacts actually presented, and fails closed on any mismatch, or when a Tier 1 attestation is presented without the inputs needed to check it. It also re-derives what the predicate asserts: each threshold must be the conformal quantile of its bound calibration set, the tier and verdict must follow from the recorded scores, and, given the base, every score is recomputed (within 1e-6). So even a correctly signed but false predicate is rejected. Optional freshness, revocation (a trusted-base registry) and policy checks bound replay. The subject digest covers the full state dict, BatchNorm statistics included, and signatures cover DSSE's pre-authentication encoding. Tamper tests change each input in turn. Building this surfaced two binding gaps, both fixed: the gate_version parameter was dropped (every predicate said 0.1.0), and the envelope and floor calibration sets weren't digested at all."]));
children.push(bullet([["The probe-seed commitment is a reproducibility record, not a defence. ", { bold: true }], "The seed and its hash are published together, and a small integer seed can be recovered from its hash by enumeration, so the commitment proves nothing about ordering. The adaptive pilot also shows that an attacker who knows only the probe distribution already collapses detection. Making the seed a real defence would require publishing a salted commitment before the candidate is submitted, and E7 testing a fixed-pool attacker against it."]));
children.push(bullet([["Signing: ", { bold: true }], "the gate signs a DSSE envelope as a Sigstore bundle, using keyless signing where the cluster is connected, and a private Sigstore instance or KMS key in air-gapped HPE Private Cloud deployments — replacing the original RSA-4096 JWT \"MAT\" and bespoke JSON MBOM with standard, verifiable formats."]));
children.push(bullet([["Placement: ", { bold: true }], "the current model-signing tooling (1.1.1) documents no custom-predicate option, so the attestation ships alongside the OMS bundle, not inside it."]));

children.push(h3(["3.5  Risk-tiered admission."]));
children.push(bodyPara(["A policy engine verifies the OMS signature and the SecureModelGate attestation, both bound to the same digests. It can run as a validating webhook, Kyverno, or the Sigstore policy-controller, and can also hook into the KServe storage initializer for non-OCI storage URIs."]));
children.push(bullet([["Tier 0: ", { bold: true }], "trusted signed publisher and unmodified bytes → admit (signature check only)."]));
children.push(bullet([["Tier 1: ", { bold: true }], "OMS-signed declared base, derivative → lineage + envelope test. Pass both → admit with attestation; fail envelope only → escalate to Tier 2 (lineage held, behavior is merely uncertain — deep scan can still clear it); fail lineage (S_w scored, below threshold) → block directly, without deep scan (see A1 above for why)."]));
children.push(bullet([["Tier 2: ", { bold: true }], "no verifiable declared base (lineage cannot even be scored), or escalated from a failed envelope test → deep scan (pickle/code scanners, then MM-BD for classifiers or BAIT/PEFTGuard-class tools for LLMs/adapters), with human review on a positive result."]));
children.push(subBullet(["A synchronous-webhook gap this design must state, not leave implicit: a Kubernetes ValidatingAdmissionWebhook call is synchronous with a bounded timeout (a handful of seconds by default); deep scan is explicitly \"too slow ... to run on every import\" (Section 1) and will not finish inside that window. So ESCALATE cannot mean \"wait and see\" at the webhook: it must map to allowed: false, with deep scan running out of band and a later re-submission carrying the resolved ADMIT/BLOCK. securemodelgate/webhook.py implements this mapping, where only ADMIT becomes allowed: true; ESCALATE, BLOCK and any unrecognised verdict deny. Any other integration (Kyverno, policy-controller) must reproduce it."]));

children.push(h3(["Architecture (single figure, description)."]));
children.push(bodyPara(["A left-to-right pipeline: (1) Intake receives model files, the OMS bundle and the ML-BOM from Hugging Face, NGC or an internal registry. (2) The Reference Resolver reads pedigree.ancestors, fetches the base and verifies its OMS signature against the publisher allow-list. (3) A Tier Router branches into three lanes — Tier 0 (signature only), Tier 1 (the Lineage Engine: S_w, then probe-set CKA/JS against the base, then conformal decision using calibration sets stored as digests), and Tier 2 (the Deep Scan lane: pickle/code scanners, then MM-BD/BAIT). (4) An Attestor signs the in-toto predicate (Sigstore/KMS). (5) The Admission Controller, on HPE Private Cloud AI Kubernetes, verifies the OMS and SecureModelGate attestations and admits or blocks the model. (6) A telemetry arrow carries every decision and score to HPE OpsRamp."]));

// Section 4
children.push(h1("4. Evaluation"));
children.push(h3(["4.1  Pilot validation (not part of this evaluation)."]));
children.push(bodyPara(["Before committing GPU time to the protocol below, this revision's engineering work was pressure-tested five separate times against a small synthetic corpus — not to produce preliminary numbers for the table below (none of these substitute for it), but to catch real implementation problems before spending real GPU time on a broken design. All five found something:"]));
children.push(dataTable(
  ["Pilot", "What it tested", "Finding"],
  [
    ["A — Engineering validation", "Does the conformal/lineage/envelope machinery work end to end?", "Caught a real bug (1D BatchNorm matching collapsed lineage separation to ~0.002). Current run: 29/30 backdoors detected (one admitted), AUROC 0.997, 98.6% lineage accuracy, 0/20 benign false blocks (CI 0–16.8%). An earlier 10% figure came from a leaky test set."],
    ["B — Adaptive attacker (A4)", "Does the envelope test survive an attacker who trains directly against it?", "Detection collapses from 100% to 20% by penalty weight λ≈1; attack success stays ≥99.8% throughout. Reported as a real limitation, not fixed away."],
    ["C — Permutation robustness", "Does S_w survive an exact, function-preserving channel permutation, as Section 3.2 claims?", "Wrong as first stated: a genuine derivative was wrongly rejected. Now fully recovered on sequential networks (S_w 1.00) via sequential alignment. Residual/transformer nets need a model-specific map."],
    ["D — Critical review (three rounds)", "Adversarial scope tests, including checking an external review against the code", "A full pipeline bypass with two routes (splice; splice + reshape so foreign layers go unscored), each reaching 100% attack success at Tier 1 ADMIT — both closed; the full rule now admits 0/5 splices at every depth. Also fixed: a reference-digest binding gap, a silent NaN-corruption path, unbound calibration sets, and a broken quick-run config in the external evidence script. Spectra measured to hurt, so the default is now cosine-only. Two of its own earlier claims corrected."],
    ["E — Developer checklist (44 questions)", "Every question answered from the code, with new measurements: all 63 splice patterns, S_w-adaptive and joint attacks, fine-tune depth, pruning, quantization, probe variation, determinism, scaling", "An adaptive lineage forger and a joint-objective attacker are admitted 4/4. Benign fine-tunes 5× longer than calibrated are escalated 7/7. Six implementation bugs were found and fixed, including a fail-open on NaN JS and BatchNorm statistics outside the digest. A leaky benign test set behind the earlier 10% false-block figure was found and replaced."],
  ],
  [2000, 3300, 4500],
));
children.push(para([run("", {})], { spacing: { after: 120 } }));
children.push(bodyPara(["None of these numbers belong in the table below — they come from a 4-stage CNN on synthetic Gaussian images, not PreAct-ResNet-18 on CIFAR-10. What they establish is that the design in Section 3 is implemented correctly (or, where it wasn't, was found and either fixed or honestly scoped) before the real evaluation spends GPU time on it."]));

children.push(h3(["Real-model lineage evidence (lineage test only)."]));
children.push(bodyPara(["Run by the author with experiments/smg_bert_lineage.py (Supplementary: pilot_results_bert). Base bert-base-uncased; 10 descendants (7 public fine-tunes — SST-2, MNLI, IMDB, AG-News, Yelp, QQP, emotion — plus INT8, 30%-pruned and FFN-permuted copies of one of them); 25 MultiBERTs (same architecture and data, different seed) as unrelated models, 20 calibrating the thresholds and 5 held out. The 72 encoder weight matrices are scored, each model FFN-aligned first, cosine-only."]));
children.push(dataTable(
  ["Check", "Result"],
  [
    ["Descendants accepted", "10/10 (95% CI 69–100%)"],
    ["Held-out unrelated models rejected", "5/5 (95% CI 48–100%)"],
    ["Splice (7 base blocks + 5 MultiBERT blocks)", "Admitted by a median-only rule (median 1.000); blocked by the floor (worst matrix 0.056)"],
    ["FFN-permuted fine-tune", "Rejected without alignment (worst matrix 0.055); accepted with it (1.000)"],
    ["Weakest genuine matrix (Yelp fine-tune) vs floor", "0.930 vs 0.128: margin ≈ 0.80"],
    ["Best of 1,440 unrelated per-matrix scores", "0.128"],
    ["Same run with a 0.5 spectral blend", "Same verdicts; margin halves (0.965 vs 0.564); unrelated medians (0.554–0.555) sit on the median threshold (0.5549)"],
  ],
  [3600, 6200],
));
children.push(bodyPara(["The blend result confirms on real weights the cosine-only default chosen on the synthetic corpus. What this does not establish: the counts are small and mostly one publisher's fine-tunes; descent is taken from the model cards; no adaptive forger, no same-initialisation null and no envelope or backdoor test was run on BERT. It shows that lineage separates honest BERT descendants from honest unrelated BERTs, not that it resists the adaptive forgery measured synthetically in Section 2."]));

children.push(h3(["4.2  Retained from Parasparam 2026."]));
children.push(bodyPara(["40 CIFAR-10 ResNet-18 models (20 clean, 20 backdoored with BadNets, blended and high-frequency triggers), 3 runs, Apple M-series."]));
children.push(bullet(["The static check detected 20/20 (95% Clopper–Pearson CI 83–100%) and blocked 3/20 clean models (15%, CI 3–38%)."]));
children.push(bullet(["Per-stage times were 1.26 s (static) and 1.28 s (behavioral)."]));
children.push(bullet(["The behavioral stage at τ = 0.05 detected 0/20, consistent with the observed backdoored KL range of 0.001–0.034 nats (clean mean 0.00018, backdoored mean 0.0156). Because the maximum clean KL was not recorded, the earlier separability and AUC claims are withdrawn."]));

children.push(h3(["4.3  Revised protocol (identical models for all methods)."]));
children.push(bullet([["Base: ", { bold: true }], "one PreAct-ResNet-18 base b0 trained with BackdoorBench on CIFAR-10."]));
children.push(bullet([["Benign derivatives (n = 40, split 20 calibration / 20 test): ", { bold: true }], "fine-tunes of b0 on clean subsets with varied seeds and learning rates, magnitude-pruned variants (10–50%), and INT8 post-training-quantized variants."]));
children.push(bullet([["Backdoored derivatives (n = 50): ", { bold: true }], "10 each of BadNets, Blended, WaNet, SIG and Input-aware, implanted by fine-tuning b0. Inclusion requires attack success rate ≥ 90% and clean-accuracy drop ≤ 3 points."]));
children.push(bullet([["Lineage negatives: ", { bold: true }], "the 20 independently trained clean models from the original study."]));
children.push(bullet([["Baselines: ", { bold: true }], "Neural Cleanse [8] and MM-BD [10] (model-level), re-run on the same models. STRIP [9] is an input-level runtime detector, so it is reported separately, aggregated to a model-level score with the aggregation rule stated."]));
children.push(bullet([["Adaptive attack (A4): ", { bold: true }], "the backdoor is trained with an added penalty λ·(JS + (1 − CKA)) against b0 on the public probe pool."]));
children.push(bullet([["Metrics: ", { bold: true }], "TPR at the conformal threshold (α = 0.05); false-block rate on held-out benign derivatives with 95% CI; lineage accuracy; AUROC using the named envelope score; median time per model."]));

children.push(dataTable(
  ["Method (same models)", "Backdoor TPR (95% CI)", "False-block rate (95% CI)", "Lineage accuracy", "Median time/model"],
  [
    ["Original static check (Parasparam 2026; independent clean models)*", "20/20 = 100% (83–100%)", "3/20 = 15% (3–38%)", "n/a", "1.26 s (M-series)"],
    ["SecureModelGate Tier 1 (lineage + envelope)", "[pending E3]", "[pending E2]", "[pending E4]", "[pending E8]"],
    ["SecureModelGate tiered (Tier 1 → MM-BD on escalation)", "[pending E6]", "[pending E6]", "[pending E6]", "[pending E6]"],
    ["Neural Cleanse (re-run)", "[pending E5]", "[pending E5]", "n/a", "[pending E5]"],
    ["MM-BD (re-run)", "[pending E5]", "[pending E5]", "n/a", "[pending E5]"],
    ["STRIP (re-run, input-level, aggregated)", "[pending E5]", "[pending E5]", "n/a", "[pending E5]"],
    ["SecureModelGate Tier 1 under adaptive attack**", "[pending E7]", "—", "[pending E7]", "—"],
  ],
  [2700, 1900, 2100, 1500, 1600],
));
children.push(para([run("* The original 15% was measured on independently trained clean models, not on derivatives.", { size: 16, italic: true })], { spacing: { before: 100, after: 40 } }));
children.push(bodyPara([["** The closest evidence available before E7 runs is Supplementary Pilot B: detection 100%→20% by penalty weight λ≈1, ASR retained ≥99.8%. Those numbers are a different model/dataset scale and are not copied into this row — cited here only so the row isn't read as untested.", { size: 16, italic: true }]]));

children.push(h3(["4.4  Current execution status."]));
children.push(bodyPara(["The real-model lineage run (BERT, above) is done. As of this writing, E1–E9 have not run: this working environment has neither a GPU nor network access to huggingface.co/download.pytorch.org/the CIFAR-10 mirror, confirmed structural (not a stale cache) by an independent fresh-session retest. Schedule: E1–E9 in Oct–Nov 2026, on GPU compute with that network access; see the Verification & Reproducibility Notes at the end of this paper for the itemized diagnostic detail."]));

children.push(h3(["Planned extension (talk)."]));
children.push(bodyPara(["A public LLM lineage demo: a small open base (e.g., Qwen2.5-0.5B) against public fine-tunes, LoRA adapters and GGUF quantizations, and against a same-architecture model from another family, reporting the LLM lineage score gap. It is followed by TrojAI rounds, whose final report covers weight-analysis and trigger-inversion detectors and \"natural\" Trojans."]));

// Figures
children.push(h3(["Supplementary evidence figures."]));
children.push(...figure("pilot_results/roc_curve.png", 500, 500, "Figure 1. Envelope-score ROC, engineering validation pilot (n=40 independent, n=40 benign, n=30 backdoored; AUROC 0.997)."));
children.push(...figure("pilot_results_adaptive/adaptive_attack.png", 600, 400, "Figure 2. Adaptive attacker (A4) pilot: detection rate and attack success rate vs. penalty weight λ. Detection collapses from 100%→20% by λ≈1 while ASR stays ≥99.8%."));
children.push(...figure("pilot_results/trigger_breakdown.png", 500, 400, "Figure 3. Per-trigger-type detection rate vs. mean attack success rate (patch / blended / warped triggers)."));

// Section 5
children.push(h1("5. Related Work & Differentiation"));
children.push(bullet([["Signing and admission. ", { bold: true }], "Sigstore (Newman et al., CCS 2022 [2]), OMS v1.0 [29], the Sigstore model-validation-operator [30], KServe RFC #5789 [31], and policy engines such as Kyverno and the Sigstore policy-controller all verify byte integrity and signer identity. SecureModelGate consumes these and adds a claim about descent and behavior."]));
children.push(bullet([["Attestation gates and lifecycle provenance. ", { bold: true }], "Tan et al. (LLMSC 2026, FSE '26 Companion [24]) enforce signed training/release claims with in-toto and CycloneDX [32] but leave backdoor-level evidence to optional plug-ins; Atlas [25] records transformation attestations, but only inside an instrumented pipeline. SecureModelGate works post hoc on third-party derivatives, with no producer cooperation beyond the base's OMS signature."]));
children.push(bullet([["Lineage and fingerprinting. ", { bold: true }], "HuRef (NeurIPS 2024 [15]), REEF (ICLR 2025 [16]), modelDNA (2026 [22]) and Centered Residual Signatures (2026 [23]) score lineage but do not sign or enforce the verdict. SecureModelGate reuses these signal families rather than claiming a new lineage metric."]));
children.push(bullet([["Backdoor detection. ", { bold: true }], "Neural Cleanse [8], ABS [13], MNTD [14], UNICORN [11], BTI-DBF [12], MM-BD [10], BAIT [17] and PEFTGuard (S&P 2025 [18]), and weights-only LoRA detection (2026 preprint [26]) are deep detectors. SecureModelGate places them behind a calibrated triage tier. BackdoorBench [7] and the TrojAI final report [6] supply the evaluation methodology."]));
children.push(bullet([["Commercial scanners. ", { bold: true }], "Palo Alto Networks Prisma AIRS (Protect AI, acquisition completed July 2025 [38]), Cisco AI Defense [39] and JFrog's Hugging Face scanning [37] focus on malicious code, vulnerabilities and posture. In public documentation we found no product that issues a signed, calibrated lineage verdict."]));

// Section 6
children.push(h1("6. Business Impact for HPE"));
children.push(bullet([["Product fit. ", { bold: true }], "HPE Private Cloud AI and HPE AI Essentials Software customers import open models and create their own fine-tunes. SecureModelGate gives them fast admission for NGC-signed bases (Tier 0), a verifiable lineage record for derivatives (Tier 1), and a deep-scan lane for unknown models. It fits the Kubernetes management in HPE Private Cloud (Morpheus-based, announced May 2026) and air-gapped deployments (private Sigstore/KMS signing)."]));
children.push(bullet([["Operations. ", { bold: true }], "Decisions and scores stream to HPE OpsRamp and can feed GreenLake Intelligence workflows, so model-admission posture becomes observable fleet-wide."]));
children.push(bullet([["Compliance evidence. ", { bold: true }], "The signed predicate is machine-readable evidence for EU AI Act Art. 11/Annex IV technical documentation, Art. 15(5) resilience against data and model poisoning, and Art. 53 GPAI downstream-provider information [34]. The Digital Omnibus, Regulation (EU) 2026/1744 [34] (in force 27 July 2026), moved Annex III high-risk obligations to 2 December 2027. The work also aligns with the NSA-led multinational \"AI/ML Supply Chain Risks and Mitigations\" CSI [33] (4 March 2026), CISA's \"AI Data Security\" [33] (May 2025), NIST AI RMF, AI 600-1 [5] and AI 100-2e2025 [4], and the EO 14028/14144/14306 secure-software attestation regime [40]."]));
children.push(bullet([["Differentiation. ", { bold: true }], "Competitors scan artifacts; HPE would attest lineage. The predicate format could also be contributed to the OpenSSF AI/ML Working Group."]));

// Section 7
children.push(h1("7. Status & Roadmap"));
children.push(bullet([["Status (as of 2026-09-30): ", { bold: true }], "the Parasparam 2026 prototype (static + behavioral stages, CLI) remains as-is. The lineage engine, conformal calibration and in-toto attestor are implemented, installable, tested, and validated by four independent pilots end to end against small synthetic models. What remains is wiring it to the real evaluation corpus (E1–E9) instead of the demo's synthetic models — currently blocked on GPU compute and network access to CIFAR-10/Hugging Face (Section 4.4)."]));
children.push(bullet([["By Oct 2 2026: ", { bold: true }], "the 4-day core evaluation (Section 4) fills the placeholders in the table and abstract. At risk given the blocker above; if it slips, the placeholders remain open rather than the date silently passing."]));
children.push(bullet([["Q4 2026: ", { bold: true }], "LLM/LoRA/GGUF lineage demo on public models; admission policy (Kyverno/policy-controller) on an HPE Private Cloud AI test cluster; OpsRamp telemetry."]));
children.push(bullet([["Q1 2027: ", { bold: true }], "TrojAI-round evaluation and scaled calibration (n ≥ 99 per architecture family, for 1% false-block claims); draft predicate spec shared with the OpenSSF AI/ML WG."]));
children.push(bullet([["Tech Con 2027: ", { bold: true }], "live demo of tiered admission and the lineage attestation."]));

// References
children.push(h1("References"));
const refs = [
"T. Gu, B. Dolan-Gavitt, S. Garg, \"BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain,\" arXiv:1708.06733, 2017. Journal version: T. Gu, K. Liu, B. Dolan-Gavitt, S. Garg, \"BadNets: Evaluating Backdooring Attacks on Deep Neural Networks,\" IEEE Access 7:47230–47244, 2019, doi:10.1109/ACCESS.2019.2909068.",
"Z. Newman, J. S. Meyers, S. Torres-Arias, \"Sigstore: Software Signing for Everybody,\" ACM CCS 2022, doi:10.1145/3548606.3560596.",
"M. Goldblum et al., \"Dataset Security for Machine Learning: Data Poisoning, Backdoor Attacks, and Defenses,\" IEEE TPAMI 45(2):1563–1580, 2023, doi:10.1109/TPAMI.2022.3162397; arXiv:2012.10544.",
"A. Vassilev et al., \"Adversarial Machine Learning: A Taxonomy and Terminology of Attacks and Mitigations,\" NIST AI 100-2e2025, 2025, doi:10.6028/NIST.AI.100-2e2025.",
"NIST, \"AI RMF: Generative AI Profile,\" NIST AI 600-1, 2024, doi:10.6028/NIST.AI.600-1.",
"K. W. Reese et al., \"Trojans in Artificial Intelligence (TrojAI) Final Report,\" arXiv:2602.07152, 2026.",
"B. Wu et al., \"BackdoorBench: A Comprehensive Benchmark of Backdoor Learning,\" NeurIPS 2022 Datasets & Benchmarks, arXiv:2206.12654.",
"B. Wang et al., \"Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks,\" IEEE S&P 2019, doi:10.1109/SP.2019.00031.",
"Y. Gao et al., \"STRIP: A Defence Against Trojan Attacks on Deep Neural Networks,\" ACSAC 2019, doi:10.1145/3359789.3359790; arXiv:1902.06531.",
"H. Wang, Z. Xiang, D. J. Miller, G. Kesidis, \"MM-BD: Post-Training Detection of Backdoor Attacks with Arbitrary Backdoor Pattern Types Using a Maximum Margin Statistic,\" IEEE S&P 2024, doi:10.1109/SP54263.2024.00015; arXiv:2205.06900.",
"Z. Wang, K. Mei, J. Zhai, S. Ma, \"UNICORN: A Unified Backdoor Trigger Inversion Framework,\" ICLR 2023, arXiv:2304.02786.",
"X. Xu et al., \"Towards Reliable and Efficient Backdoor Trigger Inversion via Decoupling Benign Features\" (BTI-DBF), ICLR 2024, OpenReview id Tw9wemV6cb.",
"Y. Liu et al., \"ABS: Scanning Neural Networks for Back-doors by Artificial Brain Stimulation,\" ACM CCS 2019, doi:10.1145/3319535.3363216.",
"X. Xu et al., \"Detecting AI Trojans Using Meta Neural Analysis\" (MNTD), IEEE S&P 2021, doi:10.1109/SP40001.2021.00034; arXiv:1910.03137.",
"B. Zeng et al., \"HuRef: HUman-REadable Fingerprint for Large Language Models,\" NeurIPS 2024, arXiv:2312.04828.",
"J. Zhang et al., \"REEF: Representation Encoding Fingerprints for Large Language Models,\" ICLR 2025, arXiv:2410.14273.",
"G. Shen et al., \"BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target,\" IEEE S&P 2025, pp. 1676–1694.",
"Z. Sun et al., \"PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning,\" IEEE S&P 2025, pp. 1713–1731.",
"S. Kornblith, M. Norouzi, H. Lee, G. Hinton, \"Similarity of Neural Network Representations Revisited,\" ICML 2019, arXiv:1905.00414.",
"A. N. Angelopoulos, S. Bates, \"A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification,\" arXiv:2107.07511.",
"T. A. Nguyen, A. Tran, \"WaNet – Imperceptible Warping-based Backdoor Attack,\" ICLR 2021, arXiv:2102.10369; \"Input-Aware Dynamic Backdoor Attack,\" NeurIPS 2020, arXiv:2010.08138.",
"M. A. Bin Adil, S. Aamir, \"modelDNA: Calibrated Lineage Verification and Merge Decomposition from Sampled Weight Fingerprints,\" arXiv:2607.10617, 2026.",
"A. S. Thakur, R. Khoury, \"Training Leaves Traces: Centered Residual Signatures for Language Model Lineage Verification,\" arXiv:2608.14929, 2026 (preprint).",
"Z. Tan, J. Singer, C. Anagnostopoulos, \"Attesting LLM Pipelines: Enforcing Verifiable Training and Release Claims,\" 2nd Int'l Workshop on LLM Supply Chain Analysis (LLMSC 2026), co-located with FSE '26, in FSE Companion '26, ACM, doi:10.1145/3803437.3805530; arXiv:2603.28988.",
"M. Spoczynski, M. S. Melara, S. Szyller, \"Atlas: A Framework for ML Lifecycle Provenance & Transparency,\" IEEE EuroS&PW 2025, pp. 448–461; arXiv:2502.19567.",
"D. Puertolas Merenciano, E. Vasyagina, K. Zhu, J. Ferrando, M. Chaudhary, \"Weight Space Detection of Backdoors in LoRA Adapters,\" arXiv:2602.15195, 2026 (preprint).",
"A. D. Kellas et al., \"PickleBall: Secure Deserialization of Pickle-based Machine Learning Models,\" ACM CCS 2025, doi:10.1145/3719027.3765037; arXiv:2508.15987.",
"J. Zhao et al., \"Models Are Codes: Towards Measuring Malicious Code Poisoning Attacks on Pre-trained Model Hubs,\" ASE 2024, doi:10.1145/3691620.3695271; arXiv:2409.09368.",
"OpenSSF, \"OpenSSF Model Signing (OMS) Specification,\" github.com/ossf/model-signing-spec; \"Launch of Model Signing v1.0,\" OpenSSF blog, 4 Apr 2025.",
"Sigstore, \"model-validation-operator,\" github.com/sigstore/model-validation-operator; Sigstore blog, \"Trusting AI Models in Kubernetes.\"",
"KServe, \"RFC: Model artifact integrity verification for non-OCI storage URIs,\" Issue #5789.",
"in-toto Attestation Framework v1 (Statement, Predicate specs), github.com/in-toto/attestation; CycloneDX v1.7 specification (ML-BOM, pedigree).",
"NSA AISC et al., \"Artificial Intelligence and Machine Learning – Supply Chain Risks and Mitigations,\" CSI, 4 Mar 2026; NSA/CISA/FBI et al., \"AI Data Security,\" 22 May 2025.",
"Regulation (EU) 2024/1689 (AI Act); Regulation (EU) 2026/1744 (Digital Omnibus on AI).",
"E. Horwitz et al., \"Model Atlas,\" NeurIPS 2025 (position paper track), arXiv:2503.10633; project page horwitz.ai/model-atlas. Verified against the primary source (previously cited only secondhand via modelDNA [22]).",
"NVIDIA Developer Blog, \"Bringing Verifiable Trust to AI Models: Model Signing in NGC,\" developer.nvidia.com/blog/bringing-verifiable-trust-to-ai-models-model-signing-in-ngc, 28 Jul 2025. Confirms NGC Catalog models signed with OpenSSF Model Signing since March 2025.",
"JFrog and Hugging Face, \"JFrog and Hugging Face Join Forces,\" jfrog.com/blog/jfrog-and-hugging-face-join-forces (mirrored at huggingface.co/blog/jfrog); confirms the >96% false-positive-reduction figure for JFrog's Hugging Face Hub scanning/curation.",
"Palo Alto Networks, \"Palo Alto Networks Completes Acquisition of Protect AI,\" press release, 22 Jul 2025, paloaltonetworks.com/company/press/2025/palo-alto-networks-completes-acquisition-of-protect-ai; now inside Prisma AIRS.",
"Cisco, \"AI Defense\" product documentation, on its AI Supply Chain Risk Management capability — exact URL/version still unconfirmed (no primary source was located for this pass). Locate and cite before submission, or drop the Cisco AI Defense mention from Section 5 if it cannot be confirmed in time.",
"Executive Order 14028, \"Improving the Nation's Cybersecurity,\" 12 May 2021 (establishes the CISA self-attestation regime this paper cites). Its provisions were built upon by Executive Order 14144, \"Strengthening and Promoting Innovation in the Nation's Cybersecurity,\" 16 Jan 2025, which was in turn amended by Executive Order 14306, \"Sustaining Select Efforts To Strengthen the Nation's Cybersecurity,\" 6 Jun 2025 (EO 14306 amends EO 14144 and, separately, EO 13694 — it does not amend EO 14028 directly; verified against whitehouse.gov and the Federal Register, correcting an earlier draft of this reference).",
"M. Barni, K. Kallas, B. Tondi, \"A New Backdoor Attack in CNNs by Training Set Corruption Without Label Poisoning\" (SIG), IEEE ICIP 2019. Verified against the BackdoorBench benchmark's own citation for its SIG attack implementation (github.com/SCLBD/BackdoorBench, attack/sig.py), not merely a secondary index — resolves the previously-withdrawn placeholder for Section 2's A2 attack list.",
"X. Chen, C. Liu, B. Li, K. Lu, D. Song, \"Targeted Backdoor Attacks on Deep Learning Systems Using Data Poisoning\" (Blended), arXiv:1712.05526, 2017.",
"S. K. Ainsworth, J. Hayase, S. Srinivasa, \"Git Re-Basin: Merging Models modulo Permutation Symmetries,\" ICLR 2023, arXiv:2209.04836. Cited in Section 3.2 as the concrete direction for fully solving cascading channel-permutation recovery in S_w.",
];
refs.forEach((r, i) => {
  children.push(new Paragraph({
    children: seg([[`[${i + 1}] `, { bold: true }], r], { size: 18 }),
    indent: { left: 260, hanging: 260 },
    spacing: { after: 80 },
  }));
});

// Notes section
children.push(h1("Verification & Reproducibility Notes"));
children.push(bodyPara(["This revision was produced with heavy tooling support and documents its own verification status rather than presenting every claim as settled. The points below should be resolved before this paper is treated as submission-ready."]));
children.push(bullet([["Citations. ", { bold: true }], "References [35]–[43] have been checked against primary or benchmark-code sources and corrected where an earlier draft mischaracterized them — see [40]'s entry for the EO amendment-chain correction, and [41]'s entry for how \"SIG\" was resolved to a specific paper rather than a secondary-index guess. [39] (Cisco AI Defense) remains unconfirmed — no primary source was located; locate one or drop the Cisco mention from Section 5 before submission. HuRef's and REEF's exact venues/author lists, the NIST AI 600-1 DOI, and BTI-DBF's citation (no arXiv version; cited via OpenReview) were confirmed only through secondary indexes and are still worth a final check."]));
children.push(bullet([["References [23]–[26]. ", { bold: true }], "Corrected in a later pass: [26]'s title was wrong (the real title is \"Weight Space Detection of Backdoors in LoRA Adapters\"), [24]'s venue was incomplete (an LLMSC 2026 workshop paper published in the FSE '26 Companion), and [23], [25] and [26] had no authors. The corrected details came from search-index results because arXiv, dblp and the ACM DL were blocked from this environment; open each landing page and confirm the byline before submission."]));
children.push(bullet([["Real-model lineage evidence. ", { bold: true }], "experiments/smg_bert_lineage.py (bert-base-uncased fine-tunes against MultiBERTs) was run by the author on a Mac, cosine-only and with a 0.5 spectral blend; the numbers in Section 4 are transcribed from those console logs into pilot_results_bert/README.md. Commit the machine-readable results_cosine.json and results_blend.json there before submission."]));
children.push(bullet([["Preprints. ", { bold: true }], "modelDNA, Centered Residual Signatures, the weights-only LoRA detection paper, and the Tan et al. FSE '26 companion are preprints as of this writing; re-confirm publication status before submission."]));
children.push(bullet([["Experiments E1–E9 not yet run. ", { bold: true }], "See Section 4.4. The working environment used to prepare this revision has no GPU and its network policy blocks Hugging Face, PyTorch's download host, and the CIFAR-10 mirror; a network-access change was requested and, on retest — including in an independently spun-up fresh session — the block was unchanged. This is a structural limitation of that environment, not a transient issue. Running E1–E9 requires GPU compute and network access to those hosts from outside that environment."]));
children.push(bullet([["Internal approvals outstanding. ", { bold: true }], "The predicate URI domain (securemodelgate.hpe.com) is not a registered HPE domain. Every HPE product integration described (Private Cloud AI, AI Essentials, Private Cloud/Morpheus, GreenLake Intelligence, OpsRamp) is proposed, not shipped — confirm Product and Legal are comfortable with that framing before external review."]));
children.push(bullet([["Tech Con process. ", { bold: true }], "Word limits, scoring rubric, and whether a Parasparam-non-selected idea is eligible for Tech Con resubmission are not publicly documented; confirm with the Tech Con program office or a sponsoring Distinguished Technologist/Fellow."]));
children.push(bullet([["Adversarial review (Supplementary Pilot D). ", { bold: true }], "Three rounds of scope tests found and fixed a full pipeline bypass by two routes (splicing layers to game S_w's median, then reshaping the spliced layers so the scorer skipped them; each, combined with adaptive training against the envelope score, reached 100% attack success at Tier 1 ADMIT), a reference-digest binding gap, a silent NaN-corruption path in the conformal threshold, unbound calibration sets, and a broken quick-run configuration in the external BERT evidence script. The first round's claim to have fixed the bypass was incomplete and is corrected in 10_critical_review_pilot.md. A non-adaptive backdoor stealth sweep and a per-channel rescaling symmetry test came back clean."]));
children.push(bullet([["Developer checklist (Supplementary Pilot E). ", { bold: true }], "44 developer questions answered from the code in 13_developer_checklist_answers.md, with new measurements (securemodelgate defensibility-check) and a per-model evaluation.json. It found and fixed: a leaky benign test set behind the earlier 10% false-block figure; a fail-open path where a NaN JS divergence was ignored; BatchNorm statistics outside the subject digest; an unauthenticated DSSE payload type; a verifier that did not re-derive thresholds or verdicts; and an attacker-controlled matching cost. It measured, and did not fix: adaptive lineage forgery and the joint S_w + envelope attack are admitted 4/4; benign fine-tunes 5× longer than calibrated are escalated 7/7. tests/test_end_to_end_attacks.py runs each attack the gate should stop through the deployment entry points, and pins the known gap."]));
children.push(bullet([["Source and reproducibility. ", { bold: true }], "The full implementation (securemodelgate package), test suite, and all five supplementary pilot reports referenced above are maintained in the project's git repository alongside the markdown source of this paper, so every number in Section 4.1 and every figure above can be regenerated with a single command (securemodelgate evaluate / adaptive / permutation-check)."]));

// ---------- build doc ----------
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: FONT, size: 21 } },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1080, bottom: 1080, left: 1260, right: 1260 },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(REPO, "SecureModelGate_TechCon2027.docx"), buf);
  console.log("wrote docx");
});
