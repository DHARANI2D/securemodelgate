"""A small, real, statistically-reported evaluation of the pipeline.

This is NOT Experiments E1-E9
(`paper/techcon2027_revision/04_experiments_plan.md`) — those are the
paper's PreAct-ResNet-18/CIFAR-10/BackdoorBench protocol, still to be
run on real GPU time, and their numbers are what belongs in the paper's
Table 1 (Section 4). This module runs the SAME statistical protocol
(split-conformal calibration, Clopper-Pearson CIs, TPR@FPR, per-trigger
breakdown, lineage accuracy) end to end against
`securemodelgate/demo_models.py`'s small synthetic corpus, so the
pipeline's actual quantitative behavior — not just six hand-picked
demo scenarios — is checked and reported before that GPU time is spent.
Results belong in the paper as a labeled engineering-validation pilot,
never substituted into Table 1.

Run with `securemodelgate evaluate` or `run_evaluation(EvaluationConfig())`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Optional

from . import demo_models
from .lineage.behavioral_delta import compute_behavioral_delta_from_models
from .lineage.conformal import envelope_test
from .lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models
from .pipeline import envelope_anomaly_score, lineage_verdict_for
from .stats import Proportion, auroc, clopper_pearson, tpr_at_fpr


@dataclass
class EvaluationConfig:
    base_seed: int = 1
    n_independent_calibration: int = 20     # lineage-test null population
    n_independent_test: int = 20            # held-out lineage negatives, for E4's accuracy figure
    n_benign_calibration: int = 20          # envelope-test null population (fine-tune/pruned/quantized mix)
    n_benign_test: int = 20                 # held-out benign derivatives, for the false-block rate
    n_backdoored_per_trigger: int = 10      # x len(TRIGGER_TYPES) backdoored derivatives
    alpha_lineage: float = 0.05
    alpha_envelope: float = 0.05
    target_fpr: float = 0.05
    probe_seed: int = 20260101


@dataclass
class TriggerBreakdown:
    trigger_type: str
    n: int
    tp: int
    mean_asr: float


@dataclass
class EvaluationReport:
    config: EvaluationConfig
    false_block_rate: Proportion                        # E2
    backdoor_tpr_at_conformal_threshold: Proportion      # E3
    backdoor_tpr_at_target_fpr: float
    envelope_auroc: float
    trigger_breakdown: list[TriggerBreakdown]
    lineage_accuracy: Proportion                         # E4
    median_pipeline_latency_s: float                     # E8-lite
    lineage_calibration_scores: list[float]
    envelope_calibration_scores: list[float]
    envelope_score_labels: list[int]                     # 0 = benign test, 1 = backdoored
    envelope_scores: list[float]
    per_model: list = None                               # one record per model: see _record()

    def to_dict(self) -> dict:
        return asdict(self)


def _benign_spec(i: int, seed_offset: int) -> dict:
    """Every benign derivative starts from its own seeded fine-tune, so no two
    are byte-identical. (An earlier version pruned and quantized the BASE
    itself: deterministic, so the "pruned" and "quantized" models were
    repeated copies -- 11 distinct models out of 20 per half, 13 of the 20
    test models byte-identical to calibration models -- which made the
    reported false-block rate look better than the held-out evidence
    supported. `run_evaluation` now refuses any calibration/test overlap.)"""
    spec = {"seed": seed_offset + i, "steps": 8, "lr": 0.005}
    kind = i % 3
    if kind == 0:
        return {"kind": "fine_tune", **spec}
    if kind == 1:
        amount = 0.1 + 0.3 * ((i // 3) % 3) / 2   # vary prune amount a bit: 0.1, 0.25, 0.4, ...
        return {"kind": "fine_tune+magnitude_pruned", "amount": min(amount, 0.5), **spec}
    return {"kind": "fine_tune+quantized", "levels": 16, **spec}


def _make_benign_population(base_model, n: int, seed_offset: int) -> list:
    """A mix of fine-tunes, pruned fine-tunes, and quantized fine-tunes —
    mirroring the paper's "fine-tuned / pruned / INT8" benign-derivative
    shape (Section 4) at small scale."""
    out = []
    for i in range(n):
        spec = _benign_spec(i, seed_offset)
        m = demo_models.make_benign_fine_tune(base_model, seed=spec["seed"])
        if "amount" in spec:
            m = demo_models.make_pruned_derivative(m, amount=spec["amount"])
        elif "levels" in spec:
            m = demo_models.make_quantized_derivative(m, levels=spec["levels"])
        out.append(m)
    return out


def _record(model_id: str, population: str, spec: dict, model, lineage: dict, verdict, floor_verdict,
            behavioral: Optional[dict], envelope_score: Optional[float], envelope_threshold: Optional[float],
            asr: Optional[float] = None, lineage_threshold: Optional[float] = None) -> dict:
    """One per-model row of evaluation.json: everything needed to recompute
    the model's verdict, and to find which model a headline count missed."""
    from .digest import weight_digest

    lineage_pass = bool(verdict.exceeds_threshold)
    env_fail = None if envelope_score is None else bool(envelope_score > envelope_threshold)
    if not lineage_pass:
        decision = "BLOCK"
    elif env_fail is None:
        decision = "n/a"
    else:
        decision = "ESCALATE" if env_fail else "ADMIT"
    return {
        "id": model_id, "population": population, **spec,
        "weight_digest": weight_digest(model),
        "S_w": lineage["S_w"], "min_layer_score": lineage["min_layer_score"],
        "per_layer": dict(zip(lineage["layer_names"], lineage["per_layer"])),
        "exempt": lineage["exempt"],
        "lineage_threshold": lineage_threshold,
        "layer_floor_threshold": floor_verdict.threshold,
        "lineage_pass": lineage_pass,
        "lineage_failure": verdict.failure_reason or ("" if lineage_pass else
                           ("per-layer floor" if lineage["min_layer_score"] <= floor_verdict.threshold else "median")),
        "min_cka": None if behavioral is None else behavioral["min_cka"],
        "cka_per_layer": None if behavioral is None else dict(zip(behavioral["layers_compared"], behavioral["cka_per_layer"])),
        "js": None if behavioral is None else behavioral["js_divergence"],
        "envelope_score": envelope_score, "envelope_threshold": envelope_threshold,
        "envelope_exceeds": env_fail,
        "gate_decision": decision,
        "attack_success_rate": asr,
        "clean_accuracy": demo_models.clean_accuracy(model),
    }


def run_evaluation(config: Optional[EvaluationConfig] = None, verbose: bool = True) -> EvaluationReport:
    config = config or EvaluationConfig()
    log = print if verbose else (lambda *a, **k: None)

    log("Training base model b0 ...")
    base_model = demo_models.make_base_model(seed=config.base_seed)
    probe_loader = demo_models.make_probe_loader(seed=config.probe_seed)

    log(f"Building lineage-negative population "
        f"(n={config.n_independent_calibration + config.n_independent_test}) ...")
    independent_models = [
        demo_models.make_independent_model(seed=1000 + i)
        for i in range(config.n_independent_calibration + config.n_independent_test)
    ]
    lineage_score_dicts_all = [weight_lineage_score_from_models(m, base_model) for m in independent_models]
    lineage_calibration_dicts = lineage_score_dicts_all[:config.n_independent_calibration]
    lineage_test_negative_dicts = lineage_score_dicts_all[config.n_independent_calibration:]
    lineage_calibration_scores = [d["S_w"] for d in lineage_calibration_dicts]
    lineage_test_negative_scores = [d["S_w"] for d in lineage_test_negative_dicts]
    # Pooled per-layer scores from the CALIBRATION half only (no train/test
    # leakage into the floor threshold, mirroring lineage_calibration_scores).
    layer_floor_calibration_scores = pooled_layer_scores(lineage_calibration_dicts)

    log(f"Building benign-derivative population "
        f"(n={config.n_benign_calibration + config.n_benign_test}: fine-tune/pruned/quantized mix) ...")
    benign_all = _make_benign_population(
        base_model, config.n_benign_calibration + config.n_benign_test, seed_offset=2000,
    )
    benign_calibration = benign_all[:config.n_benign_calibration]
    benign_test = benign_all[config.n_benign_calibration:]
    from .digest import weight_digest
    cal_digests = {weight_digest(m) for m in benign_calibration}
    test_digests = [weight_digest(m) for m in benign_test]
    if cal_digests & set(test_digests) or len(set(test_digests)) != len(test_digests):
        raise RuntimeError("benign test set overlaps the calibration set or repeats a model; "
                           "the held-out false-block rate would be meaningless")

    behavioral_cache: dict[int, dict] = {}

    def behavioral_of(model) -> dict:
        if id(model) not in behavioral_cache:
            behavioral_cache[id(model)] = compute_behavioral_delta_from_models(model, base_model, probe_loader)
        return behavioral_cache[id(model)]

    def envelope_score_of(model) -> float:
        return envelope_anomaly_score(behavioral_of(model))

    envelope_calibration_scores = [envelope_score_of(m) for m in benign_calibration]

    log(f"Building backdoored-derivative population "
        f"(n={config.n_backdoored_per_trigger} x {len(demo_models.TRIGGER_TYPES)} trigger types) ...")
    backdoored_by_type: dict[str, list] = {}
    for trigger_type in demo_models.TRIGGER_TYPES:
        backdoored_by_type[trigger_type] = [
            demo_models.make_backdoored_derivative(base_model, seed=3000 + i, trigger_type=trigger_type)
            for i in range(config.n_backdoored_per_trigger)
        ]

    log("Scoring the held-out benign test set (E2: false-block rate) ...")
    benign_test_scores = [envelope_score_of(m) for m in benign_test]
    benign_test_verdicts = [
        envelope_test(s, envelope_calibration_scores, alpha=config.alpha_envelope)
        for s in benign_test_scores
    ]
    false_blocks = sum(1 for v in benign_test_verdicts if v.exceeds_threshold)
    false_block_rate = clopper_pearson(false_blocks, len(benign_test_verdicts), alpha=0.05)

    log("Scoring backdoored derivatives (E3: TPR, AUROC, per-trigger breakdown) ...")
    trigger_breakdown = []
    backdoor_scores_all = []
    for trigger_type, models in backdoored_by_type.items():
        scores = [envelope_score_of(m) for m in models]
        backdoor_scores_all.extend(scores)
        verdicts = [envelope_test(s, envelope_calibration_scores, alpha=config.alpha_envelope) for s in scores]
        tp = sum(1 for v in verdicts if v.exceeds_threshold)
        asrs = [demo_models.attack_success_rate(m, seed=3000 + i, trigger_type=trigger_type)
                for i, m in enumerate(models)]
        trigger_breakdown.append(TriggerBreakdown(
            trigger_type=trigger_type, n=len(models), tp=tp, mean_asr=sum(asrs) / len(asrs),
        ))

    tp_total = sum(tb.tp for tb in trigger_breakdown)
    n_backdoored_total = sum(tb.n for tb in trigger_breakdown)
    backdoor_tpr = clopper_pearson(tp_total, n_backdoored_total, alpha=0.05)

    envelope_labels = [0] * len(benign_test_scores) + [1] * len(backdoor_scores_all)
    envelope_scores = benign_test_scores + backdoor_scores_all
    envelope_auroc = auroc(envelope_labels, envelope_scores)
    backdoor_tpr_at_target = tpr_at_fpr(envelope_labels, envelope_scores, target_fpr=config.target_fpr)

    log("Scoring lineage accuracy (E4: independent-negatives vs. genuine-derivative-positives) ...")
    # Positives: benign_test + all backdoored derivatives (all genuinely
    # descend from base_model). Negatives: the held-out independent
    # models NOT used to calibrate the lineage test (no train/test leakage).
    positive_models = benign_test + [m for models in backdoored_by_type.values() for m in models]
    positive_dicts = [weight_lineage_score_from_models(m, base_model) for m in positive_models]

    def _lineage_verdict(score_dict: dict):
        return lineage_verdict_for(score_dict, lineage_calibration_scores, layer_floor_calibration_scores,
                                    alpha=config.alpha_lineage)[0]

    lineage_positive_verdicts = [_lineage_verdict(d) for d in positive_dicts]
    lineage_negative_verdicts = [_lineage_verdict(d) for d in lineage_test_negative_dicts]
    correct = (sum(1 for v in lineage_positive_verdicts if v.exceeds_threshold)
               + sum(1 for v in lineage_negative_verdicts if not v.exceeds_threshold))
    total = len(lineage_positive_verdicts) + len(lineage_negative_verdicts)
    lineage_accuracy = clopper_pearson(correct, total, alpha=0.05)

    log("Writing per-model records ...")
    from .lineage.conformal import conformal_quantile
    env_tau = conformal_quantile(envelope_calibration_scores, config.alpha_envelope)
    lin_tau = conformal_quantile(lineage_calibration_scores, config.alpha_lineage)
    per_model = []
    for i, (m, d) in enumerate(zip(independent_models, lineage_score_dicts_all)):
        pop = "independent_calibration" if i < config.n_independent_calibration else "independent_test"
        v, fv = lineage_verdict_for(d, lineage_calibration_scores, layer_floor_calibration_scores,
                                     alpha=config.alpha_lineage)
        per_model.append(_record(f"indep-{1000 + i}", pop, {"kind": "independent", "seed": 1000 + i},
                                 m, d, v, fv, None, None, None, lineage_threshold=lin_tau))
    n_bc = config.n_benign_calibration
    for i, m in enumerate(benign_all):
        pop = "benign_calibration" if i < n_bc else "benign_test"
        d = weight_lineage_score_from_models(m, base_model)
        v, fv = lineage_verdict_for(d, lineage_calibration_scores, layer_floor_calibration_scores,
                                     alpha=config.alpha_lineage)
        per_model.append(_record(f"benign-{i}", pop, _benign_spec(i, 2000), m, d, v, fv,
                                 behavioral_of(m), envelope_score_of(m), env_tau, lineage_threshold=lin_tau))
    for trigger_type, models in backdoored_by_type.items():
        for i, m in enumerate(models):
            d = weight_lineage_score_from_models(m, base_model)
            v, fv = lineage_verdict_for(d, lineage_calibration_scores, layer_floor_calibration_scores,
                                         alpha=config.alpha_lineage)
            per_model.append(_record(
                f"backdoor-{trigger_type}-{3000 + i}", "backdoored",
                {"kind": "backdoored_fine_tune", "trigger": trigger_type, "seed": 3000 + i, "steps": 60, "lr": 0.03,
                 "poison_fraction": 0.5}, m, d, v, fv, behavioral_of(m), envelope_score_of(m), env_tau,
                asr=demo_models.attack_success_rate(m, seed=3000 + i, trigger_type=trigger_type),
                lineage_threshold=lin_tau))

    log("Timing the scoring pipeline (E8-lite: S_w + D_b, excluding model training) ...")
    timing_candidates = (benign_test[:5] + [models[0] for models in backdoored_by_type.values()])
    latencies = []
    for m in timing_candidates:
        t0 = time.perf_counter()
        weight_lineage_score_from_models(m, base_model)
        compute_behavioral_delta_from_models(m, base_model, probe_loader)
        latencies.append(time.perf_counter() - t0)
    latencies.sort()
    median_latency = latencies[len(latencies) // 2]

    report = EvaluationReport(
        config=config,
        false_block_rate=false_block_rate,
        backdoor_tpr_at_conformal_threshold=backdoor_tpr,
        backdoor_tpr_at_target_fpr=backdoor_tpr_at_target,
        envelope_auroc=envelope_auroc,
        trigger_breakdown=trigger_breakdown,
        lineage_accuracy=lineage_accuracy,
        median_pipeline_latency_s=median_latency,
        lineage_calibration_scores=lineage_calibration_scores,
        envelope_calibration_scores=envelope_calibration_scores,
        envelope_score_labels=envelope_labels,
        envelope_scores=envelope_scores,
        per_model=per_model,
    )
    log("Done.")
    return report


def render_markdown_report(report: EvaluationReport) -> str:
    c = report.config
    lines = [
        "# SecureModelGate — engineering validation pilot",
        "",
        "**This is not Experiments E1-E9.** It is the same statistical "
        "protocol (split-conformal calibration, Clopper-Pearson CIs, "
        "TPR@FPR) run against `securemodelgate/demo_models.py`'s small "
        "synthetic corpus (a 4-stage CNN, synthetic Gaussian-noise "
        "images), not the paper's PreAct-ResNet-18/CIFAR-10/BackdoorBench "
        "protocol. Do not paste these numbers into the paper's Table 1 — "
        "see `paper/techcon2027_revision/04_experiments_plan.md` for "
        "what has to run instead.",
        "",
        f"Config: `{report.config}`",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| False-block rate (benign derivatives) | {report.false_block_rate} |",
        f"| Backdoor TPR @ conformal threshold (α={c.alpha_envelope}) | {report.backdoor_tpr_at_conformal_threshold} |",
        f"| Backdoor TPR @ target FPR={c.target_fpr} | {report.backdoor_tpr_at_target_fpr:.1%} |",
        f"| Envelope-score AUROC | {report.envelope_auroc:.4f} |",
        f"| Lineage accuracy (positives vs. held-out negatives) | {report.lineage_accuracy} |",
        f"| Median pipeline latency (S_w + D_b only) | {report.median_pipeline_latency_s * 1000:.2f} ms |",
        "",
        "### Per-trigger-type breakdown",
        "",
        "| Trigger | n | TP | TPR | Mean ASR |",
        "|---|---|---|---|---|",
    ]
    for tb in report.trigger_breakdown:
        lines.append(f"| {tb.trigger_type} | {tb.n} | {tb.tp} | {tb.tp / tb.n:.1%} | {tb.mean_asr:.1%} |")
    lines.append("")
    return "\n".join(lines)
