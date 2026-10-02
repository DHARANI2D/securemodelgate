"""Experiment E7: does the envelope test survive the A4 adaptive
adversary (`securemodelgate.adaptive_attack`)?

Trains backdoored derivatives at several penalty weights λ (λ=0 is an
ordinary, non-adaptive poisoned fine-tune) against the SAME conformal
threshold a normal evaluation would calibrate, and reports detection
rate and attack-success-rate retained at each λ. This is deliberately
allowed to show the defense failing at high λ — Section 2's threat
model gives the attacker the full algorithm, and an evaluation that
can't show a limitation isn't testing anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import demo_models
from .adaptive_attack import make_adaptive_backdoored_derivative
from .lineage.behavioral_delta import compute_behavioral_delta_from_models
from .lineage.conformal import conformal_quantile
from .pipeline import envelope_anomaly_score
from .stats import Proportion, clopper_pearson


@dataclass
class AdaptiveAttackConfig:
    base_seed: int = 1
    n_benign_calibration: int = 20
    lambdas: tuple = (0.0, 0.5, 2.0, 5.0)
    n_per_lambda: int = 8
    alpha: float = 0.05
    trigger_type: str = "patch"
    poison_target: int = 0
    probe_seed: int = 20260101


@dataclass
class LambdaResult:
    lam: float
    n: int
    detection_rate: Proportion
    mean_attack_success_rate: float
    mean_envelope_score: float


@dataclass
class AdaptiveAttackReport:
    config: AdaptiveAttackConfig
    envelope_threshold: float
    results: list[LambdaResult]


def run_adaptive_attack_evaluation(config: AdaptiveAttackConfig | None = None,
                                    verbose: bool = True) -> AdaptiveAttackReport:
    config = config or AdaptiveAttackConfig()
    log = print if verbose else (lambda *a, **k: None)

    log("Training base model and benign-derivative calibration set ...")
    base_model = demo_models.make_base_model(seed=config.base_seed)
    probe_loader = demo_models.make_probe_loader(seed=config.probe_seed)

    def envelope_score_of(model) -> float:
        beh = compute_behavioral_delta_from_models(model, base_model, probe_loader)
        return envelope_anomaly_score(beh)

    calibration_scores = [
        envelope_score_of(demo_models.make_benign_fine_tune(base_model, seed=5000 + i))
        for i in range(config.n_benign_calibration)
    ]
    threshold = conformal_quantile(calibration_scores, alpha=config.alpha)
    log(f"Envelope conformal threshold (alpha={config.alpha}, n={config.n_benign_calibration}): {threshold:.4f}")

    results = []
    for lam in config.lambdas:
        log(f"Training {config.n_per_lambda} adaptive backdoors at lambda={lam} ...")
        detected = 0
        asrs, scores = [], []
        for i in range(config.n_per_lambda):
            model = make_adaptive_backdoored_derivative(
                base_model, seed=6000 + i, lam=lam,
                trigger_type=config.trigger_type, poison_target=config.poison_target,
            )
            score = envelope_score_of(model)
            asr = demo_models.attack_success_rate(model, seed=6000 + i, target=config.poison_target,
                                                    trigger_type=config.trigger_type)
            scores.append(score)
            asrs.append(asr)
            if score > threshold:
                detected += 1

        results.append(LambdaResult(
            lam=lam, n=config.n_per_lambda,
            detection_rate=clopper_pearson(detected, config.n_per_lambda, alpha=0.05),
            mean_attack_success_rate=sum(asrs) / len(asrs),
            mean_envelope_score=sum(scores) / len(scores),
        ))
        log(f"  lambda={lam}: detected {detected}/{config.n_per_lambda}, "
            f"mean ASR retained {results[-1].mean_attack_success_rate:.1%}, "
            f"mean envelope score {results[-1].mean_envelope_score:.4f}")

    return AdaptiveAttackReport(config=config, envelope_threshold=threshold, results=results)


def render_markdown_report(report: AdaptiveAttackReport) -> str:
    lines = [
        "# SecureModelGate — adaptive attacker (A4) pilot",
        "",
        "Attacker knows the exact envelope score (`max(1 - min-layer CKA, JS)`) "
        "and trains a poisoned fine-tune with an added penalty "
        "`lambda * envelope_score` on a public probe pool "
        "(Section 2's A4; `securemodelgate.adaptive_attack`). lambda=0 is an "
        "ordinary, non-adaptive poisoned fine-tune.",
        "",
        f"Envelope conformal threshold used for detection: **{report.envelope_threshold:.4f}** "
        f"(alpha={report.config.alpha}, n={report.config.n_benign_calibration} benign calibration models).",
        "",
        "| lambda | n | Detected | Mean ASR retained | Mean envelope score |",
        "|---|---|---|---|---|",
    ]
    for r in report.results:
        lines.append(f"| {r.lam} | {r.n} | {r.detection_rate} | "
                      f"{r.mean_attack_success_rate:.1%} | {r.mean_envelope_score:.4f} |")
    lines.append("")
    return "\n".join(lines)
