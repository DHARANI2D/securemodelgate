"""`securemodelgate` command-line interface.

    securemodelgate demo [--out-dir DIR]
        Builds a small base model and five kinds of candidate (trusted
        publisher, benign fine-tune, pruned derivative, poisoned
        derivative, lineage-forged/independent model), runs each
        through the full admission pipeline, and writes a signed
        in-toto attestation JSON per candidate.

    securemodelgate verify <attestation.json> [--secret SECRET] [--keyid KEYID]
        Verifies a previously issued attestation's DSSE signature and
        prints its predicate.

Install with `pip install -e .` (from the repo root) to get the
`securemodelgate` entry point, or run as `python -m securemodelgate.cli`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEMO_SIGNER_KEY_ID = "securemodelgate-demo-key"
DEMO_SIGNER_SECRET = b"securemodelgate-demo-secret-not-for-production"


def _cmd_demo(args: argparse.Namespace) -> int:
    # Imported lazily: torch is only needed for `demo`/`admit`, not for
    # anything that only touches securemodelgate.lineage's pure-math core.
    from . import demo_models, oms, pipeline
    from .lineage.behavioral_delta import compute_behavioral_delta_from_models
    from .lineage.intoto_attestation import LocalHmacSigner
    from .lineage.lineage_score import pooled_layer_scores, weight_lineage_score_from_models

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signer = LocalHmacSigner(key_id=DEMO_SIGNER_KEY_ID, secret=DEMO_SIGNER_SECRET)
    allowed_signers = {"ngc-catalog-demo"}
    base_purl = "pkg:huggingface/demo-org/tinyconvnet-base@v1"

    print("Training base model b0 ...")
    base_model = demo_models.make_base_model(seed=1)
    mlbom = oms.build_mlbom(base_purl, base_model, signer_identity="ngc-catalog-demo")
    unsigned_mlbom = oms.build_unsigned_mlbom()
    probe_loader = demo_models.make_probe_loader(seed=20260101)
    probe_seed = 20260101

    print("Building calibration populations ...")
    lineage_calibration = []
    lineage_score_dicts = []
    for i in range(5):
        indep = demo_models.make_independent_model(seed=100 + i)
        score_dict = weight_lineage_score_from_models(indep, base_model)
        lineage_calibration.append(score_dict["S_w"])
        lineage_score_dicts.append(score_dict)
    layer_floor_calibration = pooled_layer_scores(lineage_score_dicts)

    envelope_calibration = []
    for i in range(5):
        benign = demo_models.make_benign_fine_tune(base_model, seed=200 + i)
        beh = compute_behavioral_delta_from_models(benign, base_model, probe_loader)
        envelope_calibration.append(pipeline.envelope_anomaly_score(beh))

    print(f"  lineage calibration (independent models): {[round(s, 4) for s in lineage_calibration]}")
    print(f"  envelope calibration (benign derivatives): {[round(s, 4) for s in envelope_calibration]}")

    scenarios = []

    scenarios.append((
        "trusted_publisher", base_model, mlbom,
        dict(trusted_publisher_signed=True),
    ))
    scenarios.append((
        "unsigned_import", demo_models.make_independent_model(seed=7), unsigned_mlbom,
        dict(deep_scan_flagged=False),
    ))
    scenarios.append((
        "benign_fine_tune", demo_models.make_benign_fine_tune(base_model, seed=42), mlbom,
        dict(),
    ))
    scenarios.append((
        "pruned_derivative", demo_models.make_pruned_derivative(base_model, amount=0.2), mlbom,
        dict(),
    ))
    backdoored = demo_models.make_backdoored_derivative(base_model, seed=99)
    scenarios.append((
        "poisoned_derivative", backdoored, mlbom,
        dict(deep_scan_flagged=True),
    ))
    scenarios.append((
        "lineage_forged", demo_models.make_independent_model(seed=55), mlbom,
        dict(),
    ))

    print(f"\nPoisoned-derivative attack success rate on trigger inputs: "
          f"{demo_models.attack_success_rate(backdoored, seed=99):.1%}\n")

    print(f"{'scenario':22s} {'tier':5s} {'verdict':10s} reason")
    print("-" * 100)
    for name, candidate, this_mlbom, extra in scenarios:
        result = pipeline.run_admission(
            candidate=candidate, candidate_subject_name=f"{name}.pt", mlbom=this_mlbom,
            allowed_signers=allowed_signers, signer=signer,
            base_model=base_model, probe_loader=probe_loader, probe_seed=probe_seed,
            lineage_calibration_scores=lineage_calibration,
            envelope_calibration_scores=envelope_calibration,
            lineage_layer_floor_calibration_scores=layer_floor_calibration,
            **extra,
        )
        print(f"{name:22s} {result.decision.tier:5s} {result.decision.verdict:10s} {result.decision.reason}")

        out_path = out_dir / f"{name}.attestation.json"
        out_path.write_text(json.dumps(result.attestation, indent=2))

    print(f"\nSigned attestations written to {out_dir}/")
    print(f"Verify one with: securemodelgate verify {out_dir}/benign_fine_tune.attestation.json")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .lineage.intoto_attestation import LocalHmacSigner

    envelope = json.loads(Path(args.attestation_file).read_text())
    signer = LocalHmacSigner(key_id=args.keyid, secret=args.secret.encode())
    ok, payload = signer.verify(envelope)

    if not ok:
        print(f"INVALID: {payload}", file=sys.stderr)
        return 1

    print("VALID signature.")
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from . import figures
    from .evaluation import EvaluationConfig, render_markdown_report, run_evaluation

    config = EvaluationConfig(
        n_independent_calibration=args.n_independent // 2,
        n_independent_test=args.n_independent - args.n_independent // 2,
        n_benign_calibration=args.n_benign // 2,
        n_benign_test=args.n_benign - args.n_benign // 2,
        n_backdoored_per_trigger=args.n_backdoored_per_trigger,
        alpha_lineage=args.alpha,
        alpha_envelope=args.alpha,
        target_fpr=args.target_fpr,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_evaluation(config)

    summary = report.to_dict()
    per_model = summary.pop("per_model")
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / "evaluation.json").write_text(json.dumps(
        {"config": summary["config"], "models": per_model}, indent=2, default=str))
    (out_dir / "report.md").write_text(render_markdown_report(report))
    fig_paths = figures.write_all_figures(report, out_dir)

    print()
    print(render_markdown_report(report))
    print(f"Wrote report.json, evaluation.json (per model), report.md and {len(fig_paths)} figures to {out_dir}/")
    return 0


def _cmd_adaptive(args: argparse.Namespace) -> int:
    from . import figures
    from .adaptive_evaluation import AdaptiveAttackConfig, render_markdown_report, run_adaptive_attack_evaluation

    config = AdaptiveAttackConfig(
        n_benign_calibration=args.n_benign_calibration,
        lambdas=tuple(args.lambdas),
        n_per_lambda=args.n_per_lambda,
        alpha=args.alpha,
        trigger_type=args.trigger_type,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_adaptive_attack_evaluation(config)

    def _to_dict(r):
        return {"lam": r.lam, "n": r.n,
                "detection_rate": {"successes": r.detection_rate.successes, "n": r.detection_rate.n,
                                    "point_estimate": r.detection_rate.point_estimate,
                                    "ci_low": r.detection_rate.ci_low, "ci_high": r.detection_rate.ci_high},
                "mean_attack_success_rate": r.mean_attack_success_rate,
                "mean_envelope_score": r.mean_envelope_score}

    payload = {"envelope_threshold": report.envelope_threshold, "results": [_to_dict(r) for r in report.results]}
    (out_dir / "report.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "report.md").write_text(render_markdown_report(report))
    fig_path = out_dir / "adaptive_attack.png"
    figures.plot_adaptive_attack(report, fig_path)

    print()
    print(render_markdown_report(report))
    print(f"Wrote report.json, report.md and {fig_path.name} to {out_dir}/")
    return 0


def _cmd_permutation_check(args: argparse.Namespace) -> int:
    from .permutation_check import render_markdown_report, run_permutation_check
    from dataclasses import asdict

    report = run_permutation_check(base_seed=args.base_seed, derivative_seed=args.derivative_seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2))
    (out_dir / "report.md").write_text(render_markdown_report(report))

    print()
    print(render_markdown_report(report))
    print(f"Wrote report.json and report.md to {out_dir}/")
    return 0


def _cmd_lineage_scope_check(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .lineage_scope_check import render_markdown_report, run_lineage_scope_check

    report = run_lineage_scope_check(max_exempt=args.max_exempt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2))
    (out_dir / "report.md").write_text(render_markdown_report(report))
    print()
    print(render_markdown_report(report))
    print(f"Wrote report.json and report.md to {out_dir}/")
    return 0


def _cmd_defensibility_check(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from .defensibility_checks import render_markdown_report, run_defensibility_checks

    report = run_defensibility_checks(n_adaptive=args.n_adaptive)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(asdict(report), indent=2, default=str))
    (out_dir / "report.md").write_text(render_markdown_report(report))
    print()
    print(render_markdown_report(report))
    print(f"Wrote report.json and report.md to {out_dir}/")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="securemodelgate",
        description="Signed behavioral-lineage attestation for admitting third-party AI models.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo_p = sub.add_parser("demo", help="Run the full pipeline on small demo models.")
    demo_p.add_argument("--out-dir", default="./attestations", help="Where to write signed attestation JSON files.")
    demo_p.set_defaults(func=_cmd_demo)

    verify_p = sub.add_parser("verify", help="Verify a signed attestation's DSSE envelope.")
    verify_p.add_argument("attestation_file", help="Path to an attestation JSON file (as written by `demo`/`admit`).")
    verify_p.add_argument("--keyid", default=DEMO_SIGNER_KEY_ID, help="Expected signer key id.")
    verify_p.add_argument("--secret", default=DEMO_SIGNER_SECRET.decode(),
                           help="Shared HMAC secret (demo stand-in for real Sigstore/KMS verification).")
    verify_p.set_defaults(func=_cmd_verify)

    eval_p = sub.add_parser(
        "evaluate",
        help="Run a statistically-reported pilot evaluation (NOT the paper's E1-E9 protocol — see its --help).",
        description="Runs the paper's evaluation protocol (split-conformal calibration, Clopper-Pearson "
                     "CIs, TPR@FPR, per-trigger breakdown, lineage accuracy) against securemodelgate."
                     "demo_models's small synthetic corpus. This is an engineering-validation pilot, not "
                     "Experiments E1-E9 (paper/techcon2027_revision/04_experiments_plan.md) — those need the "
                     "paper's real PreAct-ResNet-18/CIFAR-10/BackdoorBench setup and GPU time.",
    )
    eval_p.add_argument("--out-dir", default="./evaluation", help="Where to write report.json/report.md/figures.")
    eval_p.add_argument("--n-independent", type=int, default=40,
                         help="Total independent (lineage-negative) models; split in half calibration/test.")
    eval_p.add_argument("--n-benign", type=int, default=40,
                         help="Total benign derivatives (fine-tune/pruned/quantized mix); split in half calibration/test.")
    eval_p.add_argument("--n-backdoored-per-trigger", type=int, default=10,
                         help="Backdoored derivatives PER trigger type (patch/blended/warped).")
    eval_p.add_argument("--alpha", type=float, default=0.05, help="Conformal alpha for both lineage and envelope tests.")
    eval_p.add_argument("--target-fpr", type=float, default=0.05, help="FPR to report backdoor TPR at, in addition to the conformal threshold.")
    eval_p.set_defaults(func=_cmd_evaluate)

    adaptive_p = sub.add_parser(
        "adaptive",
        help="Run the A4 adaptive-attacker pilot (Section 2 / Experiment E7).",
        description="Trains poisoned derivatives whose loss adds lambda * envelope_score on top of the "
                     "usual poisoned cross-entropy loss -- an attacker who knows exactly what the envelope "
                     "test measures and optimizes directly against it. Reports detection rate and attack "
                     "success rate retained at each lambda, including the honest failure case at high lambda.",
    )
    adaptive_p.add_argument("--out-dir", default="./adaptive", help="Where to write report.json/report.md/figure.")
    adaptive_p.add_argument("--n-benign-calibration", type=int, default=20,
                             help="Benign derivatives used to set the envelope conformal threshold.")
    adaptive_p.add_argument("--lambdas", type=float, nargs="+", default=[0.0, 0.5, 2.0, 5.0],
                             help="Penalty weights to sweep (0.0 = non-adaptive baseline).")
    adaptive_p.add_argument("--n-per-lambda", type=int, default=8, help="Backdoored derivatives trained per lambda.")
    adaptive_p.add_argument("--alpha", type=float, default=0.05, help="Conformal alpha for the envelope threshold.")
    adaptive_p.add_argument("--trigger-type", choices=["patch", "blended", "warped"], default="patch")
    adaptive_p.set_defaults(func=_cmd_adaptive)

    perm_p = sub.add_parser(
        "permutation-check",
        help="Test whether S_w survives an exact, function-preserving channel permutation.",
        description="Section 3.2 claims S_w 'resists simple permutation ... obfuscation'. This "
                     "permutes a genuine derivative's output channels (single layer, then the whole "
                     "network cascading), verifies each permutation is EXACTLY function-preserving "
                     "(forward-pass diff), and reports whether S_w actually survives it.",
    )
    perm_p.add_argument("--out-dir", default="./permutation_check", help="Where to write report.json/report.md.")
    perm_p.add_argument("--base-seed", type=int, default=1)
    perm_p.add_argument("--derivative-seed", type=int, default=42)
    perm_p.set_defaults(func=_cmd_permutation_check)

    scope_p = sub.add_parser(
        "lineage-scope-check",
        help="Splice-depth sweep and spectral ablation for the lineage rule.",
        description="Replaces a genuine fine-tune's last k weight matrices with an unrelated model's "
                     "(k=1..4) and reports how many splices the median-only rule, the full rule, and the "
                     "full rule with declared replacements admit; then ablates the spectral component.",
    )
    scope_p.add_argument("--out-dir", default="./lineage_scope_check", help="Where to write report.json/report.md.")
    scope_p.add_argument("--max-exempt", type=int, default=1)
    scope_p.set_defaults(func=_cmd_lineage_scope_check)

    dc_p = sub.add_parser(
        "defensibility-check",
        help="Splice patterns, S_w-adaptive and joint attacks, declared-head backdoor, benign sweeps, "
             "same-init siblings, determinism and scaling.",
    )
    dc_p.add_argument("--out-dir", default="./defensibility_check", help="Where to write report.json/report.md.")
    dc_p.add_argument("--n-adaptive", type=int, default=4, help="Models per adaptive-attack cell.")
    dc_p.set_defaults(func=_cmd_defensibility_check)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
