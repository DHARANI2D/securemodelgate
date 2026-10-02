"""Where does the lineage rule stop working? Two scope tests on the
synthetic corpus, both run through the exact rule the gate enforces
(`pipeline.lineage_verdict_for`: median S_w AND per-layer floor AND the
exemption cap):

1. Splice depth. Take a genuine fine-tune and replace its last k weight
   matrices (k = 1..4, head first) with those of an unrelated model, for
   several unrelated donors. Each splice is scored three ways: median only
   (the pre-floor rule), the full rule, and the full rule with the replaced
   matrices declared in the ML-BOM (exempt, but capped).

2. Spectral ablation. Score genuine derivatives, held-out unrelated models
   and splices with spectral_weight in {0, 0.5, 1} (cosine only, the
   default blend, spectra only) and report how well each separates them.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import demo_models
from .lineage.conformal import lineage_test
from .lineage.lineage_score import DEFAULT_SPECTRAL_WEIGHT, pooled_layer_scores, weight_lineage_score_from_models
from .permutation_check import _CASCADE, permute_layer_output_channels
from .pipeline import DEFAULT_MAX_EXEMPT, lineage_verdict_for
from .stats import auroc

# Trailing modules of TinyConvNet, head first, and the weight matrix each owns.
_TAIL = [("fc", "fc.weight"), ("layer4", "layer4.conv.weight"),
         ("layer3", "layer3.conv.weight"), ("layer2", "layer2.conv.weight")]


@dataclass
class SpliceRow:
    k_foreign: int
    n: int
    admitted_median_only: int
    admitted_full_rule: int
    admitted_declared: int


@dataclass
class AblationRow:
    spectral_weight: float
    auroc_median: float
    auroc_min_layer: float
    worst_genuine_min_layer: float
    best_independent_min_layer: float
    genuine_admitted: str
    independent_admitted: str
    splice_admitted: str


@dataclass
class ScopeReport:
    max_exempt: int
    alpha: float
    splice: list[SpliceRow] = field(default_factory=list)
    ablation: list[AblationRow] = field(default_factory=list)


def _splice(genuine, donor, k):
    out = copy.deepcopy(genuine)
    for module, _ in _TAIL[:k]:
        getattr(out, module).load_state_dict(getattr(donor, module).state_dict())
    return out


def _genuine_set(base):
    ft = demo_models.make_benign_fine_tune(base, seed=42)
    permuted = ft
    for i, (layer, nxt) in enumerate(_CASCADE):
        permuted = permute_layer_output_channels(permuted, layer, nxt, seed=i + 1)
    return ([demo_models.make_benign_fine_tune(base, seed=300 + i) for i in range(6)]
            + [demo_models.make_pruned_derivative(base, a) for a in (0.1, 0.3, 0.5)]
            + [demo_models.make_quantized_derivative(base, lv) for lv in (4, 8, 16)]
            + [permuted])


def run_lineage_scope_check(n_calibration: int = 20, n_donors: int = 5, alpha: float = 0.05,
                             max_exempt: int = DEFAULT_MAX_EXEMPT, verbose: bool = True) -> ScopeReport:
    log = print if verbose else (lambda *a, **k: None)
    base = demo_models.make_base_model(seed=1)
    calibration_models = [demo_models.make_independent_model(seed=100 + i) for i in range(n_calibration)]
    donors = [demo_models.make_independent_model(seed=5000 + i) for i in range(n_donors)]
    held_out = [demo_models.make_independent_model(seed=6000 + i) for i in range(20)]
    genuine = _genuine_set(base)
    ft = demo_models.make_benign_fine_tune(base, seed=42)
    report = ScopeReport(max_exempt=max_exempt, alpha=alpha)

    def calibrate(sw):
        dicts = [weight_lineage_score_from_models(m, base, spectral_weight=sw) for m in calibration_models]
        return [d["S_w"] for d in dicts], pooled_layer_scores(dicts)

    def admitted(score, cal, declared_score=None):
        return lineage_verdict_for(declared_score or score, cal[0], cal[1], alpha, max_exempt)[0].exceeds_threshold

    log("Splice-depth sweep ...")
    sw = DEFAULT_SPECTRAL_WEIGHT   # calibrate and score with the SAME weight the gate uses
    cal = calibrate(sw)
    for k in range(1, len(_TAIL) + 1):
        declared = [name for _, name in _TAIL[:k]]
        row = SpliceRow(k_foreign=k, n=len(donors), admitted_median_only=0, admitted_full_rule=0, admitted_declared=0)
        for donor in donors:
            spliced = _splice(ft, donor, k)
            s = weight_lineage_score_from_models(spliced, base, spectral_weight=sw)
            s_decl = weight_lineage_score_from_models(spliced, base, spectral_weight=sw, declared_replaced=declared)
            row.admitted_median_only += lineage_test(s["S_w"], cal[0], alpha_l=alpha).exceeds_threshold
            row.admitted_full_rule += admitted(s, cal)
            row.admitted_declared += admitted(s_decl, cal)
        report.splice.append(row)
        log(f"  k={k}: median-only {row.admitted_median_only}/{row.n}, full rule {row.admitted_full_rule}/{row.n}, "
            f"declared {row.admitted_declared}/{row.n}")

    log("Spectral ablation ...")
    splices = [_splice(ft, d, 3) for d in donors]
    for sw in (0.0, 0.5, 1.0):
        cal = calibrate(sw)
        g = [weight_lineage_score_from_models(m, base, spectral_weight=sw) for m in genuine]
        ind = [weight_lineage_score_from_models(m, base, spectral_weight=sw) for m in held_out]
        sp = [weight_lineage_score_from_models(m, base, spectral_weight=sw) for m in splices]
        labels = [1] * len(g) + [0] * len(ind)
        row = AblationRow(
            spectral_weight=sw,
            auroc_median=auroc(labels, [d["S_w"] for d in g + ind]),
            auroc_min_layer=auroc(labels, [d["min_layer_score"] for d in g + ind]),
            worst_genuine_min_layer=min(d["min_layer_score"] for d in g),
            best_independent_min_layer=max(d["min_layer_score"] for d in ind),
            genuine_admitted=f"{sum(admitted(d, cal) for d in g)}/{len(g)}",
            independent_admitted=f"{sum(admitted(d, cal) for d in ind)}/{len(ind)}",
            splice_admitted=f"{sum(admitted(d, cal) for d in sp)}/{len(sp)}",
        )
        report.ablation.append(row)
        log(f"  spectral_weight={sw}: {row}")
    return report


def render_markdown_report(report: ScopeReport) -> str:
    lines = [
        "# SecureModelGate — lineage-rule scope check",
        "",
        "Synthetic corpus (`demo_models.TinyConvNet`), not E1–E9. Every decision uses "
        f"`pipeline.lineage_verdict_for` (median S_w AND per-layer floor AND at most "
        f"{report.max_exempt} unscored matrix), alpha={report.alpha}.",
        "",
        "## 1. Splice depth",
        "",
        "A genuine fine-tune with its last k weight matrices (head first) replaced by an "
        "unrelated model's. Cells count splices ADMITTED by lineage (lower is better).",
        "",
        "| k foreign matrices | n | median-only rule | full rule | full rule, replacements declared |",
        "|---|---|---|---|---|",
    ]
    for r in report.splice:
        lines.append(f"| {r.k_foreign} | {r.n} | {r.admitted_median_only} | {r.admitted_full_rule} | {r.admitted_declared} |")
    lines += [
        "",
        "Declaring k <= max_exempt matrices replaced is admitted by design: that is exactly "
        "ordinary transfer learning with a new head, and lineage alone cannot tell it from a "
        "backdoored head. The envelope test is the only remaining check on that case.",
        "",
        "## 2. Spectral ablation",
        "",
        "| spectral weight | AUROC (median S_w) | AUROC (min layer) | worst genuine min-layer | "
        "best unrelated min-layer | genuine admitted | unrelated admitted | k=3 splices admitted |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in report.ablation:
        lines.append(f"| {r.spectral_weight} | {r.auroc_median:.3f} | {r.auroc_min_layer:.3f} | "
                     f"{r.worst_genuine_min_layer:.3f} | {r.best_independent_min_layer:.3f} | "
                     f"{r.genuine_admitted} | {r.independent_admitted} | {r.splice_admitted} |")
    lines.append("")
    return "\n".join(lines)
