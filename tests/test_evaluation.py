"""Tests for the engineering-validation pilot evaluation harness
(securemodelgate.evaluation / securemodelgate.figures). Uses a small
config so this stays fast; `securemodelgate evaluate`'s CLI defaults are
larger and meant to be run manually, not on every test invocation.
"""

import json

import pytest

pytest.importorskip("torch")

from securemodelgate.evaluation import EvaluationConfig, render_markdown_report, run_evaluation


@pytest.fixture(scope="module")
def small_report():
    config = EvaluationConfig(
        n_independent_calibration=6, n_independent_test=6,
        n_benign_calibration=6, n_benign_test=6,
        n_backdoored_per_trigger=3,
    )
    return run_evaluation(config, verbose=False)


def test_report_metrics_are_well_formed(small_report):
    r = small_report
    assert 0.0 <= r.false_block_rate.point_estimate <= 1.0
    assert r.false_block_rate.ci_low <= r.false_block_rate.point_estimate <= r.false_block_rate.ci_high
    assert 0.0 <= r.backdoor_tpr_at_conformal_threshold.point_estimate <= 1.0
    assert 0.0 <= r.backdoor_tpr_at_target_fpr <= 1.0
    assert 0.0 <= r.envelope_auroc <= 1.0
    assert 0.0 <= r.lineage_accuracy.point_estimate <= 1.0
    assert r.median_pipeline_latency_s > 0.0


def test_trigger_breakdown_covers_all_trigger_types(small_report):
    from securemodelgate.demo_models import TRIGGER_TYPES
    names = {tb.trigger_type for tb in small_report.trigger_breakdown}
    assert names == set(TRIGGER_TYPES)
    for tb in small_report.trigger_breakdown:
        assert tb.n == 3
        assert 0 <= tb.tp <= tb.n
        assert 0.0 <= tb.mean_asr <= 1.0


def test_backdoors_separate_reasonably_well_from_benign(small_report):
    # A real, if weak-by-construction (small n), sanity check that the
    # envelope score carries SOME signal, not just plausible-looking
    # ranges: backdoors implanted with >85% ASR should score noticeably
    # higher on average than held-out benign derivatives.
    r = small_report
    benign = [s for s, y in zip(r.envelope_scores, r.envelope_score_labels) if y == 0]
    backdoored = [s for s, y in zip(r.envelope_scores, r.envelope_score_labels) if y == 1]
    assert sum(backdoored) / len(backdoored) > sum(benign) / len(benign)


def test_report_json_round_trips(small_report):
    payload = json.dumps(small_report.to_dict(), default=str)
    parsed = json.loads(payload)
    assert parsed["envelope_auroc"] == pytest.approx(small_report.envelope_auroc)
    assert len(parsed["trigger_breakdown"]) == len(small_report.trigger_breakdown)


def test_markdown_report_names_every_trigger_type(small_report):
    md = render_markdown_report(small_report)
    assert "engineering validation pilot" in md.lower()
    for tb in small_report.trigger_breakdown:
        assert tb.trigger_type in md


def test_figures_are_written(tmp_path, small_report):
    from securemodelgate import figures
    paths = figures.write_all_figures(small_report, tmp_path)
    assert len(paths) == 3
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 0
