"""Tests for securemodelgate.permutation_check (the permutation-robustness
pilot) and its CLI wiring."""

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from securemodelgate.permutation_check import render_markdown_report, run_permutation_check


def test_report_is_well_formed_and_honest_about_the_gap():
    report = run_permutation_check(verbose=False)

    # Both permutations must be verified function-preserving (floating-point
    # noise only) -- this is the whole premise of the experiment.
    assert report.single_layer_functional_diff < 1e-3
    assert report.full_network_functional_diff < 1e-3

    # Single-layer case: fully recovered.
    assert report.s_w_single_layer_permuted == pytest.approx(report.s_w_unpermuted, abs=0.02)

    # Full-network cascading case: fully recovered by sequential alignment
    # (it was ~0.60 before each layer's channel correspondence was carried
    # into the next layer's input columns), and far above the
    # independent-model range.
    lo, hi = report.independent_model_s_w_range
    assert report.s_w_full_network_permuted > hi
    assert report.s_w_full_network_permuted == pytest.approx(report.s_w_unpermuted, abs=0.02)


def test_markdown_report_names_both_scenarios():
    report = run_permutation_check(verbose=False)
    md = render_markdown_report(report)
    assert "Single layer permuted" in md
    assert "Every layer permuted" in md
    assert "Scope limit" in md   # the residual/transformer caveat must stay in the report


def test_cli_writes_report(tmp_path):
    out_dir = tmp_path / "permutation_check"
    result = subprocess.run(
        [sys.executable, "-m", "securemodelgate.cli", "permutation-check", "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "report.json").exists()
    assert (out_dir / "report.md").exists()
    payload = json.loads((out_dir / "report.json").read_text())
    assert "s_w_unpermuted" in payload
