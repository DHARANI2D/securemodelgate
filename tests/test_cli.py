"""Smoke tests for the `securemodelgate` CLI end to end (subprocess),
covering both subcommands and their exit codes / expected output shape.
"""

import json
import subprocess
import sys

import pytest

pytest.importorskip("torch")


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "securemodelgate.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_demo_writes_attestations_for_every_scenario(tmp_path):
    out_dir = tmp_path / "attestations"
    result = _run("demo", "--out-dir", str(out_dir))

    assert result.returncode == 0, result.stderr
    for name in ("trusted_publisher", "unsigned_import", "benign_fine_tune",
                 "pruned_derivative", "poisoned_derivative", "lineage_forged"):
        path = out_dir / f"{name}.attestation.json"
        assert path.exists(), f"missing attestation for {name}\nstdout:\n{result.stdout}"
        json.loads(path.read_text())  # must be valid JSON

    # The two scenarios the pipeline design guarantees regardless of the
    # (seeded, but still stochastic-in-principle) training runs:
    assert "lineage_forged" in result.stdout and "BLOCK" in result.stdout
    assert "trusted_publisher" in result.stdout and "ADMIT" in result.stdout


def test_verify_round_trips_a_demo_attestation(tmp_path):
    out_dir = tmp_path / "attestations"
    demo_result = _run("demo", "--out-dir", str(out_dir))
    assert demo_result.returncode == 0, demo_result.stderr

    verify_result = _run("verify", str(out_dir / "benign_fine_tune.attestation.json"))
    assert verify_result.returncode == 0, verify_result.stderr
    assert "VALID signature." in verify_result.stdout


def test_verify_rejects_wrong_secret(tmp_path):
    out_dir = tmp_path / "attestations"
    demo_result = _run("demo", "--out-dir", str(out_dir))
    assert demo_result.returncode == 0, demo_result.stderr

    verify_result = _run(
        "verify", str(out_dir / "benign_fine_tune.attestation.json"), "--secret", "wrong-secret",
    )
    assert verify_result.returncode == 1
    assert "INVALID" in verify_result.stderr
