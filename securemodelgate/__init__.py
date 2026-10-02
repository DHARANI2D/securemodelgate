"""
securemodelgate — Signed Behavioral-Lineage Attestation for Admitting
Third-Party AI Models.

Top-level package. See `securemodelgate.lineage` for the scoring/
calibration/attestation building blocks (Section 3 of
`paper/techcon2027_revision/02_paper.md`), `securemodelgate.pipeline`
for how they're wired into one end-to-end admission decision, and run
`securemodelgate demo` (after `pip install -e .`) for a runnable,
real-torch-models walkthrough.
"""

__version__ = "0.3.0"
