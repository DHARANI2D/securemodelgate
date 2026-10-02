"""
securemodelgate.lineage — Signed Behavioral-Lineage Attestation
==================================================================

Implements Section 3 of the revised paper ("SecureModelGate: Signed
Behavioral-Lineage Attestation for Admitting Third-Party AI Models",
`paper/techcon2027_revision/02_paper.md`). It replaces the Parasparam
2026 prototype's undefined "clean reference" fingerprint and
hand-picked KL threshold with:

  1. reference_resolver  — resolve F_ref from a declared, OMS-signed base
                            model (ML-BOM pedigree.ancestors), not an
                            averaged "clean twin" set.               (3.1)
  2. lineage_score        — S_w: spectral + directional weight agreement
                            with the declared base.                  (3.2)
  3. behavioral_delta     — D_b: layer-wise linear CKA + output JS
                            divergence on a per-admission probe set.  (3.2)
  4. conformal            — split-conformal thresholds (replaces τ).  (3.3)
  5. intoto_attestation   — signed in-toto predicate (replaces the
                            RSA/JWT "MAT").                           (3.4)
  6. admission_policy     — Tier 0 / 1 / 2 risk-tiered routing.       (3.5)

These modules are pulled together into one callable pipeline by
`securemodelgate.pipeline.run_admission`, exercised end-to-end (real
torch models, real forward hooks, real signing) by
`securemodelgate.cli`'s `demo` and `admit` subcommands and by
`tests/test_pipeline.py`.

The original Parasparam 2026 prototype
(`from_Downloads_alt_code/src/static.py` / `fingerprint.py` /
`attestation.py` / `evaluator.py`) is left untouched — it is what
produced the retained Parasparam 2026 numbers (20/20 detection, 3/20
false blocks) cited in the revision as prior results, not something
this package supersedes silently.
"""
