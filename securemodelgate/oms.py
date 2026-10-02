"""Demo OMS bundle + ML-BOM construction.

Real OMS bundles are Sigstore-signed DSSE envelopes issued by
`model-signing`/`sigstore-python` against a publisher's Fulcio identity;
real ML-BOMs are CycloneDX documents a build pipeline emits. Neither is
vendored here (see `securemodelgate.lineage.reference_resolver`'s
module docstring). This module builds structurally valid stand-ins —
matching exactly the shape `reference_resolver.resolve_reference`
expects — so the CLI demo and `pipeline.run_admission` can be exercised
end-to-end without a real Sigstore deployment.
"""

from __future__ import annotations

from .digest import weight_digest


def build_oms_bundle_digest(base_model, signer_identity: str) -> dict:
    """Structural stand-in for an OMS Sigstore bundle over `base_model`'s
    weights. Returns the fields `reference_resolver.OMSBundle` needs.
    """
    return {
        "signerIdentity": signer_identity,
        "bundleDigest": weight_digest(base_model),
        "subjectDigest": weight_digest(base_model),
    }


def build_mlbom(base_purl: str, base_model, signer_identity: str) -> dict:
    """CycloneDX-shaped ML-BOM declaring `base_model` as the candidate's
    sole pedigree ancestor. See `reference_resolver._first_signed_ancestor`
    for the exact shape this satisfies.
    """
    bundle = build_oms_bundle_digest(base_model, signer_identity)
    return {
        "component": {
            "type": "machine-learning-model",
            "pedigree": {
                "ancestors": [
                    {
                        "purl": base_purl,
                        "properties": [
                            {"name": "securemodelgate:omsBundleDigest", "value": bundle["bundleDigest"]},
                            {"name": "securemodelgate:signerIdentity", "value": signer_identity},
                        ],
                    }
                ]
            },
        }
    }


def build_unsigned_mlbom() -> dict:
    """An ML-BOM with no declared ancestor at all (routes to Tier 2)."""
    return {"component": {"type": "machine-learning-model"}}
