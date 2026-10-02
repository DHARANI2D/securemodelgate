"""
Reference Resolution (Section 3.1)
===================================
Fixes the undefined-F_ref flaw the Parasparam 2026 prototype had: instead
of averaging fingerprints over an ad-hoc "clean model" set, F_ref is the
*declared, OMS-signed base model* named in the candidate's ML-BOM.

    1. The candidate model m ships a CycloneDX ML-BOM whose
       machine-learning-model component lists base b under
       `pedigree.ancestors`, identified by purl + OMS bundle digest.
    2. We verify b's OMS signature against an allow-list of publisher
       identities.
    3. The verified b becomes F_ref (its weights/activations are what
       lineage_score.py and behavioral_delta.py compare m against).
    4. If no signed ancestor is declared, admission_policy.py routes m
       to the Tier 2 deep-scan lane instead of computing a lineage score.

NOTE ON SCOPE: real OMS verification means checking a Sigstore bundle's
DSSE envelope against Fulcio-issued certificates and a Rekor transparency
log entry (see the `model-signing` / `sigstore-python` packages). Neither
is vendored here. `verify_oms_signature` below is a structural stand-in:
it checks bundle shape, a signer allow-list, and digest well-formedness,
and is clearly not a cryptographic verification. Swapping in real
Sigstore verification is a drop-in replacement for this one function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass
class OMSBundle:
    """Structural view of an OpenSSF Model Signing (OMS) Sigstore bundle.

    Real bundles are a DSSE envelope wrapping an in-toto Statement whose
    subjects are file digests, plus the Sigstore verification material
    (certificate chain + Rekor log entry). This dataclass only carries
    the fields this prototype's stand-in check needs.
    """
    signer_identity: str
    bundle_digest: str          # sha256 of the bundle file, hex
    subject_digest: Optional[str] = None   # sha256 of the signed model file(s)


@dataclass
class ReferenceResolution:
    verified: bool
    base_purl: Optional[str] = None
    oms_bundle_digest: Optional[str] = None
    signer_identity: Optional[str] = None
    reason: str = ""


def verify_oms_signature(bundle: OMSBundle, allowed_signers: set[str]) -> tuple[bool, str]:
    """Structural stand-in for real OMS/Sigstore verification.

    A production gate calls into `model_signing`/`sigstore-python` here to
    check the DSSE signature against Fulcio certs + Rekor inclusion. This
    stand-in only checks: (a) the signer is on the allow-list, and
    (b) the digest fields are well-formed sha256 hex. It deliberately does
    NOT prove authenticity — it exists so the rest of the pipeline
    (reference resolution → lineage scoring → attestation) has a stable
    interface to develop and test against before that integration lands.
    """
    if bundle.signer_identity not in allowed_signers:
        return False, f"signer '{bundle.signer_identity}' not in allow-list"
    if not _SHA256_HEX_RE.match(bundle.bundle_digest):
        return False, "malformed bundle digest"
    if bundle.subject_digest is not None and not _SHA256_HEX_RE.match(bundle.subject_digest):
        return False, "malformed subject digest"
    return True, "ok (structural check only — not a cryptographic verification)"


def _first_signed_ancestor(mlbom: dict) -> Optional[dict]:
    """Pull the first ancestor entry from a CycloneDX-style ML-BOM.

    Expected shape (CycloneDX 1.6/1.7 machine-learning-model component):

        {
          "component": {
            "type": "machine-learning-model",
            "pedigree": {
              "ancestors": [
                {
                  "purl": "pkg:huggingface/org/base-model@rev",
                  "properties": [
                    {"name": "securemodelgate:omsBundleDigest", "value": "<sha256 hex>"},
                    {"name": "securemodelgate:signerIdentity",  "value": "ngc-catalog"}
                  ]
                }
              ]
            }
          }
        }
    """
    ancestors = (
        mlbom.get("component", {})
             .get("pedigree", {})
             .get("ancestors", [])
    )
    return ancestors[0] if ancestors else None


def _property(entry: dict, name: str) -> Optional[str]:
    for prop in entry.get("properties", []):
        if prop.get("name") == name:
            return prop.get("value")
    return None


REPLACED_PARAMETERS_PROPERTY = "securemodelgate:replacedParameters"


def declared_replaced_parameters(mlbom: dict) -> list[str]:
    """Weight-matrix names the candidate's ML-BOM declares as replaced
    (e.g. a re-initialised classification head in ordinary transfer
    learning), as a comma-separated `securemodelgate:replacedParameters`
    component property. This is a SecureModelGate convention, not a
    CycloneDX-standard field, and it is the candidate's own claim — so the
    lineage scorer exempts these from the per-layer floor only up to the
    admission policy's `max_exempt` cap, and the attestation records them.
    """
    for prop in mlbom.get("component", {}).get("properties", []):
        if prop.get("name") == REPLACED_PARAMETERS_PROPERTY:
            return [p.strip() for p in str(prop.get("value", "")).split(",") if p.strip()]
    return []


def resolve_reference(mlbom: dict, allowed_signers: set[str],
                      trusted_bases: Optional[Mapping[str, str]] = None) -> ReferenceResolution:
    """Resolve and verify the declared base model for a candidate's ML-BOM.

    Returns a ReferenceResolution. `verified=False` means: no declared
    ancestor, or the declared ancestor's OMS signature did not check out.
    Either way, admission_policy.py should route the candidate to Tier 2
    (deep scan) rather than attempt a lineage/envelope test against an
    unverified reference.

    `trusted_bases` maps base digest -> the signer identity that signed it:
    the operator's registry of bases it has itself verified (with real
    Sigstore, the result of verifying each base's bundle once at import).
    Without it, the structural stand-in above accepts ANY well-formed
    digest whose claimed signer is on the allow-list -- so a candidate can
    name an allowed signer over a base that signer never signed (signer
    substitution). With it, the digest must be registered and the claimed
    signer must be the one that signed it; removing a digest revokes it.
    Only the first ancestor is read: multi-parent (merged) models are out
    of scope and must be declared as a single signed base.
    """
    ancestor = _first_signed_ancestor(mlbom)
    if ancestor is None:
        return ReferenceResolution(verified=False, reason="no declared ancestor in ML-BOM pedigree")

    purl = ancestor.get("purl")
    bundle_digest = _property(ancestor, "securemodelgate:omsBundleDigest")
    signer = _property(ancestor, "securemodelgate:signerIdentity")

    if not purl or not bundle_digest or not signer:
        return ReferenceResolution(
            verified=False,
            base_purl=purl,
            reason="declared ancestor missing purl / omsBundleDigest / signerIdentity",
        )

    bundle = OMSBundle(signer_identity=signer, bundle_digest=bundle_digest)
    ok, reason = verify_oms_signature(bundle, allowed_signers)
    if ok and trusted_bases is not None:
        registered = trusted_bases.get(bundle_digest)
        if registered is None:
            ok, reason = False, "declared base digest is not in the trusted-base registry (never verified, or revoked)"
        elif registered != signer:
            ok, reason = False, (f"signer substitution: the registry records '{registered}' as this base's "
                                 f"signer, the ML-BOM claims '{signer}'")
        else:
            reason = "ok (digest and signer match the trusted-base registry)"
    return ReferenceResolution(
        verified=ok,
        base_purl=purl,
        oms_bundle_digest=bundle_digest,
        signer_identity=signer,
        reason=reason,
    )
