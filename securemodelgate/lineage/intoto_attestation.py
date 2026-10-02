"""
Signed In-Toto Attestation (Section 3.4)
==========================================
Replaces the RSA-4096/HS256 JWT "MAT" and bespoke JSON "MBOM" with a
standard, independently-verifiable pair:

  - subjects: the same file digests the model's OMS manifest covers.
  - predicateType: "https://securemodelgate.hpe.com/lineage/v0.1"
    (in-toto predicate types are not centrally registered; custom types
    are explicitly permitted by the spec).
  - predicate: declaredBase / mlbomDigest / lineage / envelope / probe /
    tier / deepScan / verdict / gateVersion / timestamp, matching the
    fields listed in the revised paper's Section 3.4.

Current `model-signing` tooling (1.1.1) documents no option to attach a
custom predicate INSIDE an OMS bundle, so this attestation ships
alongside the OMS bundle as its own DSSE envelope, not embedded in it.

SIGNING: production signing is keyless Sigstore (cluster-connected) or a
private Sigstore instance / KMS key (air-gapped HPE Private Cloud AI).
Neither is vendored here. `LocalHmacSigner` below is a structural
stand-in with the same DSSE envelope shape
(`{"payload", "payloadType", "signatures": [{"keyid", "sig"}]}`) so the
admission-policy and predicate-building code can be developed and unit
tested now; swapping in real Sigstore signing means replacing
`LocalHmacSigner.sign`/`.verify` only — the envelope shape and predicate
schema do not change.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

STATEMENT_TYPE = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE = "https://securemodelgate.hpe.com/lineage/v0.1"
PAYLOAD_TYPE = "application/vnd.in-toto+json"


def build_predicate(
    *,
    declared_base: dict,          # {purl, omsBundleDigest, signerIdentity}
    mlbom_digest: str,
    lineage: dict,                # {S_w, threshold, alpha_L, calibrationSetDigest}
    envelope: dict,                # {minCKA, JS, threshold, alpha, n}
    probe: dict,                   # {poolDigest, seedCommitment, seed}
    tier: str,                     # "0" | "1" | "2"
    deep_scan: Optional[dict],     # {tool, version, resultDigest} or None
    verdict: str,                  # "ADMIT" | "ESCALATE" | "BLOCK"
    gate_version: str = "0.1.0",
) -> dict:
    return {
        "declaredBase": declared_base,
        "mlbomDigest": mlbom_digest,
        "lineage": lineage,
        "envelope": envelope,
        "probe": probe,
        "tier": tier,
        "deepScan": deep_scan,
        "verdict": verdict,
        "gateVersion": gate_version,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def build_statement(subject_name: str, subject_sha256: str, predicate: dict) -> dict:
    """In-toto Statement v1 wrapping the SecureModelGate predicate.

    `subject_sha256` should be the SAME file digest recorded in the
    candidate model's OMS manifest, so a verifier can bind the two
    attestations to the same bytes.
    """
    return {
        "_type": STATEMENT_TYPE,
        "subject": [{"name": subject_name, "digest": {"sha256": subject_sha256}}],
        "predicateType": PREDICATE_TYPE,
        "predicate": predicate,
    }


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding: what a DSSE signature covers, so
    the payload TYPE is authenticated along with the payload bytes."""
    t = payload_type.encode()
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


@dataclass
class LocalHmacSigner:
    """Structural stand-in for keyless Sigstore / KMS signing.

    NOT a substitute for real Sigstore verification (no certificate
    chain, no Rekor transparency log entry, no third-party trust root).
    Exists so the DSSE envelope shape and the sign/verify call sites are
    exercised end-to-end before that integration lands.
    """
    key_id: str
    secret: bytes = field(repr=False)

    def sign(self, statement: dict) -> dict:
        payload_bytes = json.dumps(statement, sort_keys=True).encode()
        payload_b64 = base64.b64encode(payload_bytes).decode()
        sig = hmac.new(self.secret, pae(PAYLOAD_TYPE, payload_bytes), hashlib.sha256).hexdigest()
        return {
            "payloadType": PAYLOAD_TYPE,
            "payload": payload_b64,
            "signatures": [{"keyid": self.key_id, "sig": sig}],
        }

    def verify(self, envelope: dict) -> tuple[bool, Any]:
        try:
            payload_bytes = base64.b64decode(envelope["payload"], validate=True)
        except Exception as e:
            return False, f"undecodable payload: {e}"
        if envelope.get("payloadType") != PAYLOAD_TYPE:
            return False, f"unexpected payloadType {envelope.get('payloadType')!r}"

        expected = hmac.new(self.secret, pae(PAYLOAD_TYPE, payload_bytes), hashlib.sha256).hexdigest()
        sigs = envelope.get("signatures", [])
        matched = any(
            s.get("keyid") == self.key_id and hmac.compare_digest(s.get("sig", ""), expected)
            for s in sigs
        )
        if not matched:
            return False, "no signature matched key_id/secret"
        return True, json.loads(payload_bytes)


def issue_attestation(
    *,
    signer: LocalHmacSigner,
    subject_name: str,
    subject_sha256: str,
    declared_base: dict,
    mlbom_digest: str,
    lineage: dict,
    envelope: dict,
    probe: dict,
    tier: str,
    deep_scan: Optional[dict],
    verdict: str,
    gate_version: str = "0.1.0",
) -> dict:
    """Build + sign the full DSSE-wrapped in-toto statement in one call."""
    predicate = build_predicate(
        declared_base=declared_base, mlbom_digest=mlbom_digest,
        lineage=lineage, envelope=envelope, probe=probe,
        tier=tier, deep_scan=deep_scan, verdict=verdict, gate_version=gate_version,
    )
    statement = build_statement(subject_name, subject_sha256, predicate)
    return signer.sign(statement)
