"""SHA-256 digests over model weights and JSON documents.

Shared by `oms.py` (building demo OMS bundles) and `pipeline.py`
(binding the in-toto attestation's `subject` to the same bytes an OMS
manifest would cover).
"""

from __future__ import annotations

import hashlib
import json


def weight_digest(model) -> str:
    """SHA-256 hex digest over a torch model's full state: every parameter
    AND buffer, sorted by name, each with its name, dtype and shape.
    Deterministic across runs for the same weights.

    Buffers matter: an earlier version hashed parameters only, so editing a
    BatchNorm layer's running mean/variance after admission left the digest
    -- and therefore the attestation's subject -- unchanged while changing
    13.5% of the model's predictions (measured on a demo fine-tune; see
    tests/test_end_to_end_attacks.py). Anything that changes eval-mode
    behaviour must be inside the digest.
    """
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items(), key=lambda kv: kv[0]):
        arr = t.detach().cpu().contiguous().numpy()
        h.update(f"{name}|{arr.dtype.str}|{arr.shape}|".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def json_digest(obj) -> str:
    """SHA-256 hex digest over a canonical (sorted-key) JSON encoding."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def probe_pool_digest(probe_loader) -> str:
    """SHA-256 hex digest over every tensor a probe DataLoader yields, in
    iteration order. Used as `probe.poolDigest` in the attestation so a
    verifier can confirm which probe pool a score was computed against.
    """
    h = hashlib.sha256()
    for images, labels in probe_loader:
        h.update(images.detach().cpu().contiguous().numpy().tobytes())
        h.update(labels.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
