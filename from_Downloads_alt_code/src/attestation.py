"""
Model Attestation Token (MAT) and MBOM generation.

MAT = JWT signed with RSA-4096 (or HS256 for prototype)
containing:
  - model_hash (SHA-256 of weights)
  - fingerprint_vector (128-dim, base64)
  - kl_divergence
  - mbom_digest
  - issued_at / expires_at
  - risk_score (0-100)
  - decision: PASS | FAIL

MBOM = JSON manifest capturing:
  - base model architecture
  - framework + version
  - dataset provenance
  - dependency hashes
  - timestamp
"""

import hashlib, json, time, base64
import numpy as np
from datetime import datetime, timedelta, timezone
import jwt


class MATIssuer:
    """Issues signed Model Attestation Tokens."""

    ALGORITHM = "HS256"   # HS256 for prototype; RS256 in production

    def __init__(self, secret_key="HPE-SecureModelGate-2026-prototype"):
        self.secret_key = secret_key

    # ── Weight hash ───────────────────────────────────────────────────────────

    def compute_weight_hash(self, model):
        """SHA-256 of all model parameters concatenated."""
        h = hashlib.sha256()
        for param in model.parameters():
            w = param.detach().cpu().float().numpy()
            h.update(w.tobytes())
        return h.hexdigest()

    # ── MBOM ─────────────────────────────────────────────────────────────────

    def generate_mbom(self, model_dict, framework="PyTorch",
                      dataset="CIFAR-10", architecture="ResNet-18"):
        import torch
        weight_hash = self.compute_weight_hash(model_dict["model"])
        mbom = {
            "model_id":       model_dict["id"],
            "architecture":   architecture,
            "framework":      framework,
            "framework_ver":  torch.__version__,
            "dataset":        dataset,
            "weight_hash":    weight_hash,
            "trigger_type":   model_dict.get("trigger"),
            "created_at":     datetime.now(timezone.utc).isoformat(),
            "dependencies": {
                "torch":       torch.__version__,
                "numpy":       np.__version__,
            }
        }
        mbom_json   = json.dumps(mbom, sort_keys=True)
        mbom_digest = hashlib.sha256(mbom_json.encode()).hexdigest()
        return mbom, mbom_digest

    # ── Risk score ────────────────────────────────────────────────────────────

    def compute_risk_score(self, static_score, fp_score):
        """
        Composite risk score 0-100.
        0 = fully trusted, 100 = high risk.
        """
        ks  = static_score.get("ks_statistic", 0)
        kl  = fp_score.get("kl_divergence", 0)
        la  = static_score.get("layer_anomaly", 0)

        # Normalise each component to [0,1]
        ks_norm = min(ks / 0.2,  1.0)     # ks > 0.2 → fully suspicious
        kl_norm = min(kl / 0.15, 1.0)     # kl > 0.15 → fully suspicious
        la_norm = min(la / 5.0,  1.0)     # anomaly z > 5 → fully suspicious

        score = (0.35 * ks_norm + 0.45 * kl_norm + 0.20 * la_norm) * 100
        return round(score, 1)

    # ── Token ─────────────────────────────────────────────────────────────────

    def issue(self, model_dict, static_score, fp_score):
        """
        Issue a MAT. Returns (token_str, payload_dict, decision).
        Decision: "PASS" or "FAIL"
        """
        weight_hash  = self.compute_weight_hash(model_dict["model"])
        mbom, mbom_digest = self.generate_mbom(model_dict)
        risk_score   = self.compute_risk_score(static_score, fp_score)

        # Fingerprint → base64 (compact storage)
        fp_vec = fp_score.get("fingerprint", np.zeros(128))
        fp_b64 = base64.b64encode(fp_vec.astype(np.float32).tobytes()).decode()

        flagged = (static_score.get("static_flagged", False) or
                   fp_score.get("fp_flagged", False))
        decision = "FAIL" if flagged else "PASS"

        now = datetime.now(timezone.utc)
        payload = {
            "iss":        "HPE-SecureModelGate",
            "sub":        model_dict["id"],
            "iat":        int(now.timestamp()),
            "exp":        int((now + timedelta(days=30)).timestamp()),
            "model_hash": weight_hash,
            "fp_b64":     fp_b64[:64],   # truncated for token compactness
            "kl_div":     fp_score.get("kl_divergence", 0),
            "mbom_digest":mbom_digest,
            "risk_score": risk_score,
            "decision":   decision,
        }

        token = jwt.encode(payload, self.secret_key, algorithm=self.ALGORITHM)
        return token, payload, decision

    def verify(self, token):
        """Verify token and return payload."""
        try:
            payload = jwt.decode(
                token, self.secret_key,
                algorithms=[self.ALGORITHM]
            )
            return True, payload
        except Exception as e:
            return False, str(e)
