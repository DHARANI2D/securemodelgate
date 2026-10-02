"""Loading an untrusted candidate's weights without executing its code.

The rest of the package scores in-memory `torch.nn.Module`s; this is the
boundary a real gate puts in front of them. Rules, each fail-closed
(`UnsafeModelFile` is raised, and the caller must deny):

  - `.safetensors` is loaded with `safetensors.torch.load_file`: the format
    is a JSON header plus raw tensor bytes, so parsing it runs no code.
  - `.pt` / `.pth` / `.bin` (pickle) are loaded ONLY with
    `torch.load(weights_only=True)`, which refuses any pickle opcode that
    would import or call something other than tensor/container types. A
    pickle that needs more than that is rejected, not loaded "carefully".
  - Every other suffix (`.pkl`, `.joblib`, `.h5`, `.keras`, `.onnx`, ...)
    is refused.
  - Symlinks and files over `max_bytes` are refused; every value must be a
    finite floating/integer tensor under a string key.
  - The architecture always comes from the gate (the declared base's own
    trusted model class), never from the candidate: there is no
    trust_remote_code path. `load_candidate` loads with strict=True, so a
    candidate whose names or shapes do not fit the base's architecture is
    rejected at load time -- which also means shape-manipulation attacks
    on the lineage scorer only arise for callers that bypass this loader.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

PICKLE_SUFFIXES = (".pt", ".pth", ".bin")
SAFE_SUFFIXES = (".safetensors",)
DEFAULT_MAX_BYTES = 8 * 1024 ** 3


class UnsafeModelFile(Exception):
    """The file was refused; the candidate must not be admitted."""


def load_state_dict(path, *, allow_pickle: bool = True, max_bytes: int = DEFAULT_MAX_BYTES) -> dict:
    p = Path(path)
    if p.is_symlink():
        raise UnsafeModelFile(f"{p.name}: symlinks are refused")
    if not p.is_file():
        raise UnsafeModelFile(f"{p.name}: not a regular file")
    if os.path.getsize(p) > max_bytes:
        raise UnsafeModelFile(f"{p.name}: larger than {max_bytes} bytes")
    suffix = p.suffix.lower()
    try:
        if suffix in SAFE_SUFFIXES:
            from safetensors.torch import load_file
            sd = load_file(str(p), device="cpu")
        elif suffix in PICKLE_SUFFIXES and allow_pickle:
            sd = torch.load(str(p), map_location="cpu", weights_only=True)
        else:
            raise UnsafeModelFile(f"{p.name}: format {suffix or '(none)'} is not accepted")
    except UnsafeModelFile:
        raise
    except Exception as e:
        raise UnsafeModelFile(f"{p.name}: refused by the safe loader ({type(e).__name__}: {e})") from e
    _validate(sd, p.name)
    return sd


def _validate(sd, name: str) -> None:
    if not isinstance(sd, dict) or not sd:
        raise UnsafeModelFile(f"{name}: expected a non-empty mapping of tensors")
    for k, v in sd.items():
        if not isinstance(k, str) or not isinstance(v, torch.Tensor):
            raise UnsafeModelFile(f"{name}: entry {k!r} is not a named tensor")
        if v.is_floating_point() and not bool(torch.isfinite(v).all()):
            raise UnsafeModelFile(f"{name}: tensor {k!r} contains NaN/inf")
        if not (v.is_floating_point() or v.dtype in (torch.int64, torch.int32, torch.int8, torch.uint8, torch.bool)):
            raise UnsafeModelFile(f"{name}: tensor {k!r} has unsupported dtype {v.dtype}")


def load_candidate(path, build_base_architecture, **kwargs) -> torch.nn.Module:
    """Instantiate the gate's own architecture for the declared base and
    load the candidate's weights into it, strictly."""
    sd = load_state_dict(path, **kwargs)
    model = build_base_architecture()
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        raise UnsafeModelFile(f"weights do not fit the declared base's architecture: {e}") from e
    model.eval()
    return model
