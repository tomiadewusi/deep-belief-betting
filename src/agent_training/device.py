from __future__ import annotations

from typing import Literal

import torch

DEVICE = Literal["auto", "cpu", "cuda", "mps"]

def resolve_device(name: DEVICE) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and bool(torch.backends.mps.is_available()):
            return torch.device("mps")
        return torch.device("cpu")
    elif name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA not available")
    elif name == "mps" and not hasattr(torch.backends, "mps") and not bool(torch.backends.mps.is_available()):
        raise RuntimeError("device=mps but MPS not available")
    return torch.device(name)