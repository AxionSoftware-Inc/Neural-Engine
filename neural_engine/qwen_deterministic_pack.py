"""Optional deterministic fixed-layout CUDA route packing for Qwen."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_deterministic_pack.cpp"
_CUDA = _ROOT / "qwen_deterministic_pack.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("deterministic pack requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_deterministic_pack_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def deterministic_pack(
    hidden_states: torch.Tensor,
    top_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Pack to [expert, token, hidden] without atomics or route sorting."""
    if hidden_states.device.type != "cuda":
        raise ValueError("deterministic pack requires CUDA hidden states")
    if hidden_states.dtype != torch.float32:
        raise ValueError("deterministic pack currently supports float32 only")
    if top_ids.dtype != torch.int64:
        raise ValueError("top_ids must be int64")
    if hidden_states.dim() != 2 or top_ids.dim() != 2:
        raise ValueError("hidden states and top_ids must be rank-2")
    if hidden_states.shape[0] != top_ids.shape[0]:
        raise ValueError("hidden/top_ids token dimension mismatch")
    if not hidden_states.is_contiguous() or not top_ids.is_contiguous():
        raise ValueError("deterministic pack inputs must be contiguous")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    return _extension().forward(hidden_states, top_ids, int(num_experts))
