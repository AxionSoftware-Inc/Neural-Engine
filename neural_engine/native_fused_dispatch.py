"""Optional CUDA dispatch for the native factorized additive circuit bank."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "native_fused_dispatch.cpp"
_CUDA = _ROOT / "native_fused_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("native fused dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_native_fused_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def fused_factorized_dispatch(
    state: torch.Tensor,
    circuit_ids: torch.Tensor,
    weights: torch.Tensor,
    down_factors: torch.Tensor,
    up_factors: torch.Tensor,
    bias_factors: torch.Tensor,
    factor_mix: torch.Tensor,
    address_factor_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute an additive ordered factorized bank without Python/einsum loops."""
    tensors = (
        state, circuit_ids, weights, down_factors, up_factors, bias_factors,
        factor_mix, address_factor_ids,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("native fused dispatch requires CUDA tensors")
    if state.dtype != torch.float32 or weights.dtype != torch.float32:
        raise ValueError("native fused dispatch currently supports float32 state/weights")
    if any(tensor.dtype != torch.float32 for tensor in tensors[3:7]):
        raise ValueError("native fused dispatch factor weights must be float32")
    if circuit_ids.dtype != torch.int64 or address_factor_ids.dtype != torch.int64:
        raise ValueError("native fused dispatch IDs must be int64")
    if state.dim() != 2 or circuit_ids.dim() != 2 or weights.shape != circuit_ids.shape:
        raise ValueError("state, circuit IDs and route weights have incompatible shapes")
    if down_factors.dim() != 4 or up_factors.dim() != 4 or bias_factors.dim() != 3:
        raise ValueError("factor tensors have incompatible ranks")
    if down_factors.shape[0] != 2 or up_factors.shape[0] != 2 or bias_factors.shape[0] != 2:
        raise ValueError("native fused dispatch requires ordered two-slot factors")
    if down_factors.shape[1] != up_factors.shape[1] or down_factors.shape[1] != bias_factors.shape[1]:
        raise ValueError("factor count mismatch")
    if down_factors.shape[2] != state.shape[1] or up_factors.shape[3] != state.shape[1]:
        raise ValueError("state dimension mismatch")
    if down_factors.shape[3] != up_factors.shape[2]:
        raise ValueError("circuit rank mismatch")
    if factor_mix.dim() != 2 or factor_mix.shape[1] != 2:
        raise ValueError("factor_mix must be [addresses, 2]")
    if address_factor_ids.shape != factor_mix.shape:
        raise ValueError("address factor map must match factor_mix")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    return _extension().forward(*tensors)
