"""Optional CUDA finalization for folded grouped Qwen dispatch."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_grouped_finalize.cpp"
_CUDA = _ROOT / "qwen_grouped_finalize.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("grouped finalization requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_grouped_finalize_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def grouped_finalize(
    grouped_output: torch.Tensor,
    packed_positions: torch.Tensor,
    token_ids: torch.Tensor,
    slots: torch.Tensor,
    route_weights: torch.Tensor,
    hard_route_scale: float,
) -> torch.Tensor:
    """Fuse grouped-output gather and token accumulation for float32 CUDA."""
    tensors = (
        grouped_output, packed_positions, token_ids, slots, route_weights,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("grouped finalization requires CUDA tensors")
    if grouped_output.dtype != torch.float32 or route_weights.dtype != torch.float32:
        raise ValueError("grouped finalization currently supports float32")
    if any(tensor.dtype != torch.int64 for tensor in (packed_positions, token_ids, slots)):
        raise ValueError("grouped finalization metadata must be int64")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped finalization inputs must be contiguous")
    return _extension().forward(
        grouped_output, packed_positions, token_ids, slots, route_weights,
        float(hard_route_scale),
    )


def grouped_finalize_token(
    grouped_output: torch.Tensor,
    packed_positions: torch.Tensor,
    route_weights: torch.Tensor,
    hard_route_scale: float,
) -> torch.Tensor:
    """Fuse grouped output reduction with one atomics-free block per token."""
    tensors = grouped_output, packed_positions, route_weights
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("grouped token finalization requires CUDA tensors")
    if grouped_output.dtype != torch.float32 or route_weights.dtype != torch.float32:
        raise ValueError("grouped token finalization currently supports float32")
    if packed_positions.dtype != torch.int64:
        raise ValueError("grouped token finalization positions must be int64")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped token finalization inputs must be contiguous")
    if route_weights.dim() != 2:
        raise ValueError("grouped token finalization weights must be [tokens, active]")
    return _extension().forward_token(
        grouped_output, packed_positions, route_weights,
        float(hard_route_scale),
    )


def grouped_finalize_token_ids(
    grouped_output: torch.Tensor,
    top_ids: torch.Tensor,
    route_weights: torch.Tensor,
    num_experts: int,
    hard_route_scale: float,
) -> torch.Tensor:
    """Finalize fixed grouped output by deriving rows from token expert IDs."""
    tensors = grouped_output, top_ids, route_weights
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("grouped token-id finalization requires CUDA tensors")
    if grouped_output.dtype != torch.float32 or route_weights.dtype != torch.float32:
        raise ValueError("grouped token-id finalization currently supports float32")
    if top_ids.dtype != torch.int64:
        raise ValueError("grouped token-id finalization top_ids must be int64")
    if grouped_output.dim() != 2 or top_ids.dim() != 2 or route_weights.dim() != 2:
        raise ValueError("grouped token-id finalization expects rank-2 tensors")
    if top_ids.shape != route_weights.shape:
        raise ValueError("top_ids/route_weights shape mismatch")
    if grouped_output.shape[0] != int(num_experts) * top_ids.shape[0]:
        raise ValueError("fixed grouped output row count mismatch")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped token-id finalization inputs must be contiguous")
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    return _extension().forward_token_ids(
        grouped_output, top_ids, route_weights, int(num_experts),
        float(hard_route_scale),
    )
