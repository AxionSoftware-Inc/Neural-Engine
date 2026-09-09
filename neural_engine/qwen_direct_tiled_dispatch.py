"""Optional route-aware tiled CUDA grouped projection for Qwen."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_direct_tiled_dispatch.cpp"
_CUDA = _ROOT / "qwen_direct_tiled_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("direct tiled dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_direct_tiled_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def direct_tiled_dispatch(
    flat_hidden: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_slots: torch.Tensor,
    starts: torch.Tensor,
    counts: torch.Tensor,
    route_weights: torch.Tensor,
    group_gate_weight: torch.Tensor,
    group_value_weight: torch.Tensor,
    group_output_weight: torch.Tensor,
    active_experts: int,
    hard_route_scale: float,
    uniform_accum: bool,
) -> torch.Tensor:
    """Fuse sorted-route packing, grouped SwiGLU, and token accumulation."""
    tensors = (
        flat_hidden, sorted_token_ids, sorted_slots, starts, counts,
        route_weights, group_gate_weight, group_value_weight,
        group_output_weight,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("direct tiled dispatch requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in (
        flat_hidden, route_weights, group_gate_weight, group_value_weight,
        group_output_weight,
    )):
        raise ValueError("direct tiled dispatch currently supports float32")
    if any(tensor.dtype != torch.int64 for tensor in (
        sorted_token_ids, sorted_slots, starts, counts,
    )):
        raise ValueError("direct tiled route metadata must be int64")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("direct tiled dispatch inputs must be contiguous")
    return _extension().forward(
        flat_hidden, sorted_token_ids, sorted_slots, starts, counts,
        route_weights, group_gate_weight, group_value_weight,
        group_output_weight, int(active_experts), float(hard_route_scale),
        bool(uniform_accum),
    )
