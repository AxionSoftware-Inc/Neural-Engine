"""Optional tiled CUDA grouped selected-FFN projection for Qwen."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_tiled_dispatch.cpp"
_CUDA = _ROOT / "qwen_tiled_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("tiled CUDA dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_tiled_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def tiled_grouped_dispatch(
    grouped_hidden: torch.Tensor,
    group_gate_weight: torch.Tensor,
    group_value_weight: torch.Tensor,
    group_output_weight: torch.Tensor,
) -> torch.Tensor:
    """Compute expert-major grouped SwiGLU outputs with tiled CUDA GEMMs."""
    tensors = (
        grouped_hidden, group_gate_weight, group_value_weight,
        group_output_weight,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("tiled grouped dispatch requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("tiled grouped dispatch currently supports float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("tiled grouped dispatch inputs must be contiguous")
    if grouped_hidden.dim() != 3:
        raise ValueError("grouped hidden must be [experts, rows, hidden]")
    return _extension().forward(
        grouped_hidden,
        group_gate_weight,
        group_value_weight,
        group_output_weight,
    )
