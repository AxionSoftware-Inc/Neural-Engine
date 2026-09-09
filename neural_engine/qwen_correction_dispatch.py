"""Optional CUDA kernel for selected low-rank Qwen correction dispatch."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_correction_dispatch.cpp"
_CUDA = _ROOT / "qwen_correction_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("correction dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_correction_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def correction_dispatch(
    selected_outputs: torch.Tensor,
    selected_ids: torch.Tensor,
    route_weights: torch.Tensor,
    mix_in: torch.Tensor,
    mix_out: torch.Tensor,
    hard_route_scale: float,
) -> torch.Tensor:
    """Return scaled correction for [tokens, active, hidden] selected outputs."""
    tensors = (selected_outputs, selected_ids, route_weights, mix_in, mix_out)
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("correction dispatch requires CUDA tensors")
    if selected_outputs.dtype != torch.float32:
        raise ValueError("selected outputs must be float32")
    if selected_ids.dtype != torch.int64:
        raise ValueError("selected ids must be int64")
    if route_weights.dtype != torch.float32:
        raise ValueError("route weights must be float32")
    if mix_in.dtype != torch.float32 or mix_out.dtype != torch.float32:
        raise ValueError("correction weights must be float32")
    if selected_outputs.dim() != 3:
        raise ValueError("selected outputs must be [tokens, active, hidden]")
    if selected_ids.shape != selected_outputs.shape[:2]:
        raise ValueError("selected ids shape mismatch")
    if route_weights.shape != selected_ids.shape:
        raise ValueError("route weights shape mismatch")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    return _extension().forward(
        *tensors,
        float(hard_route_scale),
    )
