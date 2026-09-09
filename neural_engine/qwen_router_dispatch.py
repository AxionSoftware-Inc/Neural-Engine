"""Optional fixed-shape CUDA router for one-token Qwen decode."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_router_dispatch.cpp"
_CUDA = _ROOT / "qwen_router_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("fused Qwen router requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_router_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def fused_router(
    hidden_states: torch.Tensor,
    first_weight: torch.Tensor,
    first_bias: torch.Tensor,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor,
    active_experts: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return top-k ids and softmax weights for a float32 router bank.

    ``hidden_states`` is flattened to ``[tokens, hidden]``.  The operation is
    deliberately opt-in and fixed to the two-linear SiLU router used by the
    Qwen sparse child; the regular PyTorch router remains the fallback.
    """
    tensors = (
        hidden_states, first_weight, first_bias, second_weight, second_bias,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("fused router requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("fused router currently supports float32 tensors")
    if hidden_states.dim() != 2:
        raise ValueError("hidden states must be [tokens, hidden]")
    if first_weight.dim() != 2 or first_bias.dim() != 1:
        raise ValueError("first router projection must be [router, hidden]")
    if second_weight.dim() != 2 or second_bias.dim() != 1:
        raise ValueError("second router projection must be [experts, router]")
    if first_weight.shape[1] != hidden_states.shape[1]:
        raise ValueError("router hidden dimension mismatch")
    if first_bias.shape[0] != first_weight.shape[0]:
        raise ValueError("first router bias dimension mismatch")
    if second_weight.shape[1] != first_weight.shape[0]:
        raise ValueError("router intermediate dimension mismatch")
    if second_bias.shape[0] != second_weight.shape[0]:
        raise ValueError("second router bias dimension mismatch")
    if not 1 <= active_experts <= second_weight.shape[0]:
        raise ValueError("active_experts must be within the router output")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    return _extension().forward(
        *tensors, int(active_experts), float(temperature),
    )


def fused_subset_router(
    hidden_states: torch.Tensor,
    first_weight: torch.Tensor,
    first_bias: torch.Tensor,
    second_weight: torch.Tensor,
    second_bias: torch.Tensor,
    subset_membership: torch.Tensor,
    active_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the hard best-subset route for the trained Qwen K-subset path."""
    tensors = (
        hidden_states, first_weight, first_bias, second_weight, second_bias,
        subset_membership,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("fused subset router requires CUDA tensors")
    if any(tensor.dtype != torch.float32 for tensor in tensors):
        raise ValueError("fused subset router currently supports float32 tensors")
    if hidden_states.dim() != 2:
        raise ValueError("hidden states must be [tokens, hidden]")
    if first_weight.dim() != 2 or first_bias.dim() != 1:
        raise ValueError("first router projection must be [router, hidden]")
    if second_weight.dim() != 2 or second_bias.dim() != 1:
        raise ValueError("second router projection must be [subsets, router]")
    if subset_membership.dim() != 2:
        raise ValueError("subset membership must be [subsets, experts]")
    if first_weight.shape[1] != hidden_states.shape[1]:
        raise ValueError("router hidden dimension mismatch")
    if first_bias.shape[0] != first_weight.shape[0]:
        raise ValueError("first router bias dimension mismatch")
    if second_weight.shape[1] != first_weight.shape[0]:
        raise ValueError("second router intermediate dimension mismatch")
    if second_bias.shape[0] != second_weight.shape[0]:
        raise ValueError("second router bias dimension mismatch")
    if subset_membership.shape[0] != second_weight.shape[0]:
        raise ValueError("subset count mismatch")
    if not 1 <= active_experts <= subset_membership.shape[1]:
        raise ValueError("active_experts must be within the expert count")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    return _extension().forward_subset(
        *tensors, int(active_experts),
    )
