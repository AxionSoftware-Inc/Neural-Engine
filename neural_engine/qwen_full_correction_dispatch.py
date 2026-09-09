"""Optional CUDA kernel for fused selected Qwen output plus correction."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_full_correction_dispatch.cpp"
_CUDA = _ROOT / "qwen_full_correction_dispatch.cu"


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("full correction dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_full_correction_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def fused_correction_dispatch(
    hidden_states: torch.Tensor,
    selected_ids: torch.Tensor,
    route_weights: torch.Tensor,
    group_gate_weight: torch.Tensor,
    group_value_weight: torch.Tensor,
    group_output_weight: torch.Tensor,
    mix_in: torch.Tensor,
    mix_out: torch.Tensor,
    hard_route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected outputs and scaled base-plus-correction output.

    Inputs use the flattened one-token shape ``[tokens, hidden]`` and
    ``[tokens, active]``.  The kernel is intentionally float32/fixed-shape
    only; the normal PyTorch path remains the default and fallback.
    """
    tensors = (
        hidden_states, selected_ids, route_weights, group_gate_weight,
        group_value_weight, group_output_weight, mix_in, mix_out,
    )
    if any(tensor.device.type != "cuda" for tensor in tensors):
        raise ValueError("full correction dispatch requires CUDA tensors")
    if hidden_states.dtype != torch.float32:
        raise ValueError("full correction dispatch currently requires float32")
    if selected_ids.dtype != torch.int64:
        raise ValueError("selected ids must be int64")
    if any(tensor.dtype != torch.float32 for tensor in tensors[2:]):
        raise ValueError("full correction dispatch weights must be float32")
    if hidden_states.dim() != 2 or selected_ids.dim() != 2:
        raise ValueError("hidden and selected ids must be rank-2 tensors")
    if route_weights.shape != selected_ids.shape:
        raise ValueError("route weights shape mismatch")
    if group_gate_weight.dim() != 3 or group_value_weight.shape != group_gate_weight.shape:
        raise ValueError("gate/value weights must be [experts, group, hidden]")
    if group_output_weight.dim() != 3:
        raise ValueError("output weights must be [experts, hidden, group]")
    if mix_in.dim() != 3 or mix_out.dim() != 3:
        raise ValueError("correction weights must be [experts, rank, hidden] and [experts, hidden, rank]")
    tensors = tuple(tensor.contiguous() for tensor in tensors)
    return _extension().forward(*tensors, float(hard_route_scale))
