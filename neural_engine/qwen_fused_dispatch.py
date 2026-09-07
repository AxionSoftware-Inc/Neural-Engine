"""Optional CUDA dispatch for transferred Qwen SwiGLU groups.

The normal PyTorch dispatch paths remain the default.  This module is lazy:
the extension is compiled only when ``dispatch_mode=fused`` is requested.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "qwen_fused_dispatch.cpp"
_CUDA = _ROOT / "qwen_fused_dispatch.cu"


def _ensure_windows_msvc_environment() -> None:
    """Import the VS x64 tool environment when Python was not started from it."""
    if os.name != "nt" or shutil.which("cl"):
        return
    candidates = sorted(
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022").glob(
            "*/Common7/Tools/VsDevCmd.bat"
        )
    )
    if not candidates:
        raise RuntimeError(
            "MSVC cl.exe was not found; install Visual Studio C++ Build Tools "
            "before building the fused CUDA dispatch"
        )
    command = (
        f'call "{candidates[-1]}" -arch=x64 >nul && set'
    )
    output = subprocess.check_output(
        command,
        shell=True,
        text=True,
        stderr=subprocess.STDOUT,
    )
    for line in output.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key:
                os.environ[key] = value
    os.environ.setdefault("DISTUTILS_USE_SDK", "1")


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("fused CUDA dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    from torch.utils.cpp_extension import load

    return load(
        name="neural_engine_qwen_fused_dispatch_v1",
        sources=[str(_CPP), str(_CUDA)],
        extra_cflags=["/O2"],
        extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
        verbose=False,
    )


def fused_dispatch(
    hidden_states: torch.Tensor,
    top_ids: torch.Tensor,
    route_weights: torch.Tensor,
    group_gate_weight: torch.Tensor,
    group_value_weight: torch.Tensor,
    group_output_weight: torch.Tensor,
    hard_route_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return unweighted selected outputs and scaled sparse output.

    The first result has shape ``[..., K, H]`` and the second ``[..., H]``.
    The initial kernel intentionally supports float32 CUDA tensors only; this
    is the benchmark's current Qwen protocol and keeps the parity contract
    explicit instead of silently changing numerical behavior for other dtypes.
    """
    if hidden_states.device.type != "cuda":
        raise ValueError("fused dispatch requires CUDA hidden states")
    if hidden_states.dtype != torch.float32:
        raise ValueError("fused dispatch currently supports float32 only")
    if top_ids.dtype != torch.int64:
        raise ValueError("top_ids must be int64")
    tensors = (
        hidden_states, top_ids, route_weights,
        group_gate_weight, group_value_weight, group_output_weight,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused dispatch inputs must be contiguous")
    if route_weights.dtype != torch.float32:
        raise ValueError("route_weights must be float32")
    if any(tensor.dtype != torch.float32 for tensor in tensors[3:]):
        raise ValueError("fused dispatch weights must be float32")
    shape = hidden_states.shape
    flat_hidden = hidden_states.reshape(-1, shape[-1])
    flat_ids = top_ids.reshape(-1, top_ids.shape[-1])
    flat_weights = route_weights.reshape(-1, route_weights.shape[-1])
    selected, output = _extension().forward(
        flat_hidden,
        flat_ids,
        flat_weights,
        group_gate_weight,
        group_value_weight,
        group_output_weight,
        float(hard_route_scale),
    )
    selected = selected.reshape(*shape[:-1], top_ids.shape[-1], shape[-1])
    return selected, output.reshape(*shape)
