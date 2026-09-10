"""Optional CUDA dispatch for the native factorized additive circuit bank."""

from __future__ import annotations

import os
from functools import lru_cache
import importlib
from pathlib import Path
import sys

import torch

from .qwen_fused_dispatch import _ensure_windows_msvc_environment


_ROOT = Path(__file__).resolve().parent
_CPP = _ROOT / "native_fused_dispatch.cpp"
_CUDA = _ROOT / "native_fused_dispatch.cu"
_EXTENSION_NAME = "neural_engine_native_fused_dispatch_v1"


def _build_directory() -> Path:
    from torch.utils.cpp_extension import _get_build_directory

    return Path(_get_build_directory(_EXTENSION_NAME, verbose=False))


def _import_existing_extension():
    """Import a source-current binary without asking a fresh process to rebuild."""

    build_directory = _build_directory()
    if not build_directory.exists():
        return None
    source_mtime = max(_CPP.stat().st_mtime_ns, _CUDA.stat().st_mtime_ns)
    candidates = sorted(
        build_directory.glob(f"{_EXTENSION_NAME}*.pyd"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for binary in candidates:
        if binary.stat().st_mtime_ns < source_mtime:
            continue
        module_name = binary.stem
        if str(build_directory) not in sys.path:
            sys.path.insert(0, str(build_directory))
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("native fused dispatch requires a CUDA device")
    _ensure_windows_msvc_environment()
    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    existing = _import_existing_extension()
    if existing is not None:
        return existing

    # A fresh Python process has an empty JIT version cache and would otherwise
    # rebuild the same output even when a previous worker already compiled it.
    # Serialize only the first build with a separate process lock; the existing
    # binary is imported after the lock is released, so the builder may keep its
    # DLL loaded while other workers start.
    from torch.utils.cpp_extension import load
    from torch.utils.file_baton import FileBaton

    build_directory = _build_directory()
    build_directory.mkdir(parents=True, exist_ok=True)
    baton = FileBaton(str(build_directory / "process_build.lock"))
    while True:
        if baton.try_acquire():
            try:
                existing = _import_existing_extension()
                if existing is not None:
                    return existing
                return load(
                    name=_EXTENSION_NAME,
                    sources=[str(_CPP), str(_CUDA)],
                    extra_cflags=["/O2"],
                    extra_cuda_cflags=["-Xcompiler", "/Zc:preprocessor"],
                    verbose=False,
                )
            finally:
                baton.release()
        baton.wait()
        existing = _import_existing_extension()
        if existing is not None:
            return existing


def ensure_native_fused_extension():
    """Build/import the extension once before spawning serving workers.

    A parent process can call this during startup so independent workers only
    import an already-built module instead of racing on the same linker output.
    """

    return _extension()


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
