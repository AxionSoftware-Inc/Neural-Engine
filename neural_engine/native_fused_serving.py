"""Safe fixed-shape serving adapter for the native fused Neural Engine path."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import os
from pathlib import Path
import tempfile
import threading
import time
from dataclasses import dataclass, field

import torch
from torch import nn


class _LogitsCall(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(inputs, adaptive=False, collect_stats=False)
        return logits


@dataclass
class _ShapeEntry:
    inputs: torch.Tensor
    graph_call: nn.Module
    replay_lock: threading.Lock = field(default_factory=threading.Lock)


class _ProcessFileLock:
    """Crash-releasing advisory lock shared by CUDA worker processes."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+b")
        self._handle.seek(0, 2)
        if self._handle.tell() == 0:
            self._handle.write(b"0")
            self._handle.flush()
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(self._handle.fileno(), msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if self._handle is None:
            return
        self._handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


_GRAPH_PROCESS_LOCK_ROOT = Path(tempfile.gettempdir())


def _graph_process_lock_path(device_index: int) -> Path:
    return _GRAPH_PROCESS_LOCK_ROOT / (
        f"neural_engine_native_fused_graph_device_{device_index}.lock"
    )


class NativeFusedShapeCache:
    """Serve fixed-width native models through a bounded CUDA-Graph shape cache.

    Dynamic-width and adaptive models deliberately use the eager model. Graph
    entries are keyed by batch, sequence length, and CUDA stream, so concurrent
    streams never share a mutable graph input buffer. A capture failure also
    falls back to eager execution and is recorded for diagnostics.
    """

    def __init__(self, model: nn.Module, max_shapes: int = 8,
                 warmup_iters: int = 5, capture_graphs: bool = True,
                 process_graph_lock: bool = False):
        if max_shapes < 1:
            raise ValueError("max_shapes must be positive")
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be positive")
        self.model = model.eval()
        self.owner_pid = os.getpid()
        self.max_shapes = int(max_shapes)
        self.warmup_iters = int(warmup_iters)
        self.capture_graphs = bool(capture_graphs)
        self.process_graph_lock = bool(process_graph_lock)
        self._entries: OrderedDict[tuple[int, int, int], _ShapeEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stream_locks: dict[int, threading.Lock] = {}
        self.capture_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.eviction_count = 0
        self.eager_fallback_count = 0
        self.capture_failures: list[dict[str, str]] = []

    def _graph_eligible(self, inputs: torch.Tensor) -> bool:
        return bool(
            self.capture_graphs
            and torch.cuda.is_available()
            and inputs.device.type == "cuda"
            and inputs.dtype == torch.long
            and not self.model.training
            and getattr(self.model, "dynamic_width_mode", "none") == "none"
            and inputs.ndim == 2
        )

    @staticmethod
    def _stream_key() -> int:
        return int(torch.cuda.current_stream().cuda_stream)

    def _key(self, inputs: torch.Tensor) -> tuple[int, int, int]:
        return (int(inputs.shape[0]), int(inputs.shape[1]), self._stream_key())

    def _capture(self, inputs: torch.Tensor) -> _ShapeEntry:
        static_inputs = inputs.detach().clone()
        wrapper = _LogitsCall(self.model).eval()
        with torch.inference_mode():
            graph_call = torch.cuda.make_graphed_callables(
                wrapper, (static_inputs,),
                num_warmup_iters=self.warmup_iters,
                allow_unused_input=True,
            )
        self.capture_count += 1
        return _ShapeEntry(static_inputs, graph_call)

    def _get_or_capture(self, key: tuple[int, int, int],
                        inputs: torch.Tensor) -> _ShapeEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self.cache_hit_count += 1
                self._entries.move_to_end(key)
                return entry
            self.cache_miss_count += 1
            try:
                # CUDA Graph capture is process-local but the CUDA driver can
                # still reject overlapping captures on one physical device.
                # The optional advisory lock is released automatically if a
                # worker exits, unlike a sentinel-directory lock.
                device_index = inputs.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                capture_lock = (
                    _ProcessFileLock(_graph_process_lock_path(device_index))
                    if self.process_graph_lock else nullcontext()
                )
                with capture_lock:
                    entry = self._capture(inputs)
            except Exception as exc:
                self.capture_failures.append({
                    "type": type(exc).__name__,
                    "message": str(exc)[-1000:],
                })
                return None
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_shapes:
                self._entries.popitem(last=False)
                self.eviction_count += 1
            return entry

    def _stream_lock(self, stream_key: int) -> threading.Lock:
        with self._lock:
            return self._stream_locks.setdefault(stream_key, threading.Lock())

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        if os.getpid() != self.owner_pid:
            raise RuntimeError(
                "NativeFusedShapeCache is process-local; construct a model and cache per worker"
            )
        if not self._graph_eligible(inputs):
            self.eager_fallback_count += 1
            with torch.inference_mode():
                logits, _ = self.model(inputs, adaptive=False, collect_stats=False)
            return logits
        key = self._key(inputs)
        # CUDA graph capture and replay on one stream must not overlap, even
        # when the requests use different cached shapes. Keep different CUDA
        # streams independent so multi-stream callers can still overlap.
        with self._stream_lock(key[2]):
            entry = self._get_or_capture(key, inputs)
            if entry is None:
                self.eager_fallback_count += 1
                with torch.inference_mode():
                    logits, _ = self.model(inputs, adaptive=False, collect_stats=False)
                return logits
            # A shape/stream entry owns one mutable graph input buffer. The
            # entry lock remains as a local defense if replay is refactored.
            with entry.replay_lock:
                device_index = inputs.device.index
                if device_index is None:
                    device_index = torch.cuda.current_device()
                replay_lock = (
                    _ProcessFileLock(_graph_process_lock_path(device_index))
                    if self.process_graph_lock else nullcontext()
                )
                with replay_lock, torch.inference_mode():
                    entry.inputs.copy_(inputs)
                    return entry.graph_call(entry.inputs).clone()

    def stats(self) -> dict[str, object]:
        return {
            "owner_pid": self.owner_pid,
            "cached_shape_count": len(self._entries),
            "cache_keys": [
                {"batch_size": key[0], "sequence_length": key[1], "stream": key[2]}
                for key in self._entries
            ],
            "capture_count": self.capture_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_miss_count": self.cache_miss_count,
            "eviction_count": self.eviction_count,
            "eager_fallback_count": self.eager_fallback_count,
            "capture_failures": list(self.capture_failures),
            "process_graph_lock": self.process_graph_lock,
        }
