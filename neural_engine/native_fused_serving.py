"""Safe fixed-shape serving adapter for the native fused Neural Engine path."""

from __future__ import annotations

from collections import OrderedDict
import os
import threading
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


class NativeFusedShapeCache:
    """Serve fixed-width native models through a bounded CUDA-Graph shape cache.

    Dynamic-width and adaptive models deliberately use the eager model. Graph
    entries are keyed by batch, sequence length, and CUDA stream, so concurrent
    streams never share a mutable graph input buffer. A capture failure also
    falls back to eager execution and is recorded for diagnostics.
    """

    def __init__(self, model: nn.Module, max_shapes: int = 8,
                 warmup_iters: int = 5, capture_graphs: bool = True):
        if max_shapes < 1:
            raise ValueError("max_shapes must be positive")
        if warmup_iters < 1:
            raise ValueError("warmup_iters must be positive")
        self.model = model.eval()
        self.owner_pid = os.getpid()
        self.max_shapes = int(max_shapes)
        self.warmup_iters = int(warmup_iters)
        self.capture_graphs = bool(capture_graphs)
        self._entries: OrderedDict[tuple[int, int, int], _ShapeEntry] = OrderedDict()
        self._lock = threading.Lock()
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
        entry = self._get_or_capture(key, inputs)
        if entry is None:
            self.eager_fallback_count += 1
            with torch.inference_mode():
                logits, _ = self.model(inputs, adaptive=False, collect_stats=False)
            return logits
        # A shape/stream entry owns one mutable graph input buffer. Serialize
        # host callers sharing that stream while allowing different streams to
        # use their independent entries concurrently.
        with entry.replay_lock:
            with torch.inference_mode():
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
        }
