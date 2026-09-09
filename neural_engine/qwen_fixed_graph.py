"""Small fixed-shape greedy decoder for the custom Qwen KV cache."""

from __future__ import annotations

from collections import OrderedDict

import torch

from benchmark_qwen_custom_kv_graph import (
    FixedDecodeCache,
    forward_logits,
)


@torch.inference_mode()
def greedy_generate_fixed_shape(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    use_cuda_graph: bool = True,
) -> torch.Tensor:
    """Greedily generate with fixed-shape one-token graph replay.

    The graph path is intentionally explicit: callers choose it only for a
    shape that has been validated.  ``use_cuda_graph=False`` uses the same
    custom cache and positions but executes eager forwards, which is the safe
    fallback for uncaptured or unsupported shapes.
    """
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if max_new_tokens == 0:
        return input_ids.clone()
    if input_ids.shape[1] < 1:
        raise ValueError("input_ids must contain at least one prefix token")
    if use_cuda_graph and input_ids.device.type != "cuda":
        raise ValueError("CUDA Graph generation requires CUDA input_ids")

    device = input_ids.device
    prefix_length = int(input_ids.shape[1])
    cache_length = prefix_length + max_new_tokens + 8
    cache = FixedDecodeCache(model.config, cache_length)
    prefix_position = torch.arange(prefix_length, device=device)
    prefix_logits = forward_logits(
        model, input_ids, cache, prefix_position,
    )
    next_token = prefix_logits[:, -1:, :].argmax(dim=-1)
    generated = torch.cat((input_ids, next_token), dim=1)
    if max_new_tokens == 1:
        return generated

    position = torch.tensor([prefix_length], device=device)
    cache.set_decode_position(position)
    static_token = next_token.clone()

    if not use_cuda_graph:
        for step in range(1, max_new_tokens):
            logits = forward_logits(model, static_token, cache, position)
            next_token = logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 < max_new_tokens:
                static_token.copy_(next_token)
                position.copy_(
                    torch.tensor([prefix_length + step], device=device),
                )
                cache.set_decode_position(position)
        return generated

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_logits = forward_logits(model, static_token, cache, position)
    for step in range(1, max_new_tokens):
        graph.replay()
        torch.cuda.synchronize()
        next_token = graph_logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat((generated, next_token), dim=1)
        if step + 1 < max_new_tokens:
            static_token.copy_(next_token)
            position.copy_(
                torch.tensor([prefix_length + step], device=device),
            )
    return generated


class FixedShapeGreedyGraphPool:
    """Reuse captured graphs for validated fixed generation shapes.

    A CUDA Graph owns pointers to the KV tensors used during capture, so a
    cache entry is reused by resetting its storage in place. Entries are
    keyed by batch size, prefix length, and generation budget; callers should
    use the eager fallback when a requested shape is not in the pool or when
    graph capture is not suitable for a model.
    """

    def __init__(self, model: torch.nn.Module, max_entries: int = 4) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.model = model
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, int, int], dict[str, object]] = OrderedDict()
        self.capture_count = 0
        self.hit_count = 0

    @staticmethod
    def _key(input_ids: torch.Tensor, max_new_tokens: int) -> tuple[int, int, int]:
        return (
            int(input_ids.shape[0]),
            int(input_ids.shape[1]),
            int(max_new_tokens),
        )

    def _capture(
        self,
        input_ids: torch.Tensor,
        cache: FixedDecodeCache,
        first_token: torch.Tensor,
        position: torch.Tensor,
    ) -> dict[str, object]:
        static_token = first_token.clone()
        cache.set_decode_position(position)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(
                self.model, static_token, cache, position,
            )
        self.capture_count += 1
        return {
            "cache": cache,
            "graph": graph,
            "static_token": static_token,
            "position": position,
            "graph_logits": graph_logits,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        use_cuda_graph: bool = True,
        capture_on_miss: bool = True,
    ) -> torch.Tensor:
        """Generate and reuse a graph, or eagerly fall back on a cache miss."""
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if max_new_tokens == 0:
            return input_ids.clone()
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one prefix token")
        if not use_cuda_graph:
            return greedy_generate_fixed_shape(
                self.model, input_ids, max_new_tokens,
                use_cuda_graph=False,
            )
        if input_ids.device.type != "cuda":
            raise ValueError("CUDA Graph generation requires CUDA input_ids")

        key = self._key(input_ids, max_new_tokens)
        entry = self._entries.get(key)
        if entry is None and not capture_on_miss:
            return greedy_generate_fixed_shape(
                self.model, input_ids, max_new_tokens,
                use_cuda_graph=False,
            )
        if entry is None:
            cache = FixedDecodeCache(
                self.model.config,
                int(input_ids.shape[1]) + max_new_tokens + 8,
            )
        else:
            self.hit_count += 1
            self._entries.move_to_end(key)
            cache = entry["cache"]
            if not isinstance(cache, FixedDecodeCache):
                raise RuntimeError("invalid fixed graph cache entry")
            cache.reset()

        prefix_length = int(input_ids.shape[1])
        prefix_position = torch.arange(
            prefix_length, device=input_ids.device,
        )
        prefix_logits = forward_logits(
            self.model, input_ids, cache, prefix_position,
        )
        next_token = prefix_logits[:, -1:, :].argmax(dim=-1)
        generated = torch.cat((input_ids, next_token), dim=1)
        if max_new_tokens == 1:
            return generated

        if entry is None:
            position = torch.tensor([prefix_length], device=input_ids.device)
            entry = self._capture(input_ids, cache, next_token, position)
            if len(self._entries) >= self.max_entries:
                self._entries.popitem(last=False)
            self._entries[key] = entry
        else:
            static_token = entry["static_token"]
            position = entry["position"]
            graph = entry["graph"]
            if not isinstance(static_token, torch.Tensor) or not isinstance(position, torch.Tensor):
                raise RuntimeError("invalid fixed graph input buffers")
            if not isinstance(graph, torch.cuda.CUDAGraph):
                raise RuntimeError("invalid fixed graph object")
            static_token.copy_(next_token)
            position.copy_(torch.tensor([prefix_length], device=input_ids.device))
            cache.set_decode_position(position)

        static_token = entry["static_token"]
        position = entry["position"]
        graph_logits = entry["graph_logits"]
        graph = entry["graph"]
        if not isinstance(static_token, torch.Tensor) or not isinstance(position, torch.Tensor):
            raise RuntimeError("invalid fixed graph input buffers")
        if not isinstance(graph_logits, torch.Tensor) or not isinstance(graph, torch.cuda.CUDAGraph):
            raise RuntimeError("invalid fixed graph output")
        for step in range(1, max_new_tokens):
            graph.replay()
            torch.cuda.synchronize()
            next_token = graph_logits[:, -1:, :].argmax(dim=-1)
            generated = torch.cat((generated, next_token), dim=1)
            if step + 1 < max_new_tokens:
                static_token.copy_(next_token)
                position.copy_(
                    torch.tensor([prefix_length + step], device=input_ids.device),
                )
        return generated
