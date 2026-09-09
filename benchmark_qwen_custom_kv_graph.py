"""Benchmark a fixed-position KV cache that is safe to replay in a CUDA Graph."""

from __future__ import annotations

import argparse
import time

import torch
from transformers.cache_utils import Cache, StaticLayer

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child


class FixedDecodeLayer(StaticLayer):
    """Static KV storage with an optional non-advancing decode write position."""

    def __init__(self, max_cache_len: int) -> None:
        super().__init__(max_cache_len=max_cache_len)
        self.max_cache_len = max_cache_len
        self.decode_position: torch.Tensor | None = None

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        if self.decode_position is None:
            kv_length = key_states.shape[-2]
            cache_position = (
                torch.arange(kv_length, device=self.device)
                + self.cumulative_length
            )
            self.cumulative_length.add_(kv_length)
        else:
            cache_position = self.decode_position
        self.keys.index_copy_(2, cache_position, key_states)
        self.values.index_copy_(2, cache_position, value_states)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.max_cache_len, 0

    def get_seq_length(self) -> int:
        return self.cumulative_length if self.is_initialized else 0

    def get_max_length(self) -> int:
        return self.max_cache_len

    def reset(self) -> None:
        super().reset()
        self.decode_position = None


class FixedDecodeCache(Cache):
    """Minimal Cache API used by Qwen attention for fixed-shape graph replay."""

    def __init__(self, config: object, max_cache_len: int) -> None:
        self.layers = [
            FixedDecodeLayer(max_cache_len)
            for _ in range(int(config.num_hidden_layers))
        ]
        super().__init__(layers=self.layers)

    def set_decode_position(self, position: torch.Tensor) -> None:
        for layer in self.layers:
            layer.decode_position = position


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: FixedDecodeCache,
    cache_position: torch.Tensor,
) -> torch.Tensor:
    return model(
        input_ids=input_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    ).logits


def make_cache_and_fill_prefix(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    cache_length: int,
) -> FixedDecodeCache:
    cache = FixedDecodeCache(model.config, cache_length)
    prefix_position = torch.arange(
        prefix_ids.shape[1], device=prefix_ids.device,
    )
    with torch.inference_mode():
        forward_logits(model, prefix_ids, cache, prefix_position)
        torch.cuda.synchronize()
    cache.set_decode_position(
        torch.tensor([prefix_ids.shape[1]], device=prefix_ids.device),
    )
    return cache


def measure_eager(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: FixedDecodeCache,
    position: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup):
            logits = forward_logits(model, input_ids, cache, position)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            logits = forward_logits(model, input_ids, cache, position)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations, logits.clone()


def measure_graph(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: FixedDecodeCache,
    position: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, input_ids, cache, position)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(model, input_ids, cache, position)
        capture_logits = graph_logits.clone()
        graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
    return (
        (time.perf_counter() - start) * 1000.0 / iterations,
        capture_logits,
        graph_logits,
        graph,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,26")
    parser.add_argument("--active-experts", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        dtype=torch.float32,
        trust_remote_code=False,
        local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True,
    )
    prefix_ids = tokenizer(
        "Neural Engine sparse circuits",
        return_tensors="pt",
    ).input_ids[:, :4].to(device)
    token_ids = tokenizer(
        " attention", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    alternate_ids = tokenizer(
        " routing", return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    cache_length = max(32, prefix_ids.shape[1] + 8)
    position = torch.tensor([prefix_ids.shape[1]], device=device)

    parent_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    parent_ms, parent_logits = measure_eager(
        model, token_ids, parent_cache, position, args.warmup, args.iterations,
    )
    parents = [model.model.layers[index].mlp for index in layers]
    for index, parent in zip(layers, parents):
        child = make_transferred_routed_qwen_child(
            parent, 8, args.active_experts, 1.0, 0,
            "base-output", "low-rank", "grouped", "contiguous", "router",
            float(args.active_experts), device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        model.model.layers[index].mlp = child

    eager_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    eager_ms, eager_logits = measure_eager(
        model, token_ids, eager_cache, position, args.warmup, args.iterations,
    )
    graph_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    graph_ms, capture_logits, graph_logits, graph = measure_graph(
        model, token_ids, graph_cache, position, args.warmup, args.iterations,
    )
    capture_diff = (capture_logits - eager_logits).abs()
    replay_diff = (graph_logits - eager_logits).abs()

    updated_graph_cache = make_cache_and_fill_prefix(
        model, prefix_ids, cache_length,
    )
    with torch.inference_mode():
        static_token_ids = token_ids.clone()
        updated_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(updated_graph):
            updated_graph_logits = forward_logits(
                model, static_token_ids, updated_graph_cache, position,
            )
        static_token_ids.copy_(alternate_ids)
        updated_graph.replay()
        torch.cuda.synchronize()
        alternate_graph_logits = updated_graph_logits.clone()
        alternate_eager_cache = make_cache_and_fill_prefix(
            model, prefix_ids, cache_length,
        )
        alternate_eager_logits = forward_logits(
            model, alternate_ids, alternate_eager_cache, position,
        ).clone()
        torch.cuda.synchronize()
    alternate_diff = (alternate_graph_logits - alternate_eager_logits).abs()
    # The output pointer observed immediately when capture exits is not the
    # serving result on this PyTorch build.  The first explicit replay is the
    # contract we benchmark and compare; it is numerically exact here.
    max_error = max(replay_diff.max().item(), alternate_diff.max().item())
    print({
        "status": "PARITY_PASS" if max_error <= 1e-3 else "PARITY_FAIL",
        "layers": layers,
        "shape": list(token_ids.shape),
        "prefix_length": int(prefix_ids.shape[1]),
        "active_experts": args.active_experts,
        "parent_ms": parent_ms,
        "sparse_eager_ms": eager_ms,
        "sparse_graph_ms": graph_ms,
        "graph_over_eager": graph_ms / eager_ms,
        "graph_over_parent": graph_ms / parent_ms,
        "max_capture_vs_eager_logit_error": capture_diff.max().item(),
        "max_replay_vs_eager_logit_error": replay_diff.max().item(),
        "max_updated_input_graph_vs_eager_logit_error": alternate_diff.max().item(),
    })


if __name__ == "__main__":
    main()
