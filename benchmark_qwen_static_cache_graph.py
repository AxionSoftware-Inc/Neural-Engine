"""Measure fixed-shape CUDA-Graph replay with a Qwen StaticCache decode step."""

from __future__ import annotations

import argparse
import time

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: object,
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
) -> object:
    from transformers import StaticCache

    cache = StaticCache(
        config=model.config,
        max_cache_len=cache_length,
        device=prefix_ids.device,
        dtype=torch.float32,
    )
    prefix_position = torch.arange(
        prefix_ids.shape[1], device=prefix_ids.device,
    )
    with torch.inference_mode():
        forward_logits(model, prefix_ids, cache, prefix_position)
        torch.cuda.synchronize()
    return cache


def measure_eager(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: object,
    cache_position: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup):
            logits = forward_logits(model, input_ids, cache, cache_position)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            logits = forward_logits(model, input_ids, cache, cache_position)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations, logits


def measure_graph(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cache: object,
    cache_position: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor, torch.cuda.CUDAGraph, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, input_ids, cache, cache_position)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(
                model, input_ids, cache, cache_position,
            )
        capture_logits = graph_logits.clone()
        graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
    return (
        (time.perf_counter() - start) * 1000.0 / iterations,
        graph_logits,
        graph,
        capture_logits,
    )


def run_path(
    model: torch.nn.Module,
    prefix_ids: torch.Tensor,
    token_ids: torch.Tensor,
    alternate_ids: torch.Tensor,
    layers: list[int],
    active_experts: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | list[int]]:
    parents = [model.model.layers[index].mlp for index in layers]
    children = []
    for index, parent in zip(layers, parents):
        child = make_transferred_routed_qwen_child(
            parent, 8, active_experts, 1.0, 0,
            "base-output", "low-rank", "grouped", "contiguous", "router",
            float(active_experts), prefix_ids.device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        model.model.layers[index].mlp = child
        children.append(child)

    # Qwen's cache implementation may allocate a shorter sliding-window view
    # when the requested length is exactly prefix+1.  Keep a small fixed
    # decode buffer with headroom while retaining the one-token shape.
    cache_length = max(32, prefix_ids.shape[1] + 8)
    position = torch.tensor([prefix_ids.shape[1]], device=prefix_ids.device)
    eager_cache = make_cache_and_fill_prefix(
        model, prefix_ids, cache_length,
    )
    eager_ms, eager_logits = measure_eager(
        model, token_ids, eager_cache, position, warmup, iterations,
    )
    graph_cache = make_cache_and_fill_prefix(
        model, prefix_ids, cache_length,
    )
    graph_ms, graph_logits, graph, capture_logits = measure_graph(
        model, token_ids, graph_cache, position, warmup, iterations,
    )
    original_graph_logits = graph_logits.clone()
    updated_eager_cache = make_cache_and_fill_prefix(
        model, prefix_ids, cache_length,
    )
    with torch.inference_mode():
        token_ids.copy_(alternate_ids)
        graph.replay()
        torch.cuda.synchronize()
        updated_graph_logits = graph_logits.clone()
        updated_eager_logits = forward_logits(
            model, alternate_ids, updated_eager_cache, position,
        ).clone()
        torch.cuda.synchronize()
    updated_diff = (updated_graph_logits - updated_eager_logits).abs()
    graph_diff = (original_graph_logits - eager_logits).abs()
    capture_diff = (capture_logits - eager_logits).abs()
    unsafe = max(
        capture_diff.max().item(),
        graph_diff.max().item(),
        updated_diff.max().item(),
    ) > 1e-3
    return {
        "status": "UNSAFE_STATIC_CACHE_GRAPH" if unsafe else "PARITY_PASS",
        "layers": layers,
        "active_experts": active_experts,
        "prefix_length": int(prefix_ids.shape[1]),
        "eager_ms": eager_ms,
        "graph_ms": graph_ms,
        "graph_over_eager": graph_ms / eager_ms,
        "max_graph_vs_eager_logit_error": graph_diff.max().item(),
        "mean_graph_vs_eager_logit_error": graph_diff.mean().item(),
        "max_capture_vs_eager_logit_error": capture_diff.max().item(),
        "mean_capture_vs_eager_logit_error": capture_diff.mean().item(),
        "max_graph_vs_updated_input_eager_error": updated_diff.max().item(),
        "mean_graph_vs_updated_input_eager_error": updated_diff.mean().item(),
    }


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
        " attention",
        return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    alternate_ids = tokenizer(
        " routing",
        return_tensors="pt",
    ).input_ids[:, -1:].to(device)
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    result = run_path(
        model, prefix_ids, token_ids, alternate_ids, layers,
        args.active_experts, args.warmup, args.iterations,
    )
    print(result)


if __name__ == "__main__":
    main()
