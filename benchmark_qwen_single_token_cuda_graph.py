"""Measure CUDA-Graph replay for the fixed-shape Qwen sparse decode smoke."""

from __future__ import annotations

import argparse
import time

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_two_layer_transplant import token_stream


def forward_logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, use_cache=False).logits


def measure_eager(
    model: torch.nn.Module,
    ids: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup):
            logits = forward_logits(model, ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            logits = forward_logits(model, ids)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations, logits


def measure_graph(
    model: torch.nn.Module,
    ids: torch.Tensor,
    warmup: int,
    iterations: int,
) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, ids)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(model, ids)
        graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations, graph_logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,26")
    parser.add_argument("--active-experts", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1)
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
    ids = token_stream(
        tokenizer, "Neural Engine", args.batch_size,
        args.sequence_length, device,
    ).reshape(args.batch_size, args.sequence_length)
    layer_indices = [
        int(value) for value in args.layers.split(",") if value.strip()
    ]

    parent_ms, parent_logits = measure_eager(
        model, ids, args.warmup, args.iterations,
    )
    parents = [model.model.layers[index].mlp for index in layer_indices]
    children = []
    for index, parent in zip(layer_indices, parents):
        child = make_transferred_routed_qwen_child(
            parent, 8, args.active_experts, 1.0, 0,
            "base-output", "low-rank", "grouped", "contiguous", "router",
            float(args.active_experts), device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        model.model.layers[index].mlp = child
        children.append(child)

    eager_ms, eager_logits = measure_eager(
        model, ids, args.warmup, args.iterations,
    )
    graph_ms, graph_logits = measure_graph(
        model, ids, args.warmup, args.iterations,
    )
    # A serving caller changes token ids between replays.  Keep the captured
    # shape and storage fixed, but update the graph input buffer in place.
    alternate_ids = token_stream(
        tokenizer, "attention free circuits", args.batch_size,
        args.sequence_length, device,
    ).reshape(args.batch_size, args.sequence_length)
    # ``measure_graph`` owns its graph, so make a small second capture here
    # with an explicit handle for the input-update parity check.  Capture the
    # original token and then replace it before replaying.
    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            updated_graph_logits = forward_logits(model, ids)
        ids.copy_(alternate_ids)
        graph.replay()
        torch.cuda.synchronize()
        updated_eager_logits = forward_logits(model, alternate_ids)
    updated_diff = (updated_graph_logits - updated_eager_logits).abs()
    graph_diff = (graph_logits - eager_logits).abs()
    sparse_diff = (eager_logits - parent_logits).abs()
    print({
        "layers": layer_indices,
        "shape": list(ids.shape),
        "active_experts": args.active_experts,
        "parent_ms": parent_ms,
        "sparse_eager_fast_ms": eager_ms,
        "sparse_cuda_graph_ms": graph_ms,
        "graph_over_eager": graph_ms / eager_ms,
        "eager_over_parent": eager_ms / parent_ms,
        "graph_over_parent": graph_ms / parent_ms,
        "max_graph_vs_eager_logit_error": graph_diff.max().item(),
        "mean_graph_vs_eager_logit_error": graph_diff.mean().item(),
        "max_graph_vs_updated_input_eager_error": updated_diff.max().item(),
        "mean_graph_vs_updated_input_eager_error": updated_diff.mean().item(),
        "max_sparse_vs_parent_logit_error": sparse_diff.max().item(),
    })


if __name__ == "__main__":
    main()
