"""Profile one-token sparse Qwen routing stages independently.

This is a diagnostic benchmark, not a quality experiment.  It keeps the
existing copied Qwen child and measures the same eight-layer shape used by
the runtime track while separating router MLP, top-k/softmax, and selected
FFN dispatch.  The measurements are intentionally done on fixed hidden
states so stage timings do not include Transformer attention or KV-cache
work.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from benchmark_qwen_multi_layer_transplant import (
    make_transferred_routed_qwen_child,
)


def measure_cuda(operation, warmup: int, iterations: int) -> float:
    """Return mean milliseconds for a fixed-shape CUDA operation."""
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def build_children(model, layer_indices, device):
    children = []
    for index in layer_indices:
        parent = model.model.layers[index].mlp
        child = make_transferred_routed_qwen_child(
            parent, 8, 6, 1.0, 0, "base-output", "low-rank",
            "grouped", "contiguous", "router", 6.0,
            device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        children.append(child)
    return children


def profile_batch(
    children,
    batch_size: int,
    hidden_size: int,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(1701 + batch_size)
    hidden = [
        torch.randn(
            batch_size, 1, hidden_size, device=device,
            dtype=torch.float32, generator=generator,
        )
        for _ in children
    ]

    def router_only() -> None:
        for child, state in zip(children, hidden):
            child.router(state)

    scores = [child.router(state) for child, state in zip(children, hidden)]
    routes = []
    for child, score in zip(children, scores):
        top_values, top_ids = score.topk(child.active_experts, dim=-1)
        weights = F.softmax(top_values / child.temperature, dim=-1)
        routes.append((top_ids, weights))

    def topk_softmax_only() -> None:
        for child, score in zip(children, scores):
            top_values, _ = score.topk(child.active_experts, dim=-1)
            F.softmax(top_values / child.temperature, dim=-1)

    def route_and_topk() -> None:
        for child, state in zip(children, hidden):
            score = child.router(state)
            top_values, _ = score.topk(child.active_experts, dim=-1)
            F.softmax(top_values / child.temperature, dim=-1)

    def dispatch_only() -> None:
        for child, state, (top_ids, weights) in zip(children, hidden, routes):
            child._forward_grouped(state, top_ids, weights)

    def full_child() -> None:
        for child, state in zip(children, hidden):
            child(state)

    router_ms = measure_cuda_op(router_only, warmup, iterations)
    topk_ms = measure_cuda_op(topk_softmax_only, warmup, iterations)
    route_topk_ms = measure_cuda_op(route_and_topk, warmup, iterations)
    dispatch_ms = measure_cuda_op(dispatch_only, warmup, iterations)
    full_ms = measure_cuda_op(full_child, warmup, iterations)
    return {
        "batch_size": batch_size,
        "layers": len(children),
        "router_mlp_ms": router_ms,
        "topk_softmax_ms": topk_ms,
        "router_plus_topk_ms": route_topk_ms,
        "selected_ffn_dispatch_ms": dispatch_ms,
        "decomposed_sum_ms": route_topk_ms + dispatch_ms,
        "full_child_forward_ms": full_ms,
        "router_fraction_of_full": router_ms / max(full_ms, 1e-9),
        "topk_fraction_of_full": topk_ms / max(full_ms, 1e-9),
        "dispatch_fraction_of_full": dispatch_ms / max(full_ms, 1e-9),
        "decomposed_over_full": (
            route_topk_ms + dispatch_ms
        ) / max(full_ms, 1e-9),
    }


def measure_cuda_op(operation, warmup: int, iterations: int) -> float:
    """Alias kept separate so call sites read as stage measurements."""
    return measure_cuda(operation, warmup, iterations)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", default="0,4,8,12,16,20,24,26")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    from transformers import AutoModelForCausalLM

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=False,
        local_files_only=True,
    ).to(device).eval()
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    children = build_children(model, layer_indices, device)
    hidden_size = int(model.config.hidden_size)
    records = [
        profile_batch(
            children, batch_size, hidden_size, device,
            args.warmup, args.iterations,
        )
        for batch_size in args.batch_sizes
    ]
    result = {
        "experiment": "V0.219_router_topk_dispatch_stage_profile",
        "model": args.model,
        "dtype": "float32",
        "layers": layer_indices,
        "active_experts": 6,
        "num_experts": 8,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "records": records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
