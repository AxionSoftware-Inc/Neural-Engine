"""Measure the one-token grouped fast path inside a full Qwen forward."""

from __future__ import annotations

import argparse
import time

import torch

from benchmark_qwen_multi_layer_transplant import make_transferred_routed_qwen_child
from benchmark_qwen_two_layer_transplant import token_stream


def forward_logits(model, ids: torch.Tensor) -> torch.Tensor:
    return model(input_ids=ids, use_cache=False).logits


def measure(model, ids: torch.Tensor, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            forward_logits(model, ids)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            forward_logits(model, ids)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="26",
                        help="comma-separated Qwen layer indices to replace")
    parser.add_argument("--dispatch-mode", choices=("grouped", "grouped-fused", "packed"),
                        default="grouped")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.float32,
        trust_remote_code=False, local_files_only=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B", local_files_only=True,
    )
    ids = token_stream(
        tokenizer, "Neural Engine", 1, 1, device,
    ).reshape(1, 1)
    layer_indices = [int(value) for value in args.layers.split(",") if value.strip()]
    parents = [model.model.layers[index].mlp for index in layer_indices]
    parent_logits = forward_logits(model, ids)
    parent_ms = measure(model, ids, args.warmup, args.iterations)

    children = []
    for index, parent in zip(layer_indices, parents):
        child = make_transferred_routed_qwen_child(
            parent, 8, 6, 1.0, 0, "base-output", "low-rank",
            args.dispatch_mode, "contiguous", "router", 6.0,
            device, torch.float32,
        ).eval()
        child.single_token_fast_path = False
        model.model.layers[index].mlp = child
        children.append(child)
    grouped_logits = forward_logits(model, ids)
    grouped_ms = measure(model, ids, args.warmup, args.iterations)
    for child in children:
        child.single_token_fast_path = args.dispatch_mode in {"grouped", "grouped-fused"}
    fast_logits = forward_logits(model, ids)
    fast_ms = measure(model, ids, args.warmup, args.iterations)
    grouped_fast_diff = (fast_logits - grouped_logits).abs()
    parent_diff = (grouped_logits - parent_logits).abs()
    print({
        "layers": layer_indices,
        "shape": list(ids.shape),
        "parent_ms": parent_ms,
        "grouped_ms": grouped_ms,
        "grouped_fast_ms": fast_ms,
        "fast_over_grouped": fast_ms / grouped_ms,
        "parent_to_fast": fast_ms / parent_ms,
        "max_fast_vs_grouped_logit_error": grouped_fast_diff.max().item(),
        "mean_fast_vs_grouped_logit_error": grouped_fast_diff.mean().item(),
        "max_sparse_vs_parent_logit_error": parent_diff.max().item(),
    })


if __name__ == "__main__":
    main()
