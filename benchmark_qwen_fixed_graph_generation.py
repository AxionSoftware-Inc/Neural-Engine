"""Parity smoke for the fixed-shape greedy generation adapter."""

from __future__ import annotations

import argparse
import json

import torch

from neural_engine.qwen_fixed_graph import (
    FixedShapeGreedyGraphPool,
    greedy_generate_fixed_shape,
)
from benchmark_qwen_multi_layer_transplant import (
    TransferredRoutedQwenChild,
    make_transferred_routed_qwen_child,
)


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least one distinct index")
    return layers


def run(args: argparse.Namespace) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32, trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model, local_files_only=args.local_files_only,
    )
    layers = parse_layers(args.layers)
    for index in layers:
        parent = model.model.layers[index].mlp
        child = make_transferred_routed_qwen_child(
            parent, 8, 5, 1.0, 0,
            "base-output", "low-rank", "grouped", "contiguous", "router",
            5.0, device, torch.float32,
        ).eval()
        route_base = next(
            nested for nested in child.modules()
            if isinstance(nested, TransferredRoutedQwenChild)
        )
        route_base.single_token_fast_path = True
        model.model.layers[index].mlp = child

    prompt = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :args.prefix_tokens].to(device)
    pool = FixedShapeGreedyGraphPool(model, max_entries=2)
    graph_tokens = pool.generate(prompt, args.new_tokens, use_cuda_graph=True)
    graph_tokens_reused = pool.generate(
        prompt, args.new_tokens, use_cuda_graph=True,
    )
    eager_tokens = greedy_generate_fixed_shape(
        model, prompt, args.new_tokens, use_cuda_graph=False,
    )
    fallback_budget = max(2, args.new_tokens - 1)
    fallback_tokens = pool.generate(
        prompt, fallback_budget, use_cuda_graph=True, capture_on_miss=False,
    )
    fallback_eager_tokens = greedy_generate_fixed_shape(
        model, prompt, fallback_budget, use_cuda_graph=False,
    )
    # Fill the bounded pool with two new shapes; the original shape is then
    # evicted and must use the explicit eager fallback when capture is off.
    pool.generate(prompt, max(2, args.new_tokens - 2), use_cuda_graph=True)
    pool.generate(prompt, max(2, args.new_tokens - 3), use_cuda_graph=True)
    evicted_tokens = pool.generate(
        prompt, args.new_tokens, use_cuda_graph=True, capture_on_miss=False,
    )
    evicted_eager_tokens = greedy_generate_fixed_shape(
        model, prompt, args.new_tokens, use_cuda_graph=False,
    )
    equal = bool(torch.equal(graph_tokens, eager_tokens))
    reused_equal = bool(torch.equal(graph_tokens, graph_tokens_reused))
    fallback_equal = bool(torch.equal(fallback_tokens, fallback_eager_tokens))
    eviction_fallback_equal = bool(
        torch.equal(evicted_tokens, evicted_eager_tokens)
    )
    result = {
        "experiment": "V0.199_qwen_fixed_graph_greedy_generation",
        "status": "PARITY_PASS" if equal else "PARITY_FAIL",
        "layers": layers,
        "prefix_tokens": int(prompt.shape[1]),
        "new_tokens": args.new_tokens,
        "exact_token_match": equal,
        "reused_shape_exact_token_match": reused_equal,
        "uncaptured_shape_eager_fallback_exact_token_match": fallback_equal,
        "evicted_shape_eager_fallback_exact_token_match": eviction_fallback_equal,
        "fallback_budget": fallback_budget,
        "graph_capture_count": pool.capture_count,
        "graph_cache_hit_count": pool.hit_count,
        "generated_ids": graph_tokens.cpu().tolist(),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--prefix-tokens", type=int, default=4)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
