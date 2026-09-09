"""Check fixed-graph reuse across two validated prefix-length shapes."""

from __future__ import annotations

import argparse
import json

import torch

from benchmark_qwen_multi_layer_transplant import (
    TransferredRoutedQwenChild,
    make_transferred_routed_qwen_child,
)
from neural_engine.qwen_fixed_graph import (
    FixedShapeGreedyGraphPool,
    greedy_generate_fixed_shape,
)


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
    layers = [int(item) for item in args.layers.split(",") if item.strip()]
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

    short_prompt = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    long_prompt = tokenizer(
        "Neural Engine sparse circuits support efficient reasoning", return_tensors="pt",
    ).input_ids[:, :8].to(device)
    pool = FixedShapeGreedyGraphPool(model, max_entries=3)
    short_graph = pool.generate(short_prompt, args.new_tokens, use_cuda_graph=True)
    long_graph = pool.generate(long_prompt, args.new_tokens, use_cuda_graph=True)
    short_eager = greedy_generate_fixed_shape(
        model, short_prompt, args.new_tokens, use_cuda_graph=False,
    )
    long_eager = greedy_generate_fixed_shape(
        model, long_prompt, args.new_tokens, use_cuda_graph=False,
    )
    short_graph_reused = pool.generate(
        short_prompt, args.new_tokens, use_cuda_graph=True,
    )
    short_equal = bool(torch.equal(short_graph, short_eager))
    long_equal = bool(torch.equal(long_graph, long_eager))
    reuse_equal = bool(torch.equal(short_graph, short_graph_reused))
    result = {
        "experiment": "V0.200_qwen_fixed_graph_prefix_shape_audit",
        "status": "PARITY_PASS" if short_equal and long_equal and reuse_equal else "PARITY_FAIL",
        "prefix_lengths": [int(short_prompt.shape[1]), int(long_prompt.shape[1])],
        "new_tokens": args.new_tokens,
        "short_prefix_graph_vs_eager_exact": short_equal,
        "long_prefix_graph_vs_eager_exact": long_equal,
        "short_prefix_reuse_exact": reuse_equal,
        "graph_capture_count": pool.capture_count,
        "graph_cache_hit_count": pool.hit_count,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
