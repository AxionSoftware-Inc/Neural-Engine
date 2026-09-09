"""Check the fixed-shape graph path for a batch of two decode requests."""

from __future__ import annotations

import argparse
import json
import time

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

    single_prompt = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    batch_prompt = single_prompt.repeat(args.batch_size, 1)
    pool = FixedShapeGreedyGraphPool(model, max_entries=2)
    graph_tokens = pool.generate(batch_prompt, args.new_tokens, use_cuda_graph=True)
    reused_tokens = pool.generate(batch_prompt, args.new_tokens, use_cuda_graph=True)
    eager_tokens = greedy_generate_fixed_shape(
        model, batch_prompt, args.new_tokens, use_cuda_graph=False,
    )
    exact = bool(torch.equal(graph_tokens, eager_tokens))
    reused_exact = bool(torch.equal(graph_tokens, reused_tokens))

    def timed(callback) -> float:
        for _ in range(args.warmup):
            callback()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            callback()
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000.0 / args.iterations

    graph_reuse_ms = timed(
        lambda: pool.generate(
            batch_prompt, args.new_tokens, use_cuda_graph=True,
        )
    )
    eager_ms = timed(
        lambda: greedy_generate_fixed_shape(
            model, batch_prompt, args.new_tokens, use_cuda_graph=False,
        )
    )
    result = {
        "experiment": "V0.201_qwen_fixed_graph_batch_shape_audit",
        "status": "PARITY_PASS" if exact and reused_exact else "PARITY_FAIL",
        "batch_size": int(batch_prompt.shape[0]),
        "prefix_length": int(batch_prompt.shape[1]),
        "new_tokens": args.new_tokens,
        "graph_vs_eager_exact": exact,
        "reused_shape_exact": reused_exact,
        "graph_capture_count": pool.capture_count,
        "graph_cache_hit_count": pool.hit_count,
        "graph_reuse_ms": graph_reuse_ms,
        "eager_ms": eager_ms,
        "graph_over_eager": graph_reuse_ms / max(eager_ms, 1e-9),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
