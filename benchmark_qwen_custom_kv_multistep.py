"""Check multi-step prefill-to-decode replay with the fixed-position KV cache."""

from __future__ import annotations

import argparse
import json

import torch

from benchmark_qwen_custom_kv_graph import FixedDecodeCache, forward_logits, make_cache_and_fill_prefix
from benchmark_qwen_multi_layer_transplant import (
    TransferredRoutedQwenChild,
    make_transferred_routed_qwen_child,
)


DEFAULT_LAYERS = "19,20,21,22,23,24,25,26"


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
    parents = [model.model.layers[index].mlp for index in layers]
    for index, parent in zip(layers, parents):
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

    prefix_ids = tokenizer(
        "Neural Engine sparse circuits", return_tensors="pt",
    ).input_ids[:, :4].to(device)
    continuation = tokenizer(
        " routing systems can scale", return_tensors="pt",
    ).input_ids.to(device)
    continuation = continuation[:, :args.steps]
    if continuation.shape[1] < 2:
        raise ValueError("continuation must contain at least two tokens")
    prefix_length = int(prefix_ids.shape[1])
    cache_length = max(32, prefix_length + int(continuation.shape[1]) + 8)

    # Capture one fixed [batch=1, token=1] graph after prefix prefill.
    graph_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    graph_position = torch.tensor([prefix_length], device=device)
    graph_cache.set_decode_position(graph_position)
    static_token = continuation[:, :1].clone()
    with torch.inference_mode():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_logits = forward_logits(
                model, static_token, graph_cache, graph_position,
            )
        graph_outputs = []
        for step in range(continuation.shape[1]):
            if step > 0:
                static_token.copy_(continuation[:, step:step + 1])
                graph_position.copy_(
                    torch.tensor([prefix_length + step], device=device),
                )
            graph.replay()
            torch.cuda.synchronize()
            graph_outputs.append(graph_logits.clone())

    # Re-run the same sequence eagerly with independently filled KV storage.
    eager_cache = make_cache_and_fill_prefix(model, prefix_ids, cache_length)
    eager_outputs = []
    with torch.inference_mode():
        for step in range(continuation.shape[1]):
            eager_position = torch.tensor(
                [prefix_length + step], device=device,
            )
            eager_cache.set_decode_position(eager_position)
            eager_outputs.append(forward_logits(
                model, continuation[:, step:step + 1],
                eager_cache, eager_position,
            ).clone())
        torch.cuda.synchronize()

    errors = [
        float((graph_output - eager_output).abs().max().item())
        for graph_output, eager_output in zip(graph_outputs, eager_outputs)
    ]
    result = {
        "experiment": "V0.198_qwen_custom_kv_multistep_prefill_decode",
        "status": "PARITY_PASS" if max(errors) <= 1e-3 else "PARITY_FAIL",
        "model": args.model,
        "layers": layers,
        "active_experts": 5,
        "prefix_length": prefix_length,
        "decode_steps": int(continuation.shape[1]),
        "shape": [1, 1],
        "per_step_max_logit_error": errors,
        "max_logit_error": max(errors),
        "contract": "fixed-shape graph input/token and decode position updates after one prefix prefill",
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layers", default=DEFAULT_LAYERS)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--local-files-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
