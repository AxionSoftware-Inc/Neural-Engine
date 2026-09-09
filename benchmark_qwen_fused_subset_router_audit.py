"""Audit the opt-in fused hard-subset router used by trained K=5 children."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from benchmark_qwen_multi_layer_transplant import (
    make_transferred_routed_qwen_child,
)
from neural_engine.qwen_router_dispatch import fused_subset_router


def measure(operation, warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def make_children(model, layers, device):
    children = []
    for layer in layers:
        child = make_transferred_routed_qwen_child(
            model.model.layers[layer].mlp, 8, 5, 1.0, 0,
            "base-output", "cross-group", "grouped", "contiguous",
            "subset-router", 5.0, device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        generator = torch.Generator(device=device).manual_seed(10100 + layer)
        child.subset_router[-1].weight.data.normal_(
            mean=0.0, std=0.01, generator=generator,
        )
        child.subset_router[-1].bias.data.normal_(
            mean=0.0, std=0.01, generator=generator,
        )
        children.append(child)
    return children


def route_reference(child, state):
    subset_scores = child.subset_router(state)
    best_subset = subset_scores.argmax(dim=-1)
    membership = child.subset_membership[best_subset]
    scores = torch.where(membership.bool(), torch.ones_like(membership), -torch.ones_like(membership))
    top_values, top_ids = scores.topk(child.active_experts, dim=-1)
    return top_ids, F.softmax(top_values / child.temperature, dim=-1)


def route_fused(child, state):
    ids, weights = fused_subset_router(
        state.reshape(-1, state.shape[-1]).contiguous(),
        child.subset_router[0].weight,
        child.subset_router[0].bias,
        child.subset_router[2].weight,
        child.subset_router[2].bias,
        child.subset_membership,
        child.active_experts,
    )
    return ids.reshape(state.shape[0], 1, child.active_experts), weights.reshape(
        state.shape[0], 1, child.active_experts,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--layers", default="19,20,21,22,23,24,25,26")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
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
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    children = make_children(model, layers, device)
    hidden_size = int(model.config.hidden_size)
    records = []
    model_records = []
    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(10200 + batch_size)
        hidden = [
            torch.randn(
                batch_size, 1, hidden_size, device=device,
                dtype=torch.float32, generator=generator,
            )
            for _ in children
        ]
        references = [route_reference(child, state) for child, state in zip(children, hidden)]
        fused = [route_fused(child, state) for child, state in zip(children, hidden)]
        set_mismatches = sum(
            int(not torch.equal(reference[0].sort(dim=-1).values, candidate[0].sort(dim=-1).values))
            for reference, candidate in zip(references, fused)
        )
        weight_error = max(
            float((reference[1] - candidate[1]).abs().max().item())
            for reference, candidate in zip(references, fused)
        )
        for child in children:
            child.single_token_router_backend = "torch"
        torch_outputs = [child(state) for child, state in zip(children, hidden)]
        for child in children:
            child.single_token_router_backend = "cuda-fused-subset"
        fused_outputs = [child(state) for child, state in zip(children, hidden)]
        child_error = max(
            float((reference - candidate).abs().max().item())
            for reference, candidate in zip(torch_outputs, fused_outputs)
        )
        for child in children:
            child.single_token_router_backend = "torch"
        torch_route_ms = measure(
            lambda: [route_reference(child, state) for child, state in zip(children, hidden)],
            args.warmup, args.iterations,
        )
        for child in children:
            child.single_token_router_backend = "cuda-fused-subset"
        fused_route_ms = measure(
            lambda: [route_fused(child, state) for child, state in zip(children, hidden)],
            args.warmup, args.iterations,
        )
        records.append({
            "batch_size": batch_size,
            "torch_subset_route_ms": torch_route_ms,
            "fused_subset_route_ms": fused_route_ms,
            "fused_over_torch": fused_route_ms / max(torch_route_ms, 1e-9),
            "set_mismatched_layers": set_mismatches,
            "max_route_weight_error": weight_error,
            "max_child_output_error": child_error,
        })

        for layer, child in zip(layers, children):
            model.model.layers[layer].mlp = child
        token_generator = torch.Generator(device=device).manual_seed(10300 + batch_size)
        token_ids = torch.randint(
            0, int(model.config.vocab_size), (batch_size, 1),
            device=device, generator=token_generator,
        )

        def full_torch():
            for child in children:
                child.single_token_router_backend = "torch"
            return model(input_ids=token_ids, use_cache=False).logits

        def full_fused():
            for child in children:
                child.single_token_router_backend = "cuda-fused-subset"
            return model(input_ids=token_ids, use_cache=False).logits

        torch_logits = full_torch().clone()
        fused_logits = full_fused().clone()
        model_records.append({
            "batch_size": batch_size,
            "torch_model_ms": measure(full_torch, args.warmup, args.iterations),
            "fused_model_ms": measure(full_fused, args.warmup, args.iterations),
            "fused_over_torch": 0.0,
            "max_model_logit_error": float(
                (torch_logits - fused_logits).abs().max().item()
            ),
        })
        model_records[-1]["fused_over_torch"] = (
            model_records[-1]["fused_model_ms"]
            / max(model_records[-1]["torch_model_ms"], 1e-9)
        )

    result = {
        "experiment": "V0.221_fused_qwen_subset_router_audit",
        "model": args.model,
        "dtype": "float32",
        "layers": layers,
        "active_experts": 5,
        "num_experts": 8,
        "subset_count": 56,
        "router_probe": "deterministic non-tied subset-router final weights",
        "warmup": args.warmup,
        "iterations": args.iterations,
        "records": records,
        "model_records": model_records,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")


if __name__ == "__main__":
    main()
