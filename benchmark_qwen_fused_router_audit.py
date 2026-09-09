"""Audit the opt-in fused Qwen router against the PyTorch route."""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from benchmark_qwen_multi_layer_transplant import (
    make_transferred_routed_qwen_child,
)
from neural_engine.qwen_router_dispatch import fused_router


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
    layers = [int(value) for value in args.layers.split(",") if value.strip()]
    children = []
    for layer in layers:
        child = make_transferred_routed_qwen_child(
            model.model.layers[layer].mlp, 8, 6, 1.0, 0, "base-output",
            "low-rank", "grouped", "contiguous", "router", 6.0,
            device, torch.float32,
        ).eval()
        child.single_token_fast_path = True
        # Fresh children intentionally have zero final router weights.  Give
        # the probe a deterministic non-tied trained-like router so route
        # ordering, rather than zero-logit tie behavior, is audited.
        generator = torch.Generator(device=device).manual_seed(9100 + layer)
        child.router[-1].weight.data.normal_(
            mean=0.0, std=0.01, generator=generator,
        )
        child.router[-1].bias.data.normal_(
            mean=0.0, std=0.01, generator=generator,
        )
        children.append(child)

    records = []
    model_records = []
    hidden_size = int(model.config.hidden_size)
    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(9200 + batch_size)
        hidden = [
            torch.randn(
                batch_size, 1, hidden_size, device=device,
                dtype=torch.float32, generator=generator,
            )
            for _ in children
        ]

        def torch_route():
            result = []
            for child, state in zip(children, hidden):
                scores = child.router(state)
                values, ids = scores.topk(child.active_experts, dim=-1)
                result.append((ids, F.softmax(values / child.temperature, dim=-1)))
            return result

        def fused_route():
            result = []
            for child, state in zip(children, hidden):
                flat_ids, flat_weights = fused_router(
                    state.reshape(-1, hidden_size).contiguous(),
                    child.router[0].weight,
                    child.router[0].bias,
                    child.router[2].weight,
                    child.router[2].bias,
                    child.active_experts,
                    child.temperature,
                )
                result.append((flat_ids, flat_weights))
            return result

        torch_routes = torch_route()
        fused_routes = fused_route()
        id_mismatches = sum(
            int(not torch.equal(reference[0].reshape_as(candidate[0]), candidate[0]))
            for reference, candidate in zip(torch_routes, fused_routes)
        )
        weight_error = max(
            float((reference[1].reshape_as(candidate[1]) - candidate[1]).abs().max().item())
            for reference, candidate in zip(torch_routes, fused_routes)
        )
        for child in children:
            child.single_token_router_backend = "torch"
        torch_outputs = [child(state) for child, state in zip(children, hidden)]
        for child in children:
            child.single_token_router_backend = "cuda-fused"
        fused_outputs = [child(state) for child, state in zip(children, hidden)]
        child_output_error = max(
            float((reference - candidate).abs().max().item())
            for reference, candidate in zip(torch_outputs, fused_outputs)
        )

        for child in children:
            child.single_token_router_backend = "torch"
        torch_child_ms = measure(
            lambda: [child(state) for child, state in zip(children, hidden)],
            args.warmup, args.iterations,
        )
        for child in children:
            child.single_token_router_backend = "cuda-fused"
        fused_child_ms = measure(
            lambda: [child(state) for child, state in zip(children, hidden)],
            args.warmup, args.iterations,
        )
        torch_ms = measure(torch_route, args.warmup, args.iterations)
        fused_ms = measure(fused_route, args.warmup, args.iterations)
        records.append({
            "batch_size": batch_size,
            "torch_router_topk_ms": torch_ms,
            "fused_router_topk_ms": fused_ms,
            "fused_over_torch": fused_ms / max(torch_ms, 1e-9),
            "torch_child_forward_ms": torch_child_ms,
            "fused_child_forward_ms": fused_child_ms,
            "fused_child_over_torch": fused_child_ms / max(torch_child_ms, 1e-9),
            "id_mismatched_layers": id_mismatches,
            "max_route_weight_error": weight_error,
            "max_child_output_error": child_output_error,
        })

        # End-to-end one-token smoke with the same non-tied router probe.  The
        # body and all unmodified Transformer layers remain the parent model;
        # this is a runtime/parity check, not a language-quality benchmark.
        for layer, child in zip(layers, children):
            model.model.layers[layer].mlp = child
        token_generator = torch.Generator(device=device).manual_seed(9300 + batch_size)
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
                child.single_token_router_backend = "cuda-fused"
            return model(input_ids=token_ids, use_cache=False).logits

        torch_logits = full_torch().clone()
        fused_logits = full_fused().clone()
        model_records.append({
            "batch_size": batch_size,
            "torch_model_ms": measure(full_torch, args.warmup, args.iterations),
            "fused_model_ms": measure(full_fused, args.warmup, args.iterations),
            "max_model_logit_error": float(
                (torch_logits - fused_logits).abs().max().item()
            ),
        })

    result = {
        "experiment": "V0.220_fused_qwen_router_audit",
        "model": args.model,
        "dtype": "float32",
        "layers": layers,
        "active_experts": 6,
        "num_experts": 8,
        "router_probe": "deterministic non-tied final router weights",
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
