"""Single-layer Qwen -> genuinely sparse attention-free circuit-bank gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from benchmark_qwen_parent_transplant import EVAL_TEXT, TRAIN_TEXT
from benchmark_qwen_two_layer_transplant import (
    MixedParentChild,
    RoutedChild,
    capture_batches,
    ce,
    evaluate_current,
    token_stream,
    train_child,
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_forward(
    model: nn.Module,
    input_ids: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("timing_iterations must be positive")
    for _ in range(warmup):
        for ids in input_ids:
            model(input_ids=ids, use_cache=False)
    synchronize(device)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        for ids in input_ids:
            model(input_ids=ids, use_cache=False)
        synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0 / len(input_ids))
    ordered = sorted(durations)
    return {
        "mean_ms_per_batch": sum(durations) / len(durations),
        "p50_ms_per_batch": ordered[len(ordered) // 2],
        "p95_ms_per_batch": ordered[max(0, int(len(ordered) * 0.95) - 1)],
    }


@torch.inference_mode()
def benchmark_module(
    module: nn.Module,
    inputs: list[torch.Tensor],
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    if iterations <= 0:
        raise ValueError("timing_iterations must be positive")
    for _ in range(warmup):
        for values in inputs:
            module(values)
    synchronize(device)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        for values in inputs:
            module(values)
        synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0 / len(inputs))
    ordered = sorted(durations)
    return {
        "mean_ms_per_batch": sum(durations) / len(durations),
        "p50_ms_per_batch": ordered[len(ordered) // 2],
        "p95_ms_per_batch": ordered[max(0, int(len(ordered) * 0.95) - 1)],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-transfer.txt first") from exc

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if not 1 <= args.active_experts <= args.num_experts:
        raise ValueError("active_experts must be within num_experts")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        trust_remote_code=False,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer or args.model,
        local_files_only=args.local_files_only,
    )
    train_ids = token_stream(
        tokenizer, TRAIN_TEXT, args.batch_size,
        args.sequence_length * args.train_batches, device,
    ).reshape(args.train_batches, args.batch_size, args.sequence_length)
    eval_ids = token_stream(
        tokenizer, EVAL_TEXT, args.batch_size,
        args.sequence_length * args.eval_batches, device,
    ).reshape(args.eval_batches, args.batch_size, args.sequence_length)
    with torch.no_grad():
        teacher_logits = [
            model(input_ids=ids, use_cache=False).logits.detach()
            for ids in eval_ids
        ]
    teacher_ce = sum(
        ce(logits.float(), ids) for logits, ids in zip(teacher_logits, eval_ids)
    ) / len(eval_ids)

    layer = model.model.layers[args.layer_index]
    parent = layer.mlp
    hidden_size = int(model.config.hidden_size)
    train_io = capture_batches(
        model, tokenizer, TRAIN_TEXT, args.batch_size, args.sequence_length,
        args.train_batches, device, args.layer_index,
    )
    eval_io = capture_batches(
        model, tokenizer, EVAL_TEXT, args.batch_size, args.sequence_length,
        args.eval_batches, device, args.layer_index,
    )
    child = RoutedChild(
        hidden_size, args.inner_size, args.num_experts, args.active_experts,
        args.routing_temperature, args.dispatch_mode,
    ).to(device=device, dtype=dtype)
    history = train_child(
        child, train_io, device, dtype, args.steps, args.learning_rate,
        args.max_grad_norm, args.log_every,
    )
    with torch.no_grad():
        local_mse = sum(
            F.mse_loss(
                child(batch["input"].to(device=device, dtype=dtype)).float(),
                batch["output"].to(device=device).float(),
            ).item()
            for batch in eval_io
        ) / len(eval_io)
    eval_inputs = [
        batch["input"].to(device=device, dtype=dtype) for batch in eval_io
    ]
    parent_ffn_timing = benchmark_module(
        parent, eval_inputs, device, args.timing_warmup, args.timing_iterations,
    )
    sparse_bank_ffn_timing = benchmark_module(
        child, eval_inputs, device, args.timing_warmup, args.timing_iterations,
    )

    variants = []
    for alpha in args.alphas:
        layer.mlp = MixedParentChild(parent, child, alpha)
        variants.append(
            evaluate_current(
                model, list(eval_ids), teacher_logits, teacher_ce,
                f"alpha_{alpha:g}",
            )
        )
    layer.mlp = parent
    alpha_zero = next(item for item in variants if item["variant"] == "alpha_0")
    layer.mlp = parent
    parent_timing = benchmark_forward(
        model, list(eval_ids), device, args.timing_warmup, args.timing_iterations,
    )
    layer.mlp = MixedParentChild(parent, child, 0.0)
    bank_timing = benchmark_forward(
        model, list(eval_ids), device, args.timing_warmup, args.timing_iterations,
    )
    layer.mlp = parent
    child_params = sum(parameter.numel() for parameter in child.parameters())
    parent_params = sum(parameter.numel() for parameter in parent.parameters())
    result = {
        "experiment": "qwen_single_layer_attention_free_circuit_bank",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layer_index": args.layer_index,
        "hidden_size": hidden_size,
        "child_inner_size": args.inner_size,
        "num_experts": args.num_experts,
        "active_experts": args.active_experts,
        "hard_route_expected_expert_fraction": args.active_experts / args.num_experts,
        "hard_route_dispatch": "selected-token-only",
        "dispatch_mode": args.dispatch_mode,
        "routing_temperature": args.routing_temperature,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "distillation_steps": args.steps,
        "teacher_ce": teacher_ce,
        "child_train_history": history,
        "child_local_eval_mse": local_mse,
        "parent_scalar_params": parent_params,
        "child_scalar_params": child_params,
        "child_parameter_fraction": child_params / max(parent_params, 1),
        "variants": variants,
        "timing": {
            "parent": parent_timing,
            "sparse_bank": bank_timing,
            "bank_over_parent_mean_ratio": (
                bank_timing["mean_ms_per_batch"]
                / max(parent_timing["mean_ms_per_batch"], 1e-9)
            ),
            "warmup": args.timing_warmup,
            "iterations": args.timing_iterations,
            "note": "end-to-end Qwen forward; no fused custom CUDA kernel",
        },
        "ffn_timing": {
            "parent": parent_ffn_timing,
            "sparse_bank": sparse_bank_ffn_timing,
            "bank_over_parent_mean_ratio": (
                sparse_bank_ffn_timing["mean_ms_per_batch"]
                / max(parent_ffn_timing["mean_ms_per_batch"], 1e-9)
            ),
            "note": "isolated layer MLP timing on captured Qwen hidden states",
        },
        "quality_gate": {
            "criterion": "alpha=0 child CE delta <= 0.05 and all outputs finite",
            "passed": bool(
                float(alpha_zero["ce_delta"]) <= args.max_ce_delta
                and torch.isfinite(torch.tensor(float(alpha_zero["ce"])))
            ),
            "max_ce_delta": args.max_ce_delta,
        },
    }
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--layer-index", type=int, default=26)
    parser.add_argument("--inner-size", type=int, default=384)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--active-experts", type=int, default=1)
    parser.add_argument("--routing-temperature", type=float, default=1.0)
    parser.add_argument("--dispatch-mode", choices=("packed", "token-loop"), default="token-loop")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_single_layer_bank.json")
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-iterations", type=int, default=30)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
