"""Sequential multi-layer Qwen circuit-bank transfer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_qwen_parent_transplant import EVAL_TEXT, TRAIN_TEXT
from benchmark_qwen_two_layer_transplant import (
    MixedParentChild,
    benchmark_forward,
    capture_batches,
    ce,
    evaluate_current,
    make_child,
    token_stream,
    train_child,
)


def parse_layers(value: str) -> list[int]:
    layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(layers) < 2 or len(set(layers)) != len(layers):
        raise ValueError("layers must contain at least two distinct indices")
    return layers


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("install requirements-transfer.txt first") from exc

    layer_indices = parse_layers(args.layers)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
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
        teacher_logits_gpu = [
            model(input_ids=ids, use_cache=False).logits.detach()
            for ids in eval_ids
        ]
    teacher_ce = sum(
        ce(logits.float(), ids)
        for logits, ids in zip(teacher_logits_gpu, eval_ids)
    ) / len(eval_ids)
    # The vocabulary logits are large; keep them off the GPU while children
    # are trained and copy only an evaluation batch back when needed.
    teacher_logits = [
        logits.to(device="cpu", dtype=torch.float16)
        for logits in teacher_logits_gpu
    ]

    layers = [model.model.layers[index] for index in layer_indices]
    parents = [layer.mlp for layer in layers]
    hidden_size = int(model.config.hidden_size)
    children = []
    histories = []
    local_eval_mse = []

    for layer_index, layer in zip(layer_indices, layers):
        train_io = capture_batches(
            model, tokenizer, TRAIN_TEXT, args.batch_size,
            args.sequence_length, args.train_batches, device, layer_index,
        )
        eval_io = capture_batches(
            model, tokenizer, EVAL_TEXT, args.batch_size,
            args.sequence_length, args.eval_batches, device, layer_index,
        )
        child = make_child(
            hidden_size, args.inner_size, args.child_kind,
            args.calibration_rank, args.num_experts, args.active_experts,
            args.routing_temperature, device, dtype, args.dispatch_mode,
            not args.child_no_norm,
        )
        histories.append(train_child(
            child, train_io, device, dtype, args.steps,
            args.learning_rate, args.max_grad_norm, args.log_every,
        ))
        with torch.no_grad():
            mse = sum(
                F.mse_loss(
                    child(batch["input"].to(device=device, dtype=dtype)).float(),
                    batch["output"].to(device=device).float(),
                ).item()
                for batch in eval_io
            ) / len(eval_io)
        local_eval_mse.append(mse)
        children.append(child)
        # The next child is calibrated on the representation produced by all
        # previously trained children, matching the eventual cascade.
        layer.mlp = child

    variants = []
    for alpha in args.alphas:
        for layer, parent, child in zip(layers, parents, children):
            layer.mlp = MixedParentChild(parent, child, alpha)
        variants.append(evaluate_current(
            model, list(eval_ids), teacher_logits, teacher_ce,
            f"shared_alpha_{alpha:g}",
        ))
    for layer, parent in zip(layers, parents):
        layer.mlp = parent

    parent_timing = benchmark_forward(
        model, list(eval_ids), device,
        args.timing_warmup, args.timing_iterations,
    )
    for layer, child in zip(layers, children):
        layer.mlp = child
    sparse_timing = benchmark_forward(
        model, list(eval_ids), device,
        args.timing_warmup, args.timing_iterations,
    )
    for layer, parent in zip(layers, parents):
        layer.mlp = parent

    alpha_zero = next(
        item for item in variants if item["variant"] == "shared_alpha_0"
    )
    child_params = [sum(parameter.numel() for parameter in child.parameters()) for child in children]
    parent_params = [sum(parameter.numel() for parameter in parent.parameters()) for parent in parents]
    result = {
        "experiment": "qwen_multi_layer_attention_free_parent_transplant",
        "model": args.model,
        "model_path_exists": Path(args.model).exists(),
        "device": str(device),
        "dtype": args.dtype,
        "seed": args.seed,
        "layers": layer_indices,
        "hidden_size": hidden_size,
        "child_inner_size": args.inner_size,
        "child_kind": args.child_kind,
        "calibration_rank": args.calibration_rank,
        "num_experts": args.num_experts,
        "active_experts": args.active_experts,
        "hard_route_expected_expert_fraction": (
            args.active_experts / args.num_experts
            if args.child_kind == "routed" else 1.0
        ),
        "hard_route_dispatch": (
            f"selected-token-only:{args.dispatch_mode}"
            if args.child_kind == "routed" else "single-child"
        ),
        "dispatch_mode": args.dispatch_mode,
        "child_internal_norm": (
            not args.child_no_norm if args.child_kind == "routed" else None
        ),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "train_batches": args.train_batches,
        "eval_batches": args.eval_batches,
        "distillation_steps_per_child": args.steps,
        "teacher_ce": teacher_ce,
        "child_train_history": histories,
        "child_local_eval_mse": local_eval_mse,
        "parent_scalar_params_each": parent_params,
        "child_scalar_params_each": child_params,
        "child_parameter_fraction_each": [
            child_count / max(parent_count, 1)
            for child_count, parent_count in zip(child_params, parent_params)
        ],
        "variants": variants,
        "timing": {
            "parent": parent_timing,
            "sparse_bank": sparse_timing,
            "bank_over_parent_mean_ratio": (
                sparse_timing["mean_ms_per_batch"]
                / max(parent_timing["mean_ms_per_batch"], 1e-9)
            ),
            "warmup": args.timing_warmup,
            "iterations": args.timing_iterations,
            "note": "end-to-end Qwen forward with all selected layers replaced",
        },
        "quality_gate": {
            "criterion": "shared alpha=0 CE delta <= 0.05 and all outputs finite",
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
    parser.add_argument("--layers", default="23,24,25,26")
    parser.add_argument("--inner-size", type=int, default=384)
    parser.add_argument("--child-kind", choices=("gelu", "swiglu", "routed"), default="routed")
    parser.add_argument("--calibration-rank", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--routing-temperature", type=float, default=1.0)
    parser.add_argument(
        "--dispatch-mode", choices=("grouped", "packed", "token-loop"),
        default="grouped",
    )
    parser.add_argument("--child-no-norm", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--train-batches", type=int, default=8)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-ce-delta", type=float, default=0.05)
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 0.75, 0.5, 0.25, 0.0])
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="results/runs/qwen_multi_layer_transplant.json")
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
