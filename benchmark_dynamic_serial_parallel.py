"""Inference-only serial versus parallel circuit-composition control."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import make_model


def forward(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    return model(
        inputs,
        collect_state_stats=False,
        return_full_logits=model.output_mode != "factorized_digits",
    )


def predictions(model: torch.nn.Module, stats: dict[str, Any]) -> torch.Tensor:
    digit_logits = stats["digit_logits"]
    base = model.output_digit_base
    result = torch.zeros(
        digit_logits[0].shape[0], dtype=torch.long, device=digit_logits[0].device
    )
    for index, logits in enumerate(digit_logits):
        result = result + logits[:, -1].argmax(dim=-1) * (
            base ** (len(digit_logits) - 1 - index)
        )
    return result


@torch.inference_mode()
def measure(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.circuit_mode = mode
    for _ in range(warmup):
        forward(model, inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        forward(model, inputs)
    torch.cuda.synchronize()
    latency = (time.perf_counter() - start) * 1000.0 / iterations
    _, stats = forward(model, inputs)
    predicted = predictions(model, stats)
    return {
        "mode": mode,
        "latency_ms_per_batch": latency,
        "samples_per_second": inputs.shape[0] / (latency / 1000.0),
        "accuracy": float(predicted.eq(targets).float().mean().cpu()),
        "avg_executed_steps": float(stats["executed_steps"].float().mean().cpu()),
        "router_entropy": float(stats["router_entropy"].cpu()),
        "predictions": predicted,
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    if config.get("architecture") != "dynamic_register":
        raise ValueError("this control requires a dynamic-register checkpoint")
    model = make_model(config).cuda().eval()
    model.load_state_dict(payload["model_state"])
    generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        split=str(config.get("eval_split", "all")),
        seed=int(config["seed"]) + 1813,
        value_min=args.value_min,
        value_max=args.value_max,
        modulus=(
            None if config.get("modulus") is None
            else int(config["modulus"])
        ),
        target_offset=int(config.get("target_offset", 0)),
        fixed_operation=args.fixed_operation,
    )
    if args.depth is None:
        batch = generator.task_balanced_batch(args.batch_size, torch.device("cuda"))
    else:
        rows = []
        for _ in range(args.batch_size):
            tokens, target, sequence_id, stage_targets, stage_mask = generator._one(args.depth)
            rows.append((tokens, target, sequence_id, args.depth, stage_targets, stage_mask))
        batch = generator._make_batch(rows, torch.device("cuda"))
    measured = {"serial": [], "parallel": []}
    for round_index in range(args.rounds):
        order = ("serial", "parallel") if round_index % 2 == 0 else ("parallel", "serial")
        for mode in order:
            measured[mode].append(
                measure(model, batch.inputs, batch.targets, mode, args.warmup, args.iterations)
            )
    serial_predictions = measured["serial"][-1].pop("predictions")
    parallel_predictions = measured["parallel"][-1].pop("predictions")
    model.circuit_mode = "serial"
    _, serial_stats = forward(model, batch.inputs)
    model.circuit_mode = "parallel"
    _, parallel_stats = forward(model, batch.inputs)
    digit_output_difference = max(
        (left - right).abs().max().item()
        for left, right in zip(serial_stats["digit_logits"], parallel_stats["digit_logits"])
    )

    summaries = {}
    for mode, rows in measured.items():
        latency = sum(row["latency_ms_per_batch"] for row in rows) / len(rows)
        for row in rows:
            row.pop("predictions", None)
        summaries[mode] = {
            "mean_latency_ms_per_batch": latency,
            "mean_samples_per_second": args.batch_size / (latency / 1000.0),
            "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
            "rounds": rows,
        }
    output = {
        "benchmark": "v0.338_dynamic_serial_parallel_inference_control",
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "depth": args.depth,
        "fixed_operation": args.fixed_operation,
        "value_range": [args.value_min, args.value_max],
        "iterations": args.iterations,
        "digit_output_max_abs_difference": digit_output_difference,
        "prediction_agreement": float(
            serial_predictions.eq(parallel_predictions).float().mean().cpu()
        ),
        "serial": summaries["serial"],
        "parallel": summaries["parallel"],
    }
    encoded = json.dumps(output, indent=2)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Serial/parallel dynamic circuit control")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--fixed-operation", choices=("add", "subtract", "multiply"), default=None
    )
    parser.add_argument("--value-min", type=int, default=0)
    parser.add_argument("--value-max", type=int, default=63)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--output", default="results/runs/v0_338_serial_parallel_control.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
