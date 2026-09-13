"""Paired runtime A/B for serial circuit dispatch implementations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import make_model


def forward(model: torch.nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    logits, _ = model(
        inputs,
        collect_state_stats=False,
        return_full_logits=model.output_mode != "factorized_digits",
    )
    return logits


@torch.inference_mode()
def measure(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    dispatch: str,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model.circuits.serial_dispatch = dispatch
    for _ in range(warmup):
        forward(model, inputs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        forward(model, inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / iterations
    return {
        "dispatch": dispatch,
        "latency_ms_per_batch": latency_ms,
        "samples_per_second": inputs.shape[0] / (latency_ms / 1000.0),
    }


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    if config.get("architecture") != "dynamic_register":
        raise ValueError("paired serial dispatch benchmark requires a dynamic checkpoint")
    model = make_model(config).cuda().eval()
    model.load_state_dict(payload["model_state"])
    report = model.parameter_report()
    results = []
    for batch_size, iterations in zip(args.batch_size, args.iterations):
        generator = DynamicCompositionGenerator(
            max_ops=int(config["max_ops"]),
            train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
            split=str(config.get("eval_split", "all")),
            seed=int(config["seed"]) + 1700 + batch_size,
            value_min=int(config.get("eval_value_min", 0)),
            value_max=int(config.get("eval_value_max", 63)),
        )
        inputs = generator.task_balanced_batch(batch_size, torch.device("cuda")).inputs
        model.circuits.serial_dispatch = "einsum"
        reference = forward(model, inputs)
        torch.cuda.synchronize()
        model.circuits.serial_dispatch = "bmm"
        optimized = forward(model, inputs)
        torch.cuda.synchronize()
        difference = (reference - optimized).abs()
        paired = []
        for round_index in range(args.rounds):
            order = ("einsum", "bmm") if round_index % 2 == 0 else ("bmm", "einsum")
            for dispatch in order:
                paired.append(measure(model, inputs, dispatch, args.warmup, iterations))
        by_dispatch = {
            dispatch: [row for row in paired if row["dispatch"] == dispatch]
            for dispatch in ("einsum", "bmm")
        }
        summary = {}
        for dispatch, rows in by_dispatch.items():
            latency = sum(row["latency_ms_per_batch"] for row in rows) / len(rows)
            summary[dispatch] = {
                "mean_latency_ms_per_batch": latency,
                "mean_samples_per_second": batch_size / (latency / 1000.0),
                "rounds": rows,
            }
        results.append({
            "batch_size": batch_size,
            "iterations_per_round": iterations,
            "rounds": args.rounds,
            "max_abs_output_difference": float(difference.max().cpu()),
            "mean_abs_output_difference": float(difference.mean().cpu()),
            "outputs_allclose": bool(torch.allclose(reference, optimized, atol=1e-5, rtol=1e-5)),
            "summary": summary,
        })
    output = {
        "benchmark": "v0.336_serial_dispatch_paired_ab",
        "checkpoint": args.checkpoint,
        "device": "cuda",
        "total_params": report["total_params"],
        "active_params_estimate": report["active_params_estimate"],
        "results": results,
    }
    encoded = json.dumps(output, indent=2)
    print(encoded)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(encoded + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired serial dispatch runtime A/B")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[1, 128])
    parser.add_argument("--iterations", type=int, nargs="+", default=[100, 30])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="results/runs/v0_336_serial_dispatch_paired_ab.json")
    args = parser.parse_args()
    if len(args.batch_size) != len(args.iterations):
        parser.error("--batch-size and --iterations must have equal lengths")
    run(args)


if __name__ == "__main__":
    main()
