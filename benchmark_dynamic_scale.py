from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from neural_engine.dynamic_register import DynamicRegisterNeuralEngine
from neural_engine.instrumentation import count_parameters


def make_model(num_circuits: int) -> DynamicRegisterNeuralEngine:
    factor_count = math.ceil(math.sqrt(num_circuits))
    return DynamicRegisterNeuralEngine(
        max_ops=6,
        seq_len=14,
        d_model=384,
        state_dim=384,
        num_circuits=num_circuits,
        circuit_rank=16,
        router_branch=8,
        router_depth=6,
        candidate_pool=32,
        active_circuits=8,
        circuit_bank_mode="factorized",
        factor_count=factor_count,
        factor_candidate_pool=8,
        circuit_mode="serial",
        route_exploration_prob=0.0,
        factor_mix_mode="shared",
    )


@torch.no_grad()
def benchmark_one(
    num_circuits: int,
    device: torch.device,
    batch_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    model = make_model(num_circuits).to(device).eval()
    generator = DynamicCompositionGenerator(
        max_ops=6, train_max_ops=4, split="all", seed=1700 + num_circuits
    )
    batch = generator.batch(batch_size, device)
    for _ in range(warmup):
        model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    durations = []
    for _ in range(iterations):
        start = time.perf_counter()
        model(batch.inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0)
    report = model.parameter_report()
    result = {
        "virtual_circuits": num_circuits,
        "factor_count": math.ceil(math.sqrt(num_circuits)),
        "total_params": count_parameters(model),
        "active_params_estimate": report["active_params_estimate"],
        "active_fraction": report["active_fraction"],
        "batch_size": batch_size,
        "iterations": iterations,
        "mean_forward_ms": sum(durations) / len(durations),
        "p50_forward_ms": sorted(durations)[len(durations) // 2],
    }
    if device.type == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    results = [
        benchmark_one(size, device, args.batch_size, args.warmup, args.iterations)
        for size in args.virtual_circuits
    ]
    output = {
        "device": str(device),
        "results": results,
        "note": "factorized virtual bank; forward-only systems stress test",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress-test virtual dynamic circuit-bank scaling")
    parser.add_argument("--virtual-circuits", type=int, nargs="+", default=[100_000, 250_000, 500_000, 1_000_000])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/runs/ne_dynamic_scale_stress.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
