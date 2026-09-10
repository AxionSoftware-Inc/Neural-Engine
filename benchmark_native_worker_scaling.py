"""Compare native fused serving with one versus multiple workers on one GPU."""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

from benchmark_native_workers import run as run_multi_worker


def _scenario_args(args: argparse.Namespace, workers: int, output: Path) -> Namespace:
    return Namespace(
        checkpoint=args.checkpoint,
        workers=workers,
        client_workers=args.client_workers,
        requests=args.requests,
        port=0,
        device=args.device,
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        sequence_length=args.sequence_length,
        max_shapes=args.max_shapes,
        warmup_iters=args.warmup_iters,
        no_graphs=args.no_graphs,
        skip_native_prebuild=args.skip_native_prebuild,
        seed=args.seed,
        output=str(output),
    )


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "workers": result["workers"],
        "client_workers": result["client_workers"],
        "requests": result["requests"],
        "requests_per_worker": result["requests_per_worker"],
        "request_wall_ms": result["request_wall_ms"],
        "client_latency_ms": result["client_latency_ms"],
        "max_cross_worker_logit_error": result["max_cross_worker_logit_error"],
        "prediction_mismatches": result["prediction_mismatches"],
        "worker_cache": result["worker_cache"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.worker_counts:
        raise ValueError("worker-counts must contain at least one worker count")
    if any(count < 1 for count in args.worker_counts):
        raise ValueError("worker counts must be positive")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scenarios: dict[str, dict[str, Any]] = {}
    for workers in args.worker_counts:
        scenario_output = output.with_name(
            f"{output.stem}.workers{workers}{output.suffix}"
        )
        result = run_multi_worker(_scenario_args(args, workers, scenario_output))
        scenarios[str(workers)] = _summary(result)

    final = {
        "experiment": "native_fused_worker_scaling_same_gpu",
        "checkpoint": str(Path(args.checkpoint)),
        "device": args.device or "auto",
        "worker_counts": args.worker_counts,
        "client_workers": args.client_workers,
        "requests": args.requests,
        "batch_sizes": args.batch_sizes,
        "sequence_lengths": args.sequence_lengths,
        "seed": args.seed,
        "scenarios": scenarios,
        "interpretation": (
            "Compare request wall time and client latency at fixed client concurrency. "
            "When CUDA graphs are enabled, multi-worker serving on one physical GPU "
            "uses a device-wide lock for correctness, so additional workers may not "
            "increase throughput without batching or separate GPUs."
        ),
    }
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--worker-counts", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 32])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=23003)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_worker_scaling_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
