"""Measure shape-bucketed micro-batching for mixed sequence lengths."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import torch

from benchmark_native_workers import (
    _free_consecutive_ports,
    _post_json,
    _stop_launcher,
    _wait_for_health,
)
from data.generator import SyntheticTaskGenerator


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _start_server(args: argparse.Namespace):
    root = Path(__file__).resolve().parent
    port = _free_consecutive_ports(1)
    command = [
        sys.executable,
        str(root / "serve_native_workers.py"),
        "--checkpoint",
        str(Path(args.checkpoint)),
        "--workers",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-shapes",
        str(args.max_shapes),
        "--warmup-iters",
        str(args.warmup_iters),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.no_graphs:
        command.append("--no-graphs")
    if args.skip_native_prebuild:
        command.append("--skip-native-prebuild")
    launcher = subprocess.Popen(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health([port])
    except Exception:
        _stop_launcher(launcher)
        raise
    return launcher, port


def _run_scenario(
    args: argparse.Namespace,
    records: list[tuple[str, list[int]]],
    microbatch_size: int,
) -> tuple[dict[str, Any], torch.Tensor, list[int]]:
    launcher, port = _start_server(args)
    try:
        base_url = f"http://127.0.0.1:{port}"
        buckets: dict[str, list[tuple[int, list[int]]]] = {}
        for index, (shape, row) in enumerate(records):
            buckets.setdefault(shape, []).append((index, row))
        groups: list[tuple[str, list[int], list[list[int]]]] = []
        for shape, items in buckets.items():
            for start in range(0, len(items), microbatch_size):
                chunk = items[start:start + microbatch_size]
                groups.append((shape, [item[0] for item in chunk], [item[1] for item in chunk]))

        # Warm each shape at the exact batch shape used by the first group.
        for _shape, _indices, group_rows in groups:
            # There is one first group per bucket; later groups are measured.
            if _indices[0] == buckets[_shape][0][0]:
                _post_json(f"{base_url}/infer", group_rows)

        started = time.perf_counter()

        def request(group: tuple[str, list[int], list[list[int]]]):
            shape, indices, group_rows = group
            request_started = time.perf_counter()
            response = _post_json(f"{base_url}/infer", group_rows)
            return shape, indices, response, (time.perf_counter() - request_started) * 1000.0

        with ThreadPoolExecutor(
            max_workers=min(args.client_workers, len(groups))
        ) as executor:
            responses = list(executor.map(request, groups))
        wall_ms = (time.perf_counter() - started) * 1000.0

        ordered_logits: list[torch.Tensor | None] = [None] * len(records)
        ordered_predictions: list[int | None] = [None] * len(records)
        latencies: list[float] = []
        for _shape, indices, response, latency in responses:
            latencies.append(latency)
            logits = torch.tensor(response["logits"])
            for offset, index in enumerate(indices):
                ordered_logits[index] = logits[offset]
                ordered_predictions[index] = response["predictions"][offset]
        if any(item is None for item in ordered_logits):
            raise RuntimeError("mixed microbatch left an output slot empty")
        if any(item is None for item in ordered_predictions):
            raise RuntimeError("mixed microbatch left a prediction slot empty")
        health_after = _wait_for_health([port])[0]
        result: dict[str, Any] = {
            "microbatch_size": microbatch_size,
            "logical_requests": len(records),
            "http_requests": len(groups),
            "groups_by_shape": {
                shape: sum(1 for item in groups if item[0] == shape)
                for shape in buckets
            },
            "client_workers": min(args.client_workers, len(groups)),
            "request_wall_ms": wall_ms,
            "logical_requests_per_second": len(records) / (wall_ms / 1000.0),
            "http_client_latency_ms": {
                "min": min(latencies),
                "mean": sum(latencies) / len(latencies),
                "p95": _p95(latencies),
                "max": max(latencies),
            },
            "cache": health_after["cache"],
        }
        return result, torch.stack([item for item in ordered_logits if item is not None]), [
            int(item) for item in ordered_predictions if item is not None
        ]
    finally:
        _stop_launcher(launcher)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.logical_requests < 1 or args.logical_requests % 2:
        raise ValueError("logical-requests must be a positive even number")
    if args.client_workers < 1:
        raise ValueError("client-workers must be positive")
    if not args.sequence_lengths or len(args.sequence_lengths) != 2:
        raise ValueError("exactly two sequence lengths are required")
    if any(size < 1 for size in args.microbatch_sizes):
        raise ValueError("microbatch sizes must be positive")

    generator = SyntheticTaskGenerator(seq_len=max(args.sequence_lengths), seed=args.seed)
    records: list[tuple[str, list[int]]] = []
    for index in range(args.logical_requests):
        shape_index = index % 2
        sequence_length = args.sequence_lengths[shape_index]
        batch = generator.task_balanced_batch(1, "cpu")
        records.append((f"s{sequence_length}", batch.inputs[0, :sequence_length].tolist()))
    scenarios: dict[str, dict[str, Any]] = {}
    reference_logits: torch.Tensor | None = None
    reference_predictions: list[int] | None = None
    for microbatch_size in args.microbatch_sizes:
        result, logits, predictions = _run_scenario(args, records, microbatch_size)
        if reference_logits is None:
            reference_logits = logits
            reference_predictions = predictions
        result["max_logit_error_vs_first_scenario"] = float(
            (logits - reference_logits).abs().max()
        )
        result["prediction_mismatches_vs_first_scenario"] = sum(
            actual != expected
            for actual, expected in zip(predictions, reference_predictions or [])
        )
        scenarios[str(microbatch_size)] = result

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    final: dict[str, Any] = {
        "experiment": "native_fused_shape_bucketed_mixed_microbatch",
        "checkpoint": str(Path(args.checkpoint)),
        "device": args.device or "auto",
        "seed": args.seed,
        "logical_requests": args.logical_requests,
        "sequence_lengths": args.sequence_lengths,
        "microbatch_sizes": args.microbatch_sizes,
        "scenarios": scenarios,
        "interpretation": (
            "Requests are bucketed by sequence length before grouping. The first "
            "scenario is the parity reference; compare logical throughput and "
            "shape-specific cache entries across microbatch sizes."
        ),
    }
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--logical-requests", type=int, default=32)
    parser.add_argument("--sequence-lengths", nargs=2, type=int, default=[6, 32])
    parser.add_argument("--microbatch-sizes", nargs="+", type=int, default=[1, 4, 8])
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=23005)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_mixed_microbatch_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
