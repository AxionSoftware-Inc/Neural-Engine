"""Measure shape-homogeneous micro-batching for native fused serving."""

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
    _get_json,
    _post_json,
    _stop_launcher,
    _wait_for_health,
)
from data.generator import SyntheticTaskGenerator


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _start_server(args: argparse.Namespace) -> tuple[subprocess.Popen[str], int]:
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
    rows: list[list[int]],
    microbatch_size: int,
) -> tuple[dict[str, Any], torch.Tensor, list[int]]:
    launcher, port = _start_server(args)
    base_url = f"http://127.0.0.1:{port}"
    groups = [
        rows[index:index + microbatch_size]
        for index in range(0, len(rows), microbatch_size)
    ]
    try:
        # Populate the exact graph shape before timing measured requests.
        _post_json(f"{base_url}/infer", groups[0])
        started = time.perf_counter()

        def request(group: list[list[int]]) -> tuple[dict[str, Any], float]:
            request_started = time.perf_counter()
            response = _post_json(f"{base_url}/infer", group)
            return response, (time.perf_counter() - request_started) * 1000.0

        with ThreadPoolExecutor(
            max_workers=min(args.client_workers, len(groups))
        ) as executor:
            responses = list(executor.map(request, groups))
        wall_ms = (time.perf_counter() - started) * 1000.0

        flat_logits = torch.cat(
            [torch.tensor(response[0]["logits"]) for response in responses], dim=0
        )
        flat_predictions = [
            prediction
            for response, _latency in responses
            for prediction in response["predictions"]
        ]
        health_after = _wait_for_health([port])[0]
        latencies = [latency for _response, latency in responses]
        result: dict[str, Any] = {
            "microbatch_size": microbatch_size,
            "logical_requests": len(rows),
            "http_requests": len(groups),
            "client_workers": min(args.client_workers, len(groups)),
            "request_wall_ms": wall_ms,
            "logical_requests_per_second": len(rows) / (wall_ms / 1000.0),
            "http_client_latency_ms": {
                "min": min(latencies),
                "mean": sum(latencies) / len(latencies),
                "p95": _p95(latencies),
                "max": max(latencies),
            },
            "cache": health_after["cache"],
        }
        return result, flat_logits, flat_predictions
    finally:
        _stop_launcher(launcher)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.logical_requests < 1 or args.client_workers < 1:
        raise ValueError("logical-requests and client-workers must be positive")
    if not args.microbatch_sizes:
        raise ValueError("microbatch-sizes must contain at least one size")
    if any(size < 1 for size in args.microbatch_sizes):
        raise ValueError("microbatch sizes must be positive")
    if any(args.logical_requests % size for size in args.microbatch_sizes):
        raise ValueError("logical-requests must be divisible by every microbatch size")

    generator = SyntheticTaskGenerator(seq_len=args.sequence_length, seed=args.seed)
    batch = generator.task_balanced_batch(args.logical_requests, "cpu")
    rows = batch.inputs[:, :args.sequence_length].tolist()
    scenarios: dict[str, dict[str, Any]] = {}
    reference_logits: torch.Tensor | None = None
    reference_predictions: list[int] | None = None
    for microbatch_size in args.microbatch_sizes:
        result, logits, predictions = _run_scenario(args, rows, microbatch_size)
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
        "experiment": "native_fused_shape_homogeneous_microbatch",
        "checkpoint": str(Path(args.checkpoint)),
        "device": args.device or "auto",
        "seed": args.seed,
        "logical_requests": args.logical_requests,
        "microbatch_sizes": args.microbatch_sizes,
        "sequence_length": args.sequence_length,
        "scenarios": scenarios,
        "interpretation": (
            "The first scenario is the reference only for output parity. Compare "
            "logical_requests_per_second and request_wall_ms across microbatch "
            "sizes; all requests use the same homogeneous sequence shape."
        ),
    }
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--logical-requests", type=int, default=16)
    parser.add_argument("--microbatch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=23004)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_microbatch_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
