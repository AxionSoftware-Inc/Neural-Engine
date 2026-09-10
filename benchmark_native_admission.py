"""Compare direct HTTP serving with the optional shape-homogeneous batcher."""

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


def _start_server(args: argparse.Namespace, batch_window_ms: float):
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
        "--max-batch-size",
        str(args.max_batch_size),
        "--batch-window-ms",
        str(batch_window_ms),
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


def _run_mode(args: argparse.Namespace, rows: list[list[int]],
              batch_window_ms: float,
              prewarm_sizes: list[int]) -> tuple[dict[str, Any], torch.Tensor, list[int]]:
    launcher, port = _start_server(args, batch_window_ms)
    try:
        base_url = f"http://127.0.0.1:{port}"
        for size in prewarm_sizes:
            _post_json(f"{base_url}/infer", rows[:size])
        started = time.perf_counter()

        def request(row: list[int]) -> tuple[dict[str, Any], float]:
            request_started = time.perf_counter()
            response = _post_json(f"{base_url}/infer", [row])
            return response, (time.perf_counter() - request_started) * 1000.0

        with ThreadPoolExecutor(max_workers=args.client_workers) as executor:
            responses = list(executor.map(request, rows))
        wall_ms = (time.perf_counter() - started) * 1000.0
        logits = torch.cat(
            [torch.tensor(response[0]["logits"]) for response in responses], dim=0
        )
        predictions = [response[0]["predictions"][0] for response in responses]
        health_after = _wait_for_health([port])[0]
        latencies = [latency for _response, latency in responses]
        result: dict[str, Any] = {
            "batch_window_ms": batch_window_ms,
            "prewarm_sizes": prewarm_sizes,
            "logical_requests": len(rows),
            "http_requests": len(rows),
            "client_workers": args.client_workers,
            "request_wall_ms": wall_ms,
            "logical_requests_per_second": len(rows) / (wall_ms / 1000.0),
            "client_latency_ms": {
                "min": min(latencies),
                "mean": sum(latencies) / len(latencies),
                "p95": _p95(latencies),
                "max": max(latencies),
            },
            "cache": health_after["cache"],
            "batching": health_after["batching"],
        }
        return result, logits, predictions
    finally:
        _stop_launcher(launcher)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.logical_requests < 1 or args.client_workers < 1:
        raise ValueError("logical-requests and client-workers must be positive")
    if args.batch_window_ms <= 0:
        raise ValueError("batch-window-ms must be positive")
    if args.max_batch_size < 1:
        raise ValueError("max-batch-size must be positive")

    generator = SyntheticTaskGenerator(seq_len=args.sequence_length, seed=args.seed)
    batch = generator.task_balanced_batch(args.logical_requests, "cpu")
    rows = batch.inputs[:, :args.sequence_length].tolist()
    direct, reference_logits, reference_predictions = _run_mode(args, rows, 0.0, [1])
    admitted, admitted_logits, admitted_predictions = _run_mode(
        args, rows, args.batch_window_ms, list(range(1, args.max_batch_size + 1))
    )
    admitted["max_logit_error_vs_direct"] = float(
        (admitted_logits - reference_logits).abs().max()
    )
    admitted["prediction_mismatches_vs_direct"] = sum(
        actual != expected
        for actual, expected in zip(admitted_predictions, reference_predictions)
    )
    final: dict[str, Any] = {
        "experiment": "native_fused_shape_homogeneous_admission",
        "checkpoint": str(Path(args.checkpoint)),
        "device": args.device or "auto",
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "logical_requests": args.logical_requests,
        "client_workers": args.client_workers,
        "max_batch_size": args.max_batch_size,
        "batch_window_ms": args.batch_window_ms,
        "direct": direct,
        "admitted": admitted,
        "interpretation": (
            "Both modes send the same individual requests. The admitted mode "
            "coalesces compatible requests inside the bounded sequence-shape "
            "window; output parity is checked against the direct mode."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--logical-requests", type=int, default=32)
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--batch-window-ms", type=float, default=2.0)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=23004)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_admission_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
