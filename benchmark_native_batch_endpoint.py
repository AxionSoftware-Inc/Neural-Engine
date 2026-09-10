"""Benchmark the explicit shape-bucketed ``/infer_batch`` endpoint."""

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
from urllib.request import Request, urlopen

import torch

from benchmark_native_workers import (
    _free_consecutive_ports,
    _post_json,
    _stop_launcher,
    _wait_for_health,
)
from data.generator import SyntheticTaskGenerator


def _post_batch(url: str, requests: list[dict[str, Any]], max_batch_size: int) -> dict[str, Any]:
    body = json.dumps({
        "requests": requests,
        "max_batch_size": max_batch_size,
    }).encode("utf-8")
    request = Request(
        f"{url}/infer_batch",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        return json.loads(response.read().decode("utf-8"))


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


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return ordered[index]


def _run_direct(args: argparse.Namespace, rows_by_shape: list[tuple[str, list[int]]]):
    launcher, port = _start_server(args)
    try:
        base_url = f"http://127.0.0.1:{port}"
        first_by_shape: dict[str, list[int]] = {}
        for shape, row in rows_by_shape:
            first_by_shape.setdefault(shape, row)
        for row in first_by_shape.values():
            _post_json(f"{base_url}/infer", [row])
        started = time.perf_counter()

        def request(item: tuple[str, list[int]]) -> tuple[dict[str, Any], float]:
            _shape, row = item
            request_started = time.perf_counter()
            response = _post_json(f"{base_url}/infer", [row])
            return response, (time.perf_counter() - request_started) * 1000.0

        with ThreadPoolExecutor(max_workers=args.client_workers) as executor:
            responses = list(executor.map(request, rows_by_shape))
        wall_ms = (time.perf_counter() - started) * 1000.0
        logits = torch.cat(
            [torch.tensor(response[0]["logits"]) for response in responses], dim=0
        )
        predictions = [response[0]["predictions"][0] for response in responses]
        health = _wait_for_health([port])[0]
        latencies = [latency for _response, latency in responses]
        return {
            "http_requests": len(rows_by_shape),
            "client_workers": args.client_workers,
            "request_wall_ms": wall_ms,
            "logical_requests_per_second": len(rows_by_shape) / (wall_ms / 1000.0),
            "client_latency_ms": {
                "min": min(latencies),
                "mean": sum(latencies) / len(latencies),
                "p95": _p95(latencies),
                "max": max(latencies),
            },
            "cache": health["cache"],
        }, logits, predictions
    finally:
        _stop_launcher(launcher)


def _run_batch_endpoint(args: argparse.Namespace,
                        requests: list[dict[str, Any]]):
    launcher, port = _start_server(args)
    try:
        base_url = f"http://127.0.0.1:{port}"
        # The first call captures the two B=8 shape buckets; the measured call
        # then reflects steady-state grouped endpoint behavior.
        _post_batch(base_url, requests, args.max_batch_size)
        started = time.perf_counter()
        response = _post_batch(base_url, requests, args.max_batch_size)
        wall_ms = (time.perf_counter() - started) * 1000.0
        logits = torch.cat(
            [torch.tensor(item["logits"]) for item in response["responses"]], dim=0
        )
        predictions = [
            prediction
            for item in response["responses"]
            for prediction in item["predictions"]
        ]
        health = _wait_for_health([port])[0]
        return {
            "http_requests": 1,
            "client_workers": 1,
            "request_wall_ms": wall_ms,
            "logical_requests_per_second": len(requests) / (wall_ms / 1000.0),
            "groups": response["groups"],
            "group_count": response["group_count"],
            "cache": health["cache"],
        }, logits, predictions
    finally:
        _stop_launcher(launcher)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.logical_requests < 1 or args.logical_requests % 2:
        raise ValueError("logical-requests must be a positive even number")
    if args.client_workers < 1 or args.max_batch_size < 1:
        raise ValueError("client-workers and max-batch-size must be positive")
    if len(args.sequence_lengths) != 2 or any(length < 1 for length in args.sequence_lengths):
        raise ValueError("sequence-lengths must contain two positive lengths")

    generator = SyntheticTaskGenerator(seq_len=max(args.sequence_lengths), seed=args.seed)
    records: list[tuple[str, list[int]]] = []
    for index in range(args.logical_requests):
        sequence_length = args.sequence_lengths[index % 2]
        batch = generator.task_balanced_batch(1, "cpu")
        records.append((f"s{sequence_length}", batch.inputs[0, :sequence_length].tolist()))
    requests = [
        {"inputs": [row], "return_logits": True}
        for _shape, row in records
    ]
    direct, reference_logits, reference_predictions = _run_direct(args, records)
    grouped, grouped_logits, grouped_predictions = _run_batch_endpoint(args, requests)
    grouped["max_logit_error_vs_direct"] = float(
        (grouped_logits - reference_logits).abs().max()
    )
    grouped["prediction_mismatches_vs_direct"] = sum(
        actual != expected
        for actual, expected in zip(grouped_predictions, reference_predictions)
    )
    final: dict[str, Any] = {
        "experiment": "native_fused_explicit_batch_endpoint",
        "checkpoint": str(Path(args.checkpoint)),
        "device": args.device or "auto",
        "seed": args.seed,
        "logical_requests": args.logical_requests,
        "sequence_lengths": args.sequence_lengths,
        "client_workers": args.client_workers,
        "max_batch_size": args.max_batch_size,
        "direct": direct,
        "grouped_endpoint": grouped,
        "interpretation": (
            "Both modes use the same logical requests. Direct mode sends one "
            "HTTP request per item; /infer_batch sends one request and buckets "
            "sequence lengths before native grouped inference."
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
    parser.add_argument("--sequence-lengths", nargs=2, type=int, default=[6, 32])
    parser.add_argument("--client-workers", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--no-graphs", action="store_true")
    parser.add_argument("--skip-native-prebuild", action="store_true")
    parser.add_argument("--seed", type=int, default=23006)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_batch_endpoint_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
