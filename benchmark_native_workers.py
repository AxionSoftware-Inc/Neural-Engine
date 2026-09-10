"""Exercise round-robin client admission across independent native workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch

from data.generator import SyntheticTaskGenerator


def _free_consecutive_ports(count: int) -> int:
    for _ in range(20):
        sockets = [socket.socket() for _ in range(count)]
        try:
            for sock in sockets:
                sock.bind(("127.0.0.1", 0))
            ports = [sock.getsockname()[1] for sock in sockets]
            if ports == list(range(ports[0], ports[0] + count)):
                return ports[0]
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("could not reserve consecutive ports")


def _get_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=3) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(url: str, rows: list[list[int]]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps({"inputs": rows, "return_logits": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError(f"unexpected HTTP status {response.status}")
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def _wait_for_health(ports: list[int], timeout: float = 45.0) -> list[dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health: list[dict[str, Any]] = []
        for port in ports:
            try:
                health.append(_get_json(f"http://127.0.0.1:{port}/health"))
            except Exception:
                break
        if len(health) == len(ports):
            return health
        time.sleep(0.2)
    raise RuntimeError("not all native workers became healthy")


def _stop_launcher(launcher: subprocess.Popen[str]) -> None:
    """Stop the launcher and descendants; Windows terminate() is not recursive."""

    if launcher.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(launcher.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        launcher.terminate()
    try:
        launcher.wait(timeout=15)
    except subprocess.TimeoutExpired:
        launcher.kill()
        launcher.wait(timeout=5)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.workers < 1 or args.requests < 1:
        raise ValueError("workers and requests must be positive")
    if args.requests < args.workers:
        raise ValueError("requests must be at least workers")
    base_port = args.port or _free_consecutive_ports(args.workers)
    if base_port + args.workers - 1 > 65535:
        raise ValueError("worker port range exceeds 65535")
    ports = [base_port + index for index in range(args.workers)]
    root = Path(__file__).resolve().parent
    command = [
        sys.executable, str(root / "serve_native_workers.py"),
        "--checkpoint", str(Path(args.checkpoint)),
        "--workers", str(args.workers), "--host", "127.0.0.1",
        "--port", str(base_port), "--max-shapes", str(args.max_shapes),
        "--warmup-iters", str(args.warmup_iters),
    ]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.no_graphs:
        command.append("--no-graphs")
    if args.skip_native_prebuild:
        command.append("--skip-native-prebuild")
    launcher = subprocess.Popen(
        command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        before = _wait_for_health(ports)
        generator = SyntheticTaskGenerator(seq_len=args.sequence_length, seed=args.seed)
        shape_inputs: dict[str, list[list[int]]] = {}
        requests: list[tuple[int, str, list[list[int]]]] = []
        shape_specs = list(product(args.batch_sizes, args.sequence_lengths))
        for index in range(args.requests):
            batch_size, sequence_length = shape_specs[index % len(shape_specs)]
            shape = f"b{batch_size}_s{sequence_length}"
            if shape not in shape_inputs:
                batch = generator.task_balanced_batch(batch_size, "cpu")
                shape_inputs[shape] = batch.inputs[:, :sequence_length].tolist()
            # Complete one shape round on each worker before repeating. This
            # makes cross-worker parity a real check rather than merely a
            # same-worker cache-reuse check.
            worker_index = (index // len(shape_specs)) % args.workers
            requests.append((worker_index, shape, shape_inputs[shape]))

        def request(item: tuple[int, str, list[list[int]]]) -> tuple[int, str, dict[str, Any]]:
            worker_index, shape, rows = item
            response = _post_json(f"http://127.0.0.1:{ports[worker_index]}/infer", rows)
            return worker_index, shape, response

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            responses = list(executor.map(request, requests))
        references: dict[str, torch.Tensor] = {}
        max_cross_worker_error = 0.0
        prediction_mismatches = 0
        for _worker_index, shape, response in responses:
            logits = torch.tensor(response["logits"])
            if shape not in references:
                references[shape] = logits
            else:
                max_cross_worker_error = max(
                    max_cross_worker_error,
                    float((logits - references[shape]).abs().max()),
                )
            if response["predictions"] != references[shape].argmax(dim=-1).tolist():
                prediction_mismatches += 1
        after = _wait_for_health(ports)
        result: dict[str, Any] = {
            "experiment": "native_fused_multi_worker_round_robin",
            "checkpoint": str(Path(args.checkpoint)),
            "device": args.device or "auto",
            "workers": args.workers,
            "ports": ports,
            "requests": args.requests,
            "batch_sizes": args.batch_sizes,
            "sequence_lengths": args.sequence_lengths,
            "unique_shapes": sorted(shape_inputs),
            "worker_pids": [item["pid"] for item in after],
            "worker_health_before": before,
            "worker_health_after": after,
            "max_cross_worker_logit_error": max_cross_worker_error,
            "prediction_mismatches": prediction_mismatches,
            "worker_cache": [item["cache"] for item in after],
        }
    finally:
        _stop_launcher(launcher)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--port", type=int, default=0)
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
        default="results/diagnostic_native_fused_multi_worker_s17_20260910.json",
    )
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
