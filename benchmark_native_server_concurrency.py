"""Stress-test concurrent HTTP requests against the native fused server."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.request import Request, urlopen

import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.native_fused_server import (
    NativeFusedHTTPServer,
    load_native_fused_service,
)


def _post(base_url: str, rows: list[list[int]]) -> dict[str, Any]:
    body = json.dumps({"inputs": rows, "return_logits": True}).encode("utf-8")
    request = Request(
        f"{base_url}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=180) as response:
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status {response.status}")
        return json.loads(response.read().decode("utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    service = load_native_fused_service(
        args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_shapes=args.max_shapes,
        warmup_iters=args.warmup_iters,
    )
    server = NativeFusedHTTPServer(("127.0.0.1", 0), service)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    generator = SyntheticTaskGenerator(seq_len=args.sequence_length, seed=args.seed)
    requests: list[tuple[str, list[list[int]], torch.Tensor]] = []
    shape_inputs: dict[str, tuple[list[list[int]], torch.Tensor]] = {}
    shape_specs = list(product(args.batch_sizes, args.sequence_lengths))
    for index in range(args.requests):
        batch_size, sequence_length = shape_specs[index % len(shape_specs)]
        shape = f"b{batch_size}_s{sequence_length}"
        if shape not in shape_inputs:
            batch = generator.task_balanced_batch(batch_size, "cpu")
            rows = batch.inputs[:, :sequence_length].tolist()
            tensor = batch.inputs[:, :sequence_length].to(service.device).contiguous()
            shape_inputs[shape] = (rows, tensor)
        rows, tensor = shape_inputs[shape]
        requests.append((shape, rows, tensor))

    references: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for shape, _rows, tensor in requests:
            if shape not in references:
                references[shape], _ = service.model(
                    tensor, adaptive=False, collect_stats=False)

    def call(item: tuple[str, list[list[int]], torch.Tensor]) -> dict[str, Any]:
        shape, rows, _tensor = item
        started = time.perf_counter()
        response = _post(base_url, rows)
        response["client_ms"] = (time.perf_counter() - started) * 1000.0
        response["shape_key"] = shape
        return response

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            responses = list(executor.map(call, requests))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=30)

    max_logit_error = 0.0
    prediction_mismatches = 0
    for item, response in zip(requests, responses):
        shape, _rows, _tensor = item
        logits = torch.tensor(response["logits"], device=service.device)
        max_logit_error = max(
            max_logit_error,
            float((logits - references[shape]).abs().max().cpu()),
        )
        if response["predictions"] != references[shape].argmax(dim=-1).cpu().tolist():
            prediction_mismatches += 1

    client_times = [float(response["client_ms"]) for response in responses]
    result: dict[str, Any] = {
        "experiment": "native_fused_http_server_concurrency",
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(service.device),
        "workers": args.workers,
        "requests": args.requests,
        "batch_sizes": args.batch_sizes,
        "sequence_lengths": args.sequence_lengths,
        "unique_shapes": sorted({shape for shape, _rows, _tensor in requests}),
        "client_latency_ms": {
            "min": min(client_times),
            "mean": sum(client_times) / len(client_times),
            "max": max(client_times),
        },
        "max_logit_error_vs_eager": max_logit_error,
        "prediction_mismatches": prediction_mismatches,
        "cache": service.cache.stats(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 32])
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23002)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_http_server_concurrency_s17_20260910.json",
    )
    args = parser.parse_args()
    if args.workers < 1 or args.requests < 1:
        parser.error("--workers and --requests must be positive")
    run(args)


if __name__ == "__main__":
    main()
