"""Smoke-test the real native fused HTTP serving entry point on a checkpoint."""

from __future__ import annotations

import argparse
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


def _post(base_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{base_url}/infer",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        assert response.status == 200
        return json.loads(response.read().decode("utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    service = load_native_fused_service(
        args.checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_shapes=args.max_shapes,
        warmup_iters=args.warmup_iters,
    )
    server = NativeFusedHTTPServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    generator = SyntheticTaskGenerator(
        seq_len=int(getattr(service.model, "seq_len", args.sequence_lengths[-1])),
        seed=args.seed,
    )
    result: dict[str, Any] = {
        "experiment": "native_fused_http_server_smoke",
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(service.device),
        "address": base_url,
        "batch_sizes": args.batch_sizes,
        "sequence_lengths": args.sequence_lengths,
        "requests": {},
    }
    try:
        health_request = Request(f"{base_url}/health", method="GET")
        with urlopen(health_request, timeout=30) as response:
            health = json.loads(response.read().decode("utf-8"))
        result["health"] = health
        for batch_size in args.batch_sizes:
            full_batch = generator.task_balanced_batch(batch_size, "cpu")
            for sequence_length in args.sequence_lengths:
                rows = full_batch.inputs[:, :sequence_length].tolist()
                tensor = full_batch.inputs[:, :sequence_length].to(service.device).contiguous()
                with torch.inference_mode():
                    eager, _ = service.model(
                        tensor, adaptive=False, collect_stats=False)
                first_started = time.perf_counter()
                first = _post(base_url, {"inputs": rows, "return_logits": True})
                first_ms = (time.perf_counter() - first_started) * 1000.0
                second_started = time.perf_counter()
                second = _post(base_url, {"inputs": rows, "return_logits": True})
                second_ms = (time.perf_counter() - second_started) * 1000.0
                cached_logits = torch.tensor(first["logits"], device=service.device)
                result["requests"][f"b{batch_size}_s{sequence_length}"] = {
                    "first_ms": first_ms,
                    "second_ms": second_ms,
                    "max_logit_error_vs_eager": float(
                        (cached_logits - eager).abs().max().cpu()),
                    "predictions_match": first["predictions"] == second["predictions"],
                    "cache_after_second": second["cache"],
                }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=30)
    result["final_cache"] = service.cache.stats()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 32])
    parser.add_argument("--max-shapes", type=int, default=8)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument(
        "--output",
        default="results/diagnostic_native_fused_http_server_s17_20260910.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
