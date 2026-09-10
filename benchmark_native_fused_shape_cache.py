"""Validate the native fused shape-cache caller on a real 500M checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.native_fused_serving import NativeFusedShapeCache
from train import make_model, seed_everything


def _timed(call, iterations: int, device: torch.device) -> float:
    with torch.inference_mode():
        for _ in range(5):
            call()
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iterations):
            call()
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iterations


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("shape-cache benchmark requires CUDA")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["circuit_dispatch_backend"] = "native_cuda_fused"
    seed_everything(int(config["seed"]))
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    generator = SyntheticTaskGenerator(
        seq_len=int(config["seq_len"]), seed=int(config["seed"]) + 23000)
    full_batch = generator.task_balanced_batch(args.batch_size, device)
    requests = {
        f"b{args.batch_size}_s{length}": full_batch.inputs[:, :length].contiguous()
        for length in args.sequence_lengths
    }
    cache = NativeFusedShapeCache(
        model, max_shapes=args.max_shapes, warmup_iters=args.warmup_iters)
    result: dict[str, Any] = {
        "experiment": "native_fused_shape_cache",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "sequence_lengths": args.sequence_lengths,
        "max_shapes": args.max_shapes,
        "warmup_iters": args.warmup_iters,
        "iterations": args.iterations,
        "requests": {},
    }
    for key, inputs in requests.items():
        with torch.inference_mode():
            eager, _ = model(inputs, adaptive=False, collect_stats=False)
            first = cache(inputs)
            second = cache(inputs.clone())
        result["requests"][key] = {
            "max_first_error_vs_eager": float((first - eager).abs().max().cpu()),
            "max_reuse_error": float((second - first).abs().max().cpu()),
            "eager_ms": _timed(
                lambda: model(inputs, adaptive=False, collect_stats=False),
                args.iterations, device),
            "cached_ms": _timed(lambda: cache(inputs), args.iterations, device),
        }
    result["cache"] = cache.stats()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[6, 32])
    parser.add_argument("--max-shapes", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", default="results/diagnostic_native_fused_shape_cache_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
