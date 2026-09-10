"""Smoke-test same-stream concurrent callers on a real native checkpoint."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.native_fused_serving import NativeFusedShapeCache
from train import make_model, seed_everything


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("concurrency benchmark requires CUDA")
    device = torch.device("cuda")
    checkpoint_path = Path(args.checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    config["circuit_dispatch_backend"] = "native_cuda_fused"
    seed_everything(int(config["seed"]))
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    inputs = SyntheticTaskGenerator(
        seq_len=int(config["seq_len"]), seed=int(config["seed"]) + 24000
    ).task_balanced_batch(args.examples_per_task, device).inputs
    inputs = inputs[:, :args.sequence_length].contiguous()
    cache = NativeFusedShapeCache(model, max_shapes=2, warmup_iters=args.warmup_iters)
    with torch.inference_mode():
        eager, _ = model(inputs, adaptive=False, collect_stats=False)
    cache(inputs)  # Capture before concurrent hits.

    def request() -> torch.Tensor:
        return cache(inputs.clone())

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outputs = list(executor.map(lambda _index: request(), range(args.requests)))
    torch.cuda.synchronize(device)
    errors = [float((output - eager).abs().max().cpu()) for output in outputs]
    result: dict[str, Any] = {
        "experiment": "native_fused_serving_concurrency",
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "batch_size": int(inputs.shape[0]),
        "sequence_length": int(inputs.shape[1]),
        "workers": args.workers,
        "requests": args.requests,
        "max_logit_error_vs_eager": max(errors),
        "all_outputs_match_reference": bool(all(error <= 1e-5 for error in errors)),
        "cache": cache.stats(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--examples-per-task", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--output", default="results/diagnostic_native_fused_serving_concurrency_20260910.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
