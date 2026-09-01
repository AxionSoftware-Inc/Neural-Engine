from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.cache import CircuitRowCache
from neural_engine.instrumentation import count_parameters
from train import load_config, make_model, seed_everything


def run_cache_size(config: dict[str, Any], checkpoint: str, cache_size: int,
                   device: torch.device, batch_size: int, iterations: int,
                   warmup: int) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = make_model(config)
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    model.circuits.to("cpu")
    cache = CircuitRowCache(model.circuits, cache_size, device)
    model.circuits.set_cache(cache)
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batches = [generator.task_balanced_batch(batch_size, device)
               for _ in range(warmup + iterations)]
    with torch.inference_mode():
        for batch in batches[:warmup]:
            model(batch.inputs)
    cache.reset_metrics(clear_cache=False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    executed_steps = []
    with torch.inference_mode():
        for batch in batches[warmup:]:
            _, stats = model(batch.inputs)
            if "executed_steps" in stats:
                executed_steps.append(float(stats["executed_steps"].float().mean().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "cache_size_rows": cache_size,
        "batch_size": batch_size,
        "iterations": iterations,
        "latency_ms_per_batch": elapsed * 1000.0 / iterations,
        "samples_per_second": batch_size * iterations / max(elapsed, 1e-9),
        "peak_vram_mb": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
        if device.type == "cuda" else None,
        "total_params": count_parameters(model),
        "circuit_bank_device": str(model.circuits.down.device),
        "resident_rows": cache.resident_rows,
        "cache_requests": cache.requests,
        "requested_unique_rows": cache.requested_rows,
        "cache_hit_rows": cache.hit_rows,
        "cache_miss_rows": cache.miss_rows,
        "cache_hit_rate": cache.hit_rate,
        "evictions": cache.evictions,
        "h2d_bytes": cache.h2d_bytes,
        "h2d_mb": cache.h2d_bytes / (1024 * 1024),
        "avg_executed_steps": sum(executed_steps) / len(executed_steps)
        if executed_steps else None,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure CPU-RAM circuit paging and cache behavior")
    parser.add_argument("--checkpoint", default="results/checkpoints/ne100_v12_coverage_full.pt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--cache-sizes", nargs="+", type=int, default=[0, 512, 2048, 7552])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"]) if args.config is None else load_config(args.config, False)
    seed_everything(int(config["seed"]))
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    results = [run_cache_size(config, args.checkpoint, cache_size, device,
                              args.batch_size, args.iterations, args.warmup)
               for cache_size in args.cache_sizes]
    report = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "cache_sizes": args.cache_sizes,
        "results": results,
    }
    serialized = json.dumps(report, indent=2)
    print(serialized)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding="utf-8")


if __name__ == "__main__":
    main()
