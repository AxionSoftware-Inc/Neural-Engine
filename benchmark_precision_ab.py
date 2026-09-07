"""Compare float32 matmul precision modes on one fixed Native batch."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def measure(model, inputs: torch.Tensor, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(20):
            model(inputs, collect_stats=False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            model(inputs, collect_stats=False)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this A/B requires CUDA")
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    inputs = generator.task_balanced_batch(args.batch_size, device).inputs
    torch.set_float32_matmul_precision("highest")
    with torch.inference_mode():
        reference, _ = model(inputs, collect_stats=False)
    highest_ms = measure(model, inputs, args.iterations)
    torch.set_float32_matmul_precision("high")
    with torch.inference_mode():
        high, _ = model(inputs, collect_stats=False)
    high_ms = measure(model, inputs, args.iterations)
    difference = (high - reference).abs()
    print({
        "batch_size": args.batch_size,
        "highest_ms": highest_ms,
        "high_ms": high_ms,
        "speed_ratio_high_over_highest": high_ms / highest_ms,
        "max_logit_abs_error": difference.max().item(),
        "mean_logit_abs_error": difference.mean().item(),
    })


if __name__ == "__main__":
    main()
