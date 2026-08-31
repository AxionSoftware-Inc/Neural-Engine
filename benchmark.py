from __future__ import annotations

import argparse
import json
import time

import torch

from data.generator import SyntheticTaskGenerator
from train import load_config, make_model, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference latency and routing statistics")
    parser.add_argument("--model", choices=("ne", "baseline"), default="ne")
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path = args.config or ("configs/ne_v0.yaml" if args.model == "ne" else "configs/transformer_30m.yaml")
    config = load_config(config_path, args.smoke)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    batch = SyntheticTaskGenerator(config["seq_len"], seed=int(config["seed"]) + 9).batch(args.batch_size, device)
    for _ in range(3):
        model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        _, stats = model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "model": args.model,
        "device": str(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "latency_ms_per_batch": elapsed * 1000 / args.iterations,
        "samples_per_second": args.batch_size * args.iterations / elapsed,
    }
    if stats:
        result["router_entropy"] = float(stats["router_entropy"].cpu())
        result["active_circuits"] = int(stats["active_circuits"].cpu())
        result["internal_steps"] = int(stats["internal_steps"].cpu())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
