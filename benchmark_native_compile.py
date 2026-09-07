"""Smoke-test torch.compile for Native Engine inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile CUDA benchmark is unavailable")

    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    seed_everything(int(config["seed"]))
    device = torch.device("cuda")
    eager = make_model(config).to(device).eval()
    eager.load_state_dict(payload["model_state"])
    compiled = torch.compile(eager, mode="reduce-overhead", fullgraph=False)
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = generator.task_balanced_batch(args.batch_size, device)
    with torch.inference_mode():
        eager_logits, _ = eager(batch.inputs)
        compile_start = time.perf_counter()
        compiled_logits, _ = compiled(batch.inputs)
        torch.cuda.synchronize()
        compile_seconds = time.perf_counter() - compile_start
        max_error = (compiled_logits - eager_logits).abs().max().item()
        for _ in range(10):
            eager(batch.inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            eager(batch.inputs)
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - start) * 1000 / args.iterations
        for _ in range(10):
            compiled(batch.inputs)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            compiled(batch.inputs)
        torch.cuda.synchronize()
        compiled_ms = (time.perf_counter() - start) * 1000 / args.iterations
    print({
        "batch_size": args.batch_size,
        "compile_seconds": compile_seconds,
        "max_logit_error": max_error,
        "eager_ms": eager_ms,
        "compiled_ms": compiled_ms,
        "speed_ratio_compiled_over_eager": compiled_ms / eager_ms,
    })


if __name__ == "__main__":
    main()
