from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from data.generator import SyntheticTaskGenerator
from neural_engine.instrumentation import estimate_neural_engine_macs, estimate_transformer_macs
from train import load_config, make_model, seed_everything


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference latency and routing statistics")
    parser.add_argument("--model", choices=("ne", "baseline"), default="ne")
    parser.add_argument("--config", default=None)
    parser.add_argument("--checkpoint", default=None,
                        help="Load model weights and, when config is omitted, its effective config")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--balanced-batch", action="store_true",
                        help="Use a near-uniform task mix for reproducible adaptive-step statistics")
    args = parser.parse_args()
    checkpoint_payload = None
    if args.checkpoint:
        checkpoint_payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
        if args.config is None:
            config = dict(checkpoint_payload["config"])
            if args.smoke:
                raise ValueError("--smoke cannot be combined with a checkpoint")
        else:
            config_path = args.config
            config = load_config(config_path, args.smoke)
    else:
        config_path = args.config or ("configs/ne_v0.yaml" if args.model == "ne" else "configs/transformer_30m.yaml")
        config = load_config(config_path, args.smoke)
    model_kind = "baseline" if config["model"] == "baseline" else "ne"
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    if checkpoint_payload is not None:
        state_dict = checkpoint_payload.get("model_state", checkpoint_payload)
        model.load_state_dict(state_dict)
    generator = SyntheticTaskGenerator(
        config["seq_len"], seed=int(config["seed"]) + 9,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    if args.balanced_batch:
        batch = generator.task_balanced_batch(args.batch_size, device)
    else:
        batch = generator.batch(args.batch_size, device)
    for _ in range(3):
        model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iterations):
        _, stats = model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "model": model_kind,
        "device": str(device),
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "latency_ms_per_batch": elapsed * 1000 / args.iterations,
        "samples_per_second": args.batch_size * args.iterations / elapsed,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "balanced_batch": bool(args.balanced_batch),
    }
    if device.type == "cuda":
        result["peak_vram_mb"] = int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
    if stats:
        result["router_entropy"] = float(stats["router_entropy"].cpu())
        result["active_circuits"] = int(stats["active_circuits"].cpu())
        result["internal_steps"] = int(stats["internal_steps"].cpu())
        if "executed_steps" in stats:
            executed = stats["executed_steps"].float()
            result["avg_executed_steps"] = float(executed.mean().cpu())
            result["active_step_fraction"] = float((executed / stats["internal_steps"].float()).mean().cpu())
            result["adaptive_inference"] = bool(getattr(model, "adaptive_inference", False))
            if model_kind == "ne":
                value_tokens = ((batch.inputs >= 32) & (batch.inputs < 96)).sum(dim=1).float().mean().item()
                result.update(estimate_neural_engine_macs(model, float(executed.mean().cpu()), value_tokens))
        elif model_kind == "ne":
            result.update(estimate_neural_engine_macs(model, float(model.internal_steps), 0.0))
    if model_kind == "baseline":
        result.update(estimate_transformer_macs(config))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
