from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data.generator import SyntheticTaskGenerator
from neural_engine.instrumentation import estimate_neural_engine_macs
from train import load_config, make_model, seed_everything


@torch.no_grad()
def evaluate_split(model, generator: SyntheticTaskGenerator, examples_per_task: int,
                   device: torch.device) -> dict[str, Any]:
    batch = generator.balanced_batch(examples_per_task, device)
    logits, stats = model(batch.inputs, adaptive=model.adaptive_inference)
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(batch.targets)
    depth_accuracy = {}
    for depth in torch.unique(batch.depths).tolist():
        mask = batch.depths.eq(depth)
        depth_accuracy[str(int(depth))] = float(correct[mask].float().mean().cpu())
    executed = stats["executed_steps"].float()
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(nn.functional.cross_entropy(logits, batch.targets).cpu()),
        "depth_accuracy": depth_accuracy,
        "avg_executed_steps": float(executed.mean().cpu()),
        "active_step_fraction": float((executed / stats["internal_steps"].float()).mean().cpu()),
        "value_tokens": float(((batch.inputs >= 32) & (batch.inputs < 96)).sum(dim=1).float().mean().cpu()),
    }


@torch.no_grad()
def measure_latency(model, generator: SyntheticTaskGenerator, batch_size: int,
                    iterations: int, device: torch.device) -> dict[str, Any]:
    batch = generator.task_balanced_batch(batch_size, device)
    for _ in range(3):
        model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    start = time.perf_counter()
    stats = {}
    for _ in range(iterations):
        _, stats = model(batch.inputs)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    executed = stats["executed_steps"].float()
    return {
        "latency_ms_per_batch": elapsed * 1000 / iterations,
        "samples_per_second": batch_size * iterations / elapsed,
        "inference_peak_vram_mb": int(torch.cuda.max_memory_allocated(device) // (1024 * 1024))
        if device.type == "cuda" else 0,
        "benchmark_avg_executed_steps": float(executed.mean().cpu()),
        "benchmark_active_step_fraction": float((executed / stats["internal_steps"].float()).mean().cpu()),
    }


@torch.no_grad()
def analyze(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    base_config = load_config(args.config, smoke=False) if args.config else dict(payload["config"])
    if base_config["model"] == "baseline":
        raise ValueError("active budget sweep requires a Neural Engine checkpoint")
    seed_everything(int(base_config["seed"]))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "model": base_config["model"],
        "device": str(device),
        "active_circuits_tested": args.active_circuits,
        "examples_per_task": args.examples_per_task,
        "latency_batch_size": args.batch_size,
        "latency_iterations": args.iterations,
        "variants": {},
    }
    for active_circuits in args.active_circuits:
        if active_circuits > int(base_config["candidate_pool"]):
            raise ValueError("active circuit budget cannot exceed candidate_pool")
        config = dict(base_config)
        config["active_circuits"] = active_circuits
        model = make_model(config).to(device).eval()
        model.load_state_dict(payload.get("model_state", payload))
        checkpoint_report = payload.get("report", {})
        if "routing_capacity" in checkpoint_report or "routing_depth" in checkpoint_report:
            model.router.set_routing_state(
                capacity=int(checkpoint_report.get("routing_capacity", model.router.routing_capacity)),
                depth=int(checkpoint_report.get("routing_depth", model.router.active_depth)),
            )
        full_generator = SyntheticTaskGenerator(
            config["seq_len"], seed=args.seed,
            value_min=int(config.get("eval_value_min", 0)),
            value_max=int(config.get("eval_value_max", 63)),
            split=str(config.get("eval_split", "all")),
        )
        heldout_generator = SyntheticTaskGenerator(
            config["seq_len"], seed=args.seed,
            value_min=int(config.get("eval_value_min", 0)),
            value_max=int(config.get("eval_value_max", 63)),
            split="heldout",
        )
        full = evaluate_split(model, full_generator, args.examples_per_task, device)
        heldout = evaluate_split(model, heldout_generator, args.examples_per_task, device)
        latency_generator = SyntheticTaskGenerator(config["seq_len"], seed=args.seed + 1)
        latency = measure_latency(model, latency_generator, args.batch_size, args.iterations, device)
        report = model.parameter_report()
        average_active = int(report["active_params_estimate"])
        if model.adaptive_inference:
            average_active = (report["active_params_estimate"] - report["active_circuit_params"]
                              + report["active_circuit_params"] * full["avg_executed_steps"])
        compute = estimate_neural_engine_macs(model, full["avg_executed_steps"], full["value_tokens"])
        result["variants"][str(active_circuits)] = {
            "active_circuits": active_circuits,
            "total_params": report["total_params"],
            "unique_active_params": report["active_params_estimate"],
            "unique_active_fraction": report["active_fraction"],
            "average_active_params": average_active,
            "average_active_fraction": average_active / report["total_params"],
            "full": full,
            "heldout": heldout,
            "latency": latency,
            "compute": compute,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep active circuit budgets on one trained checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--active-circuits", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1712)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rendered = json.dumps(analyze(args), indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
