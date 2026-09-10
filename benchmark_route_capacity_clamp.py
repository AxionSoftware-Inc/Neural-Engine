from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from analyze_routes import build_task_batch
from data.generator import SyntheticTaskGenerator
from train import make_model, seed_everything


def evaluate_mode(model: nn.Module, batch: Any, mode: str) -> dict[str, Any]:
    """Evaluate a frozen checkpoint with an optional reachable-bank prefix."""
    router = getattr(model, "router", None)
    if router is None or not hasattr(router, "set_routing_state"):
        raise TypeError("route capacity clamp requires a routed Neural Engine")

    if mode == "natural":
        capacity = int(router.routing_capacity)
        depth = int(router.active_depth)
    elif mode == "prefix_7552_depth5":
        capacity = min(7552, int(router.num_circuits))
        depth = min(5, int(router.depth))
    elif mode == "prefix_1408_depth4":
        capacity = min(1408, int(router.num_circuits))
        depth = min(4, int(router.depth))
    else:
        raise ValueError(f"unknown mode: {mode}")
    router.set_routing_state(capacity=capacity, depth=depth)

    logits, stats = model(batch.inputs, adaptive=model.adaptive_inference)
    predictions = logits.argmax(dim=-1)
    selected = stats["selected_ids"]
    flat_selected = selected[selected.ge(0)]
    counts = torch.bincount(flat_selected, minlength=router.num_circuits).float()
    return {
        "mode": mode,
        "routing_capacity": capacity,
        "routing_depth": depth,
        "accuracy": float(predictions.eq(batch.targets).float().mean().cpu()),
        "cross_entropy": float(nn.functional.cross_entropy(logits, batch.targets).cpu()),
        "avg_executed_steps": float(stats["executed_steps"].float().mean().cpu()),
        "circuits_used": int(counts.gt(0).sum()),
        "dead_circuit_fraction_full_bank": float((counts.eq(0)).float().mean()),
        "used_fraction_within_reachable_prefix": float(
            counts[:capacity].gt(0).float().mean()
        ),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    if config.get("model") == "baseline":
        raise ValueError("route capacity clamp requires a Neural Engine checkpoint")
    seed_everything(int(config["seed"]))
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload.get("model_state", payload))
    generator = SyntheticTaskGenerator(
        int(config["seq_len"]), seed=args.seed,
        value_min=int(config.get("eval_value_min", 0)),
        value_max=int(config.get("eval_value_max", 63)),
        split=str(config.get("eval_split", "all")),
    )
    batch = build_task_batch(generator, args.examples_per_task, device)
    modes = ("natural", "prefix_7552_depth5", "prefix_1408_depth4")
    results = [evaluate_mode(model, batch, mode) for mode in modes]
    return {
        "checkpoint": str(args.checkpoint),
        "seed": int(config["seed"]),
        "model": config["model"],
        "device": str(device),
        "examples_per_task": int(args.examples_per_task),
        "evaluator_seed": int(args.seed),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen whether a frozen large bank benefits from a smaller reachable prefix"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-task", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1712)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
