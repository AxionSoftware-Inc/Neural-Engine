"""Paired circuit-residual ablation for the exact integer codec path."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

from data.composition import OPERATIONS
from diagnose_dynamic_generalization import make_generator
from train_dynamic_composition import evaluate, make_model


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    results = {}
    for checkpoint_path in args.checkpoint:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = make_model(payload["config"]).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        checkpoint_results = {}
        for operation_index, operation in enumerate((None, *OPERATIONS)):
            generator = make_generator(
                payload["config"],
                seed=args.seed + operation_index,
                split="heldout",
                value_min=args.value_min,
                value_max=args.value_max,
                operation=operation,
            )
            rng_state = copy.deepcopy(generator.rng.bit_generator.state)
            model.circuit_residual_scale = args.enabled_scale
            enabled = evaluate(
                model,
                generator,
                device,
                args.examples_per_depth,
                compact_factorized=True,
            )
            generator.rng.bit_generator.state = rng_state
            model.circuit_residual_scale = 0.0
            disabled = evaluate(
                model,
                generator,
                device,
                args.examples_per_depth,
                compact_factorized=True,
            )
            name = "all_operations" if operation is None else operation
            checkpoint_results[name] = {
                "enabled_accuracy": enabled["accuracy"],
                "disabled_accuracy": disabled["accuracy"],
                "delta_pp": 100.0 * (disabled["accuracy"] - enabled["accuracy"]),
                "enabled_accuracy_by_depth": enabled["accuracy_by_depth"],
                "disabled_accuracy_by_depth": disabled["accuracy_by_depth"],
            }
        results[str(checkpoint_path)] = checkpoint_results
    report = {
        "value_range": [args.value_min, args.value_max],
        "examples_per_depth": args.examples_per_depth,
        "enabled_scale": args.enabled_scale,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a paired circuit-residual ablation on the integer codec"
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=5000)
    parser.add_argument("--examples-per-depth", type=int, default=256)
    parser.add_argument("--value-min", type=int, default=0)
    parser.add_argument("--value-max", type=int, default=95)
    parser.add_argument("--enabled-scale", type=float, default=1.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
