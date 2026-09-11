from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from data.composition import OPERATIONS
from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import evaluate, make_model


def make_generator(
    config: dict[str, Any],
    *,
    seed: int,
    split: str,
    value_min: int,
    value_max: int,
    operation: str | None = None,
) -> DynamicCompositionGenerator:
    modulus_config = config.get("generator_modulus", config.get("modulus", 64))
    modulus = None if modulus_config is None else int(modulus_config)
    generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=seed,
        value_min=value_min,
        value_max=value_max,
        split=split,
        modulus=modulus,
        target_offset=int(config.get("target_offset", 0)),
    )
    if operation is not None:
        generator.operation_names = (operation,)
    return generator


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ranges = {
        f"range_{value_min}_{value_max}": (value_min, value_max)
        for value_min, value_max in args.value_range
    }
    results: dict[str, Any] = {}
    for range_name, (value_min, value_max) in ranges.items():
        for split in ("train", "heldout"):
            for operation in (None, *OPERATIONS):
                operation_name = "all_operations" if operation is None else operation
                key = f"{range_name}/{split}/{operation_name}"
                generator = make_generator(
                    config,
                    seed=args.seed + len(results) + 1,
                    split=split,
                    value_min=value_min,
                    value_max=value_max,
                    operation=operation,
                )
                report = evaluate(
                    model,
                    generator,
                    device,
                    args.examples_per_depth,
                    compact_factorized=bool(config.get("compact_factorized_eval", False)),
                )
                report.pop("route_audit", None)
                report["value_range"] = [value_min, value_max]
                report["operation"] = operation_name
                report["split"] = split
                results[key] = report

    output = {
        "checkpoint": str(args.checkpoint),
        "model_name": config["model"],
        "device": str(device),
        "examples_per_depth": args.examples_per_depth,
        "value_ranges": [list(pair) for pair in args.value_range],
        "random_baseline": 1.0 / int(config["num_classes"]),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose depth and unseen-value generalization of a trained dynamic model"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/runs/ne_dynamic_generalization_diagnostic.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1700)
    parser.add_argument("--examples-per-depth", type=int, default=512)
    parser.add_argument(
        "--value-range", dest="value_range", action="append", nargs=2,
        type=int, metavar=("MIN", "MAX"),
        default=((0, 31), (32, 63)),
        help="inclusive operand range; repeat to evaluate multiple ranges",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
