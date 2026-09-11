from __future__ import annotations

import argparse
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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = make_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    outputs = {}
    for scale in args.scale:
        model.algebraic_integer_state_read_scale = float(scale)
        scale_results = {}
        for seed in (17, 18):
            for operation in (None, *OPERATIONS):
                name = "all_operations" if operation is None else operation
                generator = make_generator(
                    config,
                    seed=1700 + seed + len(scale_results),
                    split="heldout",
                    value_min=args.value_min,
                    value_max=args.value_max,
                    operation=operation,
                )
                scale_results[f"seed{seed}/{name}"] = evaluate(
                    model,
                    generator,
                    device,
                    args.examples_per_depth,
                    compact_factorized=bool(
                        config.get("compact_factorized_eval", False)
                    ),
                )
        outputs[str(scale)] = scale_results
    report = {
        "checkpoint": str(args.checkpoint),
        "value_range": [args.value_min, args.value_max],
        "examples_per_depth": args.examples_per_depth,
        "scales": args.scale,
        "results": outputs,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe exact integer state injection into the recurrent query"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples-per-depth", type=int, default=256)
    parser.add_argument("--value-min", type=int, default=0)
    parser.add_argument("--value-max", type=int, default=95)
    parser.add_argument("--scale", type=float, action="append", default=None)
    args = parser.parse_args()
    if args.scale is None:
        args.scale = [0.0]
    run(args)


if __name__ == "__main__":
    main()
