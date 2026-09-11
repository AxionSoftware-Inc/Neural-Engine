"""Evaluate prior-free composition checkpoints on a larger paired batch.

Each checkpoint receives the same deterministic examples per depth.  This is
an evaluation-only companion to ``train_dynamic_composition.py``; it does not
retrain or alter checkpoint weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import evaluate, make_model


def run_checkpoint(
    path: Path,
    *,
    examples_per_depth: int,
    seed: int,
    value_min: int,
    value_max: int,
    device: torch.device,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    model = make_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    modulus_config = config.get("generator_modulus", config.get("modulus", 64))
    modulus = None if modulus_config is None else int(modulus_config)
    generator = DynamicCompositionGenerator(
        max_ops=int(config["max_ops"]),
        train_max_ops=int(config.get("train_max_ops", config["max_ops"])),
        seed=seed,
        modulus=modulus,
        target_offset=int(config.get("target_offset", 0)),
        value_min=value_min,
        value_max=value_max,
        split="all",
    )
    result = evaluate(
        model,
        generator,
        device,
        examples_per_depth,
        compact_factorized=bool(config.get("compact_factorized_eval", False)),
    )
    return {
        "checkpoint": str(path),
        "model_name": config["model"],
        "seed": payload.get("report", {}).get("seed"),
        "examples_per_depth": examples_per_depth,
        "value_range": [value_min, value_max],
        "evaluation": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--examples-per-depth", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1702)
    parser.add_argument("--value-min", type=int, default=0)
    parser.add_argument("--value-max", type=int, default=95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    result = {
        "examples_per_depth": args.examples_per_depth,
        "seed": args.seed,
        "value_range": [args.value_min, args.value_max],
        "device": str(device),
        "checkpoints": [
            run_checkpoint(
                Path(path),
                examples_per_depth=args.examples_per_depth,
                seed=args.seed,
                value_min=args.value_min,
                value_max=args.value_max,
                device=device,
            )
            for path in args.checkpoint
        ],
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
