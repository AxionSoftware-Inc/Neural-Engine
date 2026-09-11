"""Probe unseen values at depth three without overflowing the class head."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.composition import OPERATIONS
from data.dynamic_composition import DynamicCompositionGenerator
from train_dynamic_composition import make_model


@torch.no_grad()
def run(args: argparse.Namespace) -> dict:
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    results = {}
    for seed, checkpoint_path in enumerate(args.checkpoint, start=17):
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = payload["config"]
        model = make_model(config).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        operation_results = {}
        for operation in (None, *OPERATIONS):
            generator = DynamicCompositionGenerator(
                max_ops=3,
                train_max_ops=2,
                seed=args.seed + seed,
                value_min=args.value_min,
                value_max=args.value_max,
                split="heldout",
                modulus=None,
                target_offset=int(config.get("target_offset", 0)),
            )
            if operation is not None:
                generator.operation_names = (operation,)
            batch = generator.balanced_batch(args.examples_per_depth, device)
            # The model has max_ops=4 while this probe intentionally uses only
            # three operations. Pad the operation and value regions separately
            # so the model's fixed layout remains valid.
            zeros = torch.zeros(
                (batch.inputs.shape[0], 1), dtype=batch.inputs.dtype, device=device
            )
            inputs = torch.cat(
                (
                    batch.inputs[:, :1],
                    batch.inputs[:, 1:4],
                    zeros,
                    batch.inputs[:, 4:8],
                    zeros,
                ),
                dim=1,
            )
            _, stats = model(inputs, return_full_logits=False)
            predictions = torch.zeros_like(batch.targets)
            for index, digit_logits in enumerate(stats["digit_logits"]):
                predictions += digit_logits[:, -1].argmax(dim=-1) * (
                    model.output_digit_base ** (model.output_digit_count - 1 - index)
                )
            correct = predictions.eq(batch.targets)
            name = "all_operations" if operation is None else operation
            operation_results[name] = {
                "accuracy": float(correct.float().mean().cpu()),
                "min_target": int(batch.targets.min().cpu()),
                "max_target": int(batch.targets.max().cpu()),
            }
        results[str(seed)] = operation_results
    report = {
        "value_range": [args.value_min, args.value_max],
        "depth": 3,
        "examples_per_depth": args.examples_per_depth,
        "checkpoints": args.checkpoint,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe depth-3 unseen value extrapolation without class overflow"
    )
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=4100)
    parser.add_argument("--examples-per-depth", type=int, default=256)
    parser.add_argument("--value-min", type=int, default=96)
    parser.add_argument("--value-max", type=int, default=127)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
