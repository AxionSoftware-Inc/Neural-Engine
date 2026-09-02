"""Measure an exact modular-register control for the dynamic benchmark.

This is a diagnostic ceiling, not a learned model.  It uses a fixed transition
table for the declared mod-64 primitives and therefore must not be presented
as evidence of learned generalization.  Its purpose is to separate the
benchmark's algebraic prior from the Neural Register Machine's learned
register/circuit implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from data.composition import MODULUS, OPERATION_TOKENS, apply_operation
from data.dynamic_composition import DynamicCompositionGenerator


def make_transition_table(device: torch.device) -> torch.Tensor:
    table = torch.empty(3, MODULUS, MODULUS, dtype=torch.long, device=device)
    names = tuple(OPERATION_TOKENS)
    for op_id, name in enumerate(names):
        for left in range(MODULUS):
            for right in range(MODULUS):
                table[op_id, left, right] = apply_operation(name, left, right)
    return table


@torch.no_grad()
def predict(inputs: torch.Tensor, table: torch.Tensor, max_ops: int) -> torch.Tensor:
    value_start = 1 + max_ops
    accumulator = (inputs[:, value_start] - 32).clamp(0, MODULUS - 1)
    for step in range(max_ops):
        operation_tokens = inputs[:, 1 + step]
        active = operation_tokens.ge(2)
        if not active.any():
            continue
        operation_ids = (operation_tokens - 2).clamp(0, 2)
        operand = (inputs[:, value_start + step + 1] - 32).clamp(0, MODULUS - 1)
        updated = table[operation_ids, accumulator, operand]
        accumulator = torch.where(active, updated, accumulator)
    return accumulator


@torch.no_grad()
def evaluate(
    generator: DynamicCompositionGenerator,
    table: torch.Tensor,
    examples_per_depth: int,
    device: torch.device,
) -> dict[str, object]:
    batch = generator.balanced_batch(examples_per_depth, device)
    predictions = predict(batch.inputs, table, generator.max_ops)
    correct = predictions.eq(batch.targets)
    return {
        "accuracy": float(correct.float().mean().cpu()),
        "accuracy_by_depth": {
            str(depth): float(correct[batch.depths.eq(depth)].float().mean().cpu())
            for depth in generator.allowed_depths
        },
        "fixed_parameters": 0,
        "attention": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exact modular control")
    parser.add_argument("--max-ops", type=int, default=6)
    parser.add_argument("--train-max-ops", type=int, default=4)
    parser.add_argument("--train-value-min", type=int, default=0)
    parser.add_argument("--train-value-max", type=int, default=31)
    parser.add_argument("--eval-value-min", type=int, default=32)
    parser.add_argument("--eval-value-max", type=int, default=63)
    parser.add_argument("--examples-per-depth", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "cpu"))
    table = make_transition_table(device)
    train_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops, seed=18,
        value_min=args.train_value_min, value_max=args.train_value_max,
        split="train",
    )
    eval_generator = DynamicCompositionGenerator(
        max_ops=args.max_ops, train_max_ops=args.train_max_ops, seed=19,
        value_min=args.eval_value_min, value_max=args.eval_value_max,
        split="heldout",
    )
    result = {
        "control": "exact_modular_register",
        "device": str(device),
        "train": evaluate(train_generator, table, args.examples_per_depth, device),
        "evaluation": evaluate(eval_generator, table, args.examples_per_depth, device),
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        Path(args.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
