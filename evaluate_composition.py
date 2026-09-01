from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import torch

from data.composition import OPERATION_TOKENS, VALUE_TOKEN_OFFSET, apply_operation
from train import make_model


def grid_rows(config: dict, grid_size: int) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if not 1 <= grid_size <= 64:
        raise ValueError("grid_size must be between 1 and 64")
    heldout_pairs = tuple(tuple(pair) for pair in config.get("heldout_pairs", []))
    pairs = heldout_pairs or tuple(
        (first, second) for first in OPERATION_TOKENS for second in OPERATION_TOKENS
    )
    seq_len = int(config["seq_len"])
    if seq_len < 6:
        raise ValueError("seq_len must leave room for program and operands")
    inputs, targets, labels = [], [], []
    for first, second in pairs:
        for a, b, c in product(range(grid_size), repeat=3):
            partial = apply_operation(first, a, b)
            target = apply_operation(second, partial, c)
            tokens = [1, OPERATION_TOKENS[first], OPERATION_TOKENS[second],
                      VALUE_TOKEN_OFFSET + a, VALUE_TOKEN_OFFSET + b,
                      VALUE_TOKEN_OFFSET + c]
            tokens.extend([0] * (seq_len - len(tokens)))
            inputs.append(tokens)
            targets.append(target)
            labels.append(f"{first}_then_{second}")
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long), labels)


@torch.no_grad()
def evaluate(checkpoint: str, grid_size: int, batch_size: int,
             device_name: str) -> dict:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    device = torch.device(device_name)
    model = make_model(config).to(device).eval()
    model.load_state_dict(payload["model_state"])
    inputs, targets, labels = grid_rows(config, grid_size)
    correct_parts = []
    for start in range(0, len(targets), batch_size):
        logits, _ = model(inputs[start:start + batch_size].to(device),
                          adaptive=model.adaptive_inference)
        correct_parts.append(logits.argmax(dim=-1).cpu().eq(targets[start:start + batch_size]))
    correct = torch.cat(correct_parts)
    per_pair = {}
    for label in dict.fromkeys(labels):
        mask = torch.tensor([item == label for item in labels])
        per_pair[label] = float(correct[mask].float().mean())
    result = {
        "checkpoint": checkpoint,
        "total_params": sum(parameter.numel() for parameter in model.parameters()),
        "grid_size": grid_size,
        "examples_per_pair": grid_size ** 3,
        "accuracy": float(correct.float().mean()),
        "per_pair_accuracy": per_pair,
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a composition checkpoint on a deterministic operand grid")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.grid_size, args.batch_size, args.device)


if __name__ == "__main__":
    main()
