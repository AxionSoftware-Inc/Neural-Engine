"""Evaluate existing Native checkpoints on fixed operation families.

This is deliberately inference-only.  It keeps the value range, target offset,
held-out depths, and sample count fixed while replacing the random operation
sequence with homogeneous add, subtract, or multiply programs.  The result is
useful for separating a multiply dataflow change from a general transition or
training regression.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from data.composition import OPERATION_TOKENS, apply_operation
from data.dynamic_composition import PROGRAM_TOKEN
from data.generator import Batch, VALUE_TOKEN_OFFSET
from train_dynamic_composition import factorized_digit_loss, make_model, seed_everything


DEFAULT_CHECKPOINTS = (
    "results/checkpoints/v0_288_base16_curriculum_2x2_typed_plain_seed17_5000.pt",
    "results/checkpoints/v0_288_base16_curriculum_2x2_typed_plain_seed18_5000.pt",
    "results/checkpoints/v0_290_base16_numeric_multiply_convolution_seed17_5000.pt",
    "results/checkpoints/v0_290_base16_numeric_multiply_convolution_seed18_5000.pt",
    "results/checkpoints/v0_291_base16_numeric_multiply_convolution_curriculum_seed17_5000.pt",
    "results/checkpoints/v0_291_base16_numeric_multiply_convolution_curriculum_seed18_5000.pt",
)
OPERATIONS = tuple(OPERATION_TOKENS)


def fixed_operation_batch(
    config: dict[str, Any],
    operation: str,
    depth: int,
    count: int,
    seed: int,
    device: torch.device,
) -> Batch:
    """Build a deterministic batch of homogeneous non-modular programs."""
    if operation not in OPERATIONS:
        raise ValueError(f"unknown operation: {operation}")
    max_ops = int(config["max_ops"])
    train_max_ops = int(config.get("train_max_ops", max_ops))
    if depth < 1 or depth > max_ops:
        raise ValueError("depth must be within max_ops")
    if depth <= train_max_ops:
        raise ValueError("operation-wise diagnostic expects held-out depth")
    value_min = int(config.get("operationwise_value_min", 0))
    value_max = int(config.get("operationwise_value_max", 95))
    target_offset = int(config.get("target_offset", 0))
    seq_len = int(config["seq_len"])
    rng = np.random.default_rng(seed)
    operation_token = OPERATION_TOKENS[operation]
    rows: list[list[int]] = []
    targets: list[int] = []
    stage_targets: list[list[int]] = []
    stage_mask: list[list[bool]] = []
    task_ids: list[int] = []
    depths: list[int] = []
    operation_id = OPERATIONS.index(operation)
    for _ in range(count):
        values = rng.integers(value_min, value_max + 1, size=depth + 1).tolist()
        accumulator = int(values[0])
        stages: list[int] = []
        for value in values[1:]:
            accumulator = apply_operation(
                operation, accumulator, int(value), modulus=None
            )
            stages.append(accumulator + target_offset)
        stages.extend([accumulator + target_offset] * (max_ops - depth))
        operations = [operation_token] * depth + [0] * (max_ops - depth)
        value_tokens = [VALUE_TOKEN_OFFSET + int(value) for value in values]
        value_tokens.extend([0] * (max_ops - depth))
        tokens = [PROGRAM_TOKEN] + operations + value_tokens
        if len(tokens) != seq_len:
            raise ValueError(
                f"generated sequence has length {len(tokens)}, expected {seq_len}"
            )
        rows.append(tokens)
        targets.append(accumulator + target_offset)
        stage_targets.append(stages)
        stage_mask.append([True] * depth + [False] * (max_ops - depth))
        task_ids.append(operation_id)
        depths.append(depth)
    return Batch(
        inputs=torch.tensor(rows, dtype=torch.long, device=device),
        targets=torch.tensor(targets, dtype=torch.long, device=device),
        task_ids=torch.tensor(task_ids, dtype=torch.long, device=device),
        depths=torch.tensor(depths, dtype=torch.long, device=device),
        stage_targets=torch.tensor(stage_targets, dtype=torch.long, device=device),
        stage_mask=torch.tensor(stage_mask, dtype=torch.bool, device=device),
    )


@torch.no_grad()
def evaluate_fixed_batch(
    model: torch.nn.Module, batch: Batch, digit_base: int
) -> dict[str, Any]:
    model.eval()
    _, stats = model(batch.inputs, return_full_logits=False)
    digit_logits = stats["digit_logits"]
    predictions = torch.zeros_like(batch.targets)
    predicted_digits = []
    target_digits = []
    for index, logits in enumerate(digit_logits):
        power = digit_base ** (len(digit_logits) - 1 - index)
        digits = logits[:, -1].argmax(dim=-1)
        predictions = predictions + digits * power
        predicted_digits.append(digits)
        target_digit = batch.targets // power
        if index:
            target_digit = target_digit.remainder(digit_base)
        target_digits.append(target_digit)
    correct = predictions.eq(batch.targets)
    digit_accuracy = [
        float(predicted.eq(target).float().mean().cpu())
        for predicted, target in zip(predicted_digits, target_digits)
    ]
    selected = stats["selected_ids"].reshape(-1)
    selected = selected[selected.ge(0)]
    return {
        "count": int(batch.targets.shape[0]),
        "accuracy": float(correct.float().mean().cpu()),
        "loss": float(factorized_digit_loss(
            digit_logits, batch.targets, digit_base
        ).cpu()),
        "digit_accuracy": digit_accuracy,
        "router_entropy": float(stats["router_entropy"].cpu()),
        "unique_virtual_circuits": int(selected.unique().numel()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    checkpoint_paths = tuple(args.checkpoint or DEFAULT_CHECKPOINTS)
    records: list[dict[str, Any]] = []
    for checkpoint_path in checkpoint_paths:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = payload["config"]
        checkpoint_seed = int(payload.get("report", {}).get("seed", config.get("seed", 0)))
        model = make_model(config).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for operation_index, operation in enumerate(OPERATIONS):
            for depth in (int(config["train_max_ops"]) + 1, int(config["max_ops"])):
                batch = fixed_operation_batch(
                    config,
                    operation,
                    depth,
                    args.examples_per_case,
                    seed=10_000 + checkpoint_seed * 100 + operation_index * 10 + depth,
                    device=device,
                )
                metrics = evaluate_fixed_batch(
                    model, batch, int(config["output_digit_base"])
                )
                records.append({
                    "checkpoint": checkpoint_path,
                    "variant": Path(checkpoint_path).stem.rsplit("_seed", 1)[0],
                    "seed": checkpoint_seed,
                    "operation": operation,
                    "depth": depth,
                    **metrics,
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "benchmark": "operationwise_fixed_checkpoint_eval",
        "device": str(device),
        "examples_per_case": args.examples_per_case,
        "operations": list(OPERATIONS),
        "records": records,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Native checkpoints by fixed operation family"
    )
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--examples-per-case", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--output",
        default="results/operationwise_fixed_checkpoint_eval_v0_292.json",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
